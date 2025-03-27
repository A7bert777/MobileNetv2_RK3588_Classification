#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mobilenet.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"



#include <opencv2/opencv.hpp>  
#include <cstdio>  

void draw_text_to_image(cv::Mat* img, const char* text, int x, int y, unsigned char color[3]) 
{  
    // 确保图像指针非空  
    if (img == nullptr) 
    {  
        printf("Error: Image pointer is null.\n");  
        return;  
    }  

    // 将颜色数组转换为OpenCV的Scalar结构  
    cv::Scalar textColor = cv::Scalar(255, 0, 0); // OpenCV使用BGR格式  

    // 在图像上绘制文本  
    cv::putText(*img, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2);  

    // 打印信息  
    printf("Drawing text '%s' at (%d, %d) on image\n", text, x, y);  
}  

cv::Mat image_buffer_to_cv_mat(const image_buffer_t* img_buffer) 
{  
    if (img_buffer->format == IMAGE_FORMAT_RGB888) 
    {  
        cv::Mat img(img_buffer->height, img_buffer->width, CV_8UC3, img_buffer->virt_addr);  
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR); // OpenCV使用BGR格式  
        return img;  
    } 
    else 
    {  
        // 处理其他图像格式  
        return cv::Mat(); // 返回一个空的Mat对象  
    }  
}


static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

typedef struct {
    float value;
    int index;
} element_t;

static void swap(element_t* a, element_t* b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

static int partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

static void quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

static void softmax(float* array, int size) {
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}

static void get_topk_with_indices(float arr[], int size, int k, mobilenet_result* result) {

    // Create an array of elements, saving values ​​and index numbers
    element_t* elements = (element_t*)malloc(size * sizeof(element_t));
    for (int i = 0; i < size; i++) {
        elements[i].value = arr[i];
        elements[i].index = i;
    }

    // Quick sort an array of elements
    quick_sort(elements, 0, size - 1);

    // Get the top K maximum values ​​and their index numbers
    for (int i = 0; i < k; i++) {
        result[i].score = elements[i].value;
        result[i].cls = elements[i].index;
    }

    free(elements);
}

int init_mobilenet_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    char* model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_mobilenet_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_mobilenet_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, mobilenet_result* out_result, int topk)
{
    int ret;
    image_buffer_t img;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(&img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    img.format = IMAGE_FORMAT_RGB888;
    img.size = get_image_size(&img);
    img.virt_addr = (unsigned char*)malloc(img.size);
    if (img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }

    ret = convert_image(src_img, &img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf   = img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
       // goto out;
    }

    // Post Process
    softmax((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems);

    get_topk_with_indices((float*)outputs[0].buf, app_ctx->output_attrs[0].n_elems, topk, out_result);

        // 假设有五个类别 
    const char* lines[] = 
    {  
        "butterfly", "carpet", "cat", "chicken", "cow", "dog", "elephant", "horse", "sequirrel", "sheep", "spider"
    };

    // 绘制得分到图像上  
    char text[256];  
    // 格式化文本  
    snprintf(text, sizeof(text), "%s:%.6f", lines[out_result[0].cls], out_result[0].score);  
    unsigned char color[3] = {255, 0, 0}; // 红色  

    // 将图像缓冲区转换为OpenCV图像
    cv::Mat cv_img = image_buffer_to_cv_mat(src_img); 
    
    // 将OpenCV图像从BGR格式转换为RGB格式
    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB); 
    
    // 在图像上绘制文本
    draw_text_to_image(&cv_img, text, 25, 25, color);

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);

out:
    if (img.virt_addr != NULL) {
        free(img.virt_addr);
    }

    return ret;
}