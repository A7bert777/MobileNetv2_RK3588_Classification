#include <stdint.h>  
#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <sys/stat.h> 
#include <dirent.h>  
#include "mobilenet.h"  
#include "image_utils.h"  
#include "file_utils.h"  
#include <iostream>
#include <vector> 
#include <filesystem>  
#include <chrono> 
#include <opencv2/opencv.hpp> 

void release_image_buffer(image_buffer_t* image_buffer) 
{  
    if (image_buffer != NULL) 
    {  
        // 释放图像数据  
        if (image_buffer->data != NULL) 
        {  
            free(image_buffer->data);  
            image_buffer->data = NULL;  
            image_buffer->size = 0;  
        }  
    }  
}



//去除文件地址&后缀 dd
std::string extractFileNameWithoutExtension(const std::string& path) 
{  
    auto pos = path.find_last_of("/\\");  
    std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);  
    
    // 查找并去除文件后缀  
    pos = filename.find_last_of(".");  
    if (pos != std::string::npos) {  
        filename = filename.substr(0, pos);  
    }  
    
    return filename;  
}

int main(int argc, char** argv)  
{  

    auto start = std::chrono::high_resolution_clock::now(); // 开始时间戳 


    const char* imagenet_classes_file = "/home/firefly/mobilenet_github/model/synset.txt";  
    const char* model_path = "/home/firefly/mobilenet_github/model/9_14MobileNetV2RKNN.rknn";  
    const char* image_folder = "/home/firefly/mobilenet_github/inputimage"; 
    const char* output_folder = "/home/firefly/mobilenet_github/outputimage";  

    int line_count;  
    char** lines = read_lines_from_file(imagenet_classes_file, &line_count);  
    if (lines == NULL) 
    {  
        printf("read classes label file fail! path=%s\n", imagenet_classes_file);  
        return -1;  
    }  

    int ret;  
    rknn_app_context_t rknn_app_ctx;  
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_ctx));  

    ret = init_mobilenet_model(model_path, &rknn_app_ctx);  
    if (ret != 0) 
    {  
        printf("init_mobilenet_model fail! ret=%d model_path=%s\n", ret, model_path);  
        return -1;  
    }  
    
    DIR* dir = opendir(image_folder);  
    if (dir == NULL) 
    {  
        printf("Failed to open directory: %s\n", image_folder);  
        return -1;  
    }  

    struct dirent* entry;  
    while ((entry = readdir(dir)) != NULL)
    {
        char image_path[1024];  
        snprintf(image_path, sizeof(image_path), "%s/%s", image_folder, entry->d_name);  

        // 获取不带后缀的文件名  
        std::string filename_without_extension = extractFileNameWithoutExtension(entry->d_name);  
        // 打印当前正在处理的图片名  
        std::cout << "Processing image: " << image_path << std::endl; 

        // 跳过非图片文件  
        if (strstr(entry->d_name, ".jpg") == NULL && strstr(entry->d_name, ".jpeg") == NULL && strstr(entry->d_name, ".png") == NULL) 
        {  
            continue;  
        }  
        // 处理每张图片  
        image_buffer_t src_image;  
        memset(&src_image, 0, sizeof(image_buffer_t));  
        ret = read_image(image_path, &src_image);  
        
        if (ret != 0) 
        {  
            printf("Failed to read image: %s\n", image_path);  
            continue;  
        }  

        int topk = 11; 
        mobilenet_result result[topk];  
        ret = inference_mobilenet_model(&rknn_app_ctx, &src_image, result, topk);  
        if (ret != 0) 
        {  
            printf("Failed to perform inference on image: %s\n", image_path);  
            continue;  
        }

        char output_path[256];  
        snprintf(output_path, sizeof(output_path), "%s/%s.jpg", output_folder,filename_without_extension.c_str());  
        ret = write_image(output_path, &src_image);  

        // 释放图片资源  
        release_image_buffer(&src_image); 
        std::cout<<"-------------------------------------------------------------------------------"<<std::endl;
    }
        closedir(dir);   
    
    auto end = std::chrono::high_resolution_clock::now(); // 结束时间戳 
    std::chrono::duration<double, std::milli> elapsed = end - start; // 计算经过的时间（毫秒）
    std::cout << "------------------------------------------------------------------------All time:" << elapsed.count() << " ms\n"; // 打印经过的时间 

out:  
    ret = release_mobilenet_model(&rknn_app_ctx);  
    if (ret != 0) 
    {  
        printf("release_mobilenet_model fail! ret=%d\n", ret);  
    }  

    if (lines != NULL) 
    {  
        free_lines(lines, line_count);  
    }  

    return 0;  



}
