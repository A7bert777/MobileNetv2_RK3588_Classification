# MobileNetv2_RK3588_Classification

1.项目代码介绍

src/main.cpp ：主程序运行文件

src/mobilenet.cpp：模型初始化、推理、反初始化等函数代码

include/mobilenet.h：各函数声明

2.配置文件介绍

3rdparty 中是第三方库

build 是编译位置

inputimage 是输入图片所在文件夹

outputimage 是输出图片所在文件夹

model 是RKNN模型以及标签名txt文件所在文件夹

rknn_lib 是瑞芯微官方动态库librknnrt.so所在位置

3.编译运行

**①cd build**

**②cmake ..**

**③make**

**④./rknn_mobilenet_demo**





CSDN地址：[【MobileNetv2图像分类部署至RK3588】模型训练→转换RKNN→开发板部署_rk3588 mobilenet-CSDN博客](https://blog.csdn.net/A_l_b_ert/article/details/142283122?spm=1001.2014.3001.5501)

QQ咨询（not free，除非你点了小星星）：2506245294