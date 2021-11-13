#include <iostream>
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/op_resolver.h"
#include <cstdlib>
#include <fstream>
#include <ctime> 
#include <iostream>
#include <typeinfo>
#include <string>

float rand_FloatRange(float a, float b)
{   
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}




int main(int argc, char** argv)
{
    std::__cxx11::string types1[]={"kTfLiteNoType",
                "kTfLiteFloat32",
                "kTfLiteInt32",
                "kTfLiteUInt8",
                "kTfLiteInt64",
                "kTfLiteString",
                "kTfLiteBool",
                "kTfLiteInt16",
                "kTfLiteComplex64",
                "kTfLiteInt8",
                "kTfLiteFloat16",
                "kTfLiteFloat64",
                "kTfLiteComplex128",
                "kTfLiteUInt64",
                "kTfLiteResource",
                "kTfLiteVariant",
                "kTfLiteUInt32"
                };
    //check_float_int8
    const char* filename = "/home/yongkang.shu/tfconvert/model/vgg16_relu_uint.tflite";

    std::fstream myfile("/home/yongkang.shu/tfconvert/model/a.txt", std::ios_base::in);
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

//添加delegete
//     if (with_delegate) {
//     ManasiDelegateOptions option;
//     auto *delegate_ptr = TfLiteManasiDelegateCreate(&option);
//     // ASSERT_TRUE(delegate_ptr != nullptr);
//     auto delegate = Interpreter::TfLiteDelegatePtr(
//         delegate_ptr,
//         [](TfLiteDelegate *delegate) { TfLiteManasiDelegateDelete(delegate); });
//     // Add delegate.
//     EXPECT_TRUE(interpreter->ModifyGraphWithDelegate(delegate.get()) !=
//                 kTfLiteError);
//   }
    interpreter->SetNumThreads(1);
    interpreter->AllocateTensors();
    TfLiteContext*  context = interpreter->subgraph(0)->context();  //获取contex
    std::cout<<"contex的类型是"<<typeid(context).name()<<std::endl;
    std::cout<<"tensor size:"<<context->tensors_size<<std::endl;
    int node_index = 1;

    ////context->GetTensor(context, node_index);
    //获取inputtensor的name、shape、dtype信息   
    auto tensor = context->tensors[1];
    std::cout<<"tensor name:"<<tensor.name<<std::endl;
    std::cout<<"tensor dtype:"<<types1[tensor.type]<<std::endl;
     std::cout<<"tensor shape size:"<<tensor.dims->size<<std::endl;
    for(int i=0; i < tensor.dims->size; i++){
        std::cout<<"tensor shape "<<i<<" :"<<tensor.dims->data[i]<<std::endl;
    }

    TfLiteNode *node;
    TfLiteRegistration *reg;    //对应算子函数的指针
    context->GetNodeAndRegistration(context, node_index, &node, &reg);
    //TODO把tensor的信息用Var对象保存




    //const auto input_type = context->tensors[node->inputs->data[0]].type;

    
    // std::vector<int> input_uint;
    // for(int i = 0; i < 1*32*32*1; i++){
    //     unsigned char input = rand() % 100;
    //     input_uint.push_back(input);
    //     interpreter->typed_input_tensor<unsigned char>(0)[i] = input;
    //     // printf("%d \n", input);
    // }

    //float 输入
    std::ofstream float_input_file("/home/yongkang.shu/temp/input_float.txt");
    for(int i = 0; i < 32*32*64; i++){
            unsigned char input = (rand() % (200-1+1))+ 1; //rand_FloatRange(-3.0,3.0);
            // interpreter->typed_input_tensor<unsigned char>(0)[i] = input;
            myfile >> interpreter->typed_input_tensor<unsigned char>(0)[i];
            //printf("%d ", input);
            float_input_file << input << "\n";
    }
    //printf("\n");

    interpreter->Invoke();

    // //float output
    // std::ofstream float_output_file("/home/yongkang.shu/temp/output_float.txt");
    // for(int i = 0; i < 32*32*64; i++)
    // {
    //     //unsigned char output  = interpreter->typed_output_tensor< unsigned char>(0)[i];
    //     float output  = interpreter->typed_output_tensor< float>(0)[i];
    //     float_output_file << output <<"\n";
    //     //printf("%f ", (output));
    // }


    // std::ofstream output_file("/home/yongkang.shu/tensorflow-1.15.5/input.txt");
    // for (const auto &e : input_uint){
    //     output_file << e << "\n";
    // } 

    printf("\n");



} 

