#include <thread>

#include "VideoCaptureThread.h"
#include "DisplayThread.h"
#include "InferenceThread.h"
#include "ActionThread.h"

std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("/home/jarvis/app/jarvis_yolov5/racecar.names"); //("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

int main(int argc, char** argv) {
    int size_model = 320;

    loguru::init(argc, argv);
    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
    
    string model_path = "/home/jarvis/app/self-driving-car/jarvis_prey_predator/model/best.onnx";

    Config config = {0.50f, 0.45f, model_path, "/home/jarvis/app/jarvis_yolov5/racecar.names", Size(size_model, size_model), false};
    LOG_F(INFO, "Start main process");
    
    
    //std::vector<std::string> class_list = load_class_list();

    //bool useGPU = false;
    //std::string l_GpuOption = "CPU" ; //(argv[3]);
    //std::transform(l_GpuOption.begin(), l_GpuOption.end(), l_GpuOption.begin(), [](unsigned char c) {
    //    return std::tolower(c); });
    //if (l_GpuOption == "gpu") {
    //    useGPU = true;
    //    std::cout << "Using GPU" << std::endl;
    //}
    //std::string model_file = "/home/deploy/app/YOLO/YOLOV5/yolov5-transfer-learning/yolov5/runs/train/exp8/weights/best.onnx" ;//argv[2];
    
    // onnxruntime setup
    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");

    //auto providers = Ort::GetAvailableProviders();
    //std::cout << "Available providers" << std::endl;
    //for (auto provider : providers) {
    //    std::cout << provider << std::endl;
    //}
    //Ort::SessionOptions session_options;

    //if (useGPU) {
    //    OrtCUDAProviderOptions l_CudaOptions;
    //    l_CudaOptions.device_id = 0;
    //    std::cout << "Before setting session options" << std::endl;
    //    session_options.AppendExecutionProvider_CUDA(l_CudaOptions);
    //    std::cout << "set session options" << std::endl;
    //} else {
    //    // session_options.SetIntraOpNumThreads(12);
    //}
    bool UseThisDummyValueOtherwhileItDoesNotCompile = true;
    bool UseThisAnotherDummyValueOtherwhileItDoesNotCompile = true;
    DisplayThread displayThread(UseThisAnotherDummyValueOtherwhileItDoesNotCompile, "prey predator");
    ActionThread actionThread(UseThisDummyValueOtherwhileItDoesNotCompile);
    
    //Ort::Experimental::Session session = Ort::Experimental::Session(env, model_file, session_options); // access experimental components via the Experimental namespace

    InferenceThread inferenceThread( config, actionThread, displayThread);
    //InferenceThread inferenceThread( session, useGPU, actionThread, displayThread);

    cv::VideoCapture videoCapture("/home/deploy/app/homeandfamily/self-driving-car/database/videos/VID_20221227_094455.mp4");
    videoCapture.set(cv::CAP_PROP_FPS, 10);
    VideoCaptureThread videoCaptureThread(videoCapture, inferenceThread);
    
    //ActionThread actionThread(inferenceThread.getInferenceResult(), inferenceThread.getJsonResult());
    //DisplayThread displayThread(inferenceThread.getInferenceResult(), "prey predator");
    //DisplayThread displayThread(UseThisAnotherDummyValueOtherwhileItDoesNotCompile, "prey predator");

    std::thread videoCaptureThreadHandle(
            &VideoCaptureThread::startCapture,
            &videoCaptureThread
            );

    std::thread inferenceThreadHandle(
            &InferenceThread::start,
            &inferenceThread
    );

    std::thread actionThreadHandle(
            &ActionThread::startAction,
            &actionThread
            );

    std::thread displayThreadHandle(
            &DisplayThread::startDisplay,
            &displayThread
            );

    videoCaptureThreadHandle.join();
    inferenceThreadHandle.join();
    actionThreadHandle.join();
    displayThreadHandle.join();

    return 0;
}
