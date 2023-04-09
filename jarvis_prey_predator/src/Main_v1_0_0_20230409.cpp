#include <thread>

#include "Inference.h"
#include "Action.h"

int main(int argc, char** argv) {
    int size_model = 320;
    
    string model_path = "/home/jarvis/app/self-driving-car/jarvis_prey_predator/model/best.onnx";
    loguru::init(argc, argv);
    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);

    Config config = {0.50f, 0.45f, model_path, "/home/jarvis/app/jarvis_yolov5/racecar.names", Size(size_model, size_model), false};
    LOG_F(INFO, "Start main process");
    Inference inference(config);
    LOG_F(INFO, "Load model done ..");
    
    
    bool useGPU = false;
    std::string l_GpuOption = "CPU" ; //(argv[3]);
    std::transform(l_GpuOption.begin(), l_GpuOption.end(), l_GpuOption.begin(), [](unsigned char c) {
        return std::tolower(c); });
    if (l_GpuOption == "gpu") {
        useGPU = true;
        std::cout << "Using GPU" << std::endl;
    }
    


    
    
    //Inference inference;
    Action action;
    
        
    cv::VideoCapture videoCapture("/home/deploy/app/homeandfamily/self-driving-car/database/videos/VID_20221227_094455.mp4");
    videoCapture.set(cv::CAP_PROP_FPS, 10);

    // Default resolutions of the frame are obtained.The default resolutions are system dependent.
    int frame_width = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    float scaleWriteVideo = 0.5;
    cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width*scaleWriteVideo,frame_height*scaleWriteVideo));

    
    cv::Mat image;
    
    while (true) {
        videoCapture.read(image);
        if (image.empty()) {
            std::cout << "End of stream\n";
            break;
        }
        
        auto startTime = std::chrono::steady_clock::now();
        
        inference.setFrame(image);
        inference.start();
        video.write(inference.getInferenceResult().clone());
        action.setJsonAction(inference.getJsonResult());
        action.startAction();        
        
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;
        
        if(!inference.getInferenceResult().empty()){
            cv::imshow("imageDisplayResized", inference.getInferenceResult());
            cv::waitKey(10);
        }

        if (cv::waitKey(1) != -1) {
            videoCapture.release();
            video.release();
            cv::destroyAllWindows();
            std::cout << "finished by user\n";
            break;
        }
    }


    return 0;
}

