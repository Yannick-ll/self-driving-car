#include <thread>

#include "VideoCaptureThread.h"
#include "DisplayThread.h"
#include "InferenceThread.h"
#include "ActionThread.h"

int main(int argc, char** argv) {
    cv::VideoCapture videoCapture("/home/deploy/app/homeandfamily/self-driving-car/database/videos/VID_20221227_094455.mp4");
    VideoCaptureThread videoCaptureThread(videoCapture);
    InferenceThread inferenceThread(videoCaptureThread.getFrame());
    ActionThread actionThread(inferenceThread.getInferenceResult(), inferenceThread.getJsonResult());
    DisplayThread displayThread(videoCaptureThread.getFrame(), "prey predator");

    std::thread videoCaptureThreadHandle(
            &VideoCaptureThread::startCapture,
            &videoCaptureThread
            );

    std::thread inferenceThreadHandle(
            &InferenceThread::startInference,
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

