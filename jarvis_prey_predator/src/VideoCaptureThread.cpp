/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "VideoCaptureThread.h"

void VideoCaptureThread::startCapture() {
    while (true) {
        cv::Mat frame;
        if (!m_videoCapture.read(frame))
            break;
        m_frame = frame;
    }
}

cv::Mat & VideoCaptureThread::getFrame() {
    return m_frame;
}
