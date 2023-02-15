/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "DisplayThread.h"

void DisplayThread::startDisplay() {
    while (true) {
            std::unique_lock<std::mutex> lock_frameMutex(m_frameMutex);
            //m_frameCondition.wait(lock, [&]{return !m_frame.empty();});
            cv::Mat frame = m_frame.clone();
            lock_frameMutex.unlock();
        if (!frame.empty() && frame.size().height > 0 && frame.size().width > 0) {
            cv::imshow(m_title, frame);
            if (cv::waitKey(1) == 27)
                break;
        } else {
            //cv::Mat aa = cv::Mat(320, 240, CV_8UC3, cv::Scalar(0, 0, 0));
            //cv::imshow(m_title, aa);
            // if (cv::waitKey(5) == 27)
            // break;
        }
    }
}

void DisplayThread::setFrame(const cv::Mat& frame)
{
        std::unique_lock<std::mutex> lock(m_frameMutex);
        m_frame = frame;
        lock.unlock();
        //m_frameCondition.notify_one();
}
