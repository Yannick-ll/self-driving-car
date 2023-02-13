/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "DisplayThread.h"

void DisplayThread::startDisplay() {
    while (true) {
        if (!m_frame.empty() && m_frame.size().height > 0 && m_frame.size().width > 0) {
            cv::imshow(m_title, m_frame);
            if (cv::waitKey(5) == 27)
                break;
        } else {
            //cv::Mat aa = cv::Mat(320, 240, CV_8UC3, cv::Scalar(0, 0, 0));
            //cv::imshow(m_title, aa);
            // if (cv::waitKey(5) == 27)
            // break;
        }
    }
}
