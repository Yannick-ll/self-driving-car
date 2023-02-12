/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   VideoCaptureThread.h
 * Author: yannick
 *
 * Created on 8 février 2023, 21:43
 */

#ifndef VIDEOCAPTURETHREAD_H
#define VIDEOCAPTURETHREAD_H

#include "Includes.h"
#include "InferenceThread.h"

class VideoCaptureThread
{
public:
  VideoCaptureThread(cv::VideoCapture& videoCapture, InferenceThread& inferenceThread)
    : m_videoCapture(videoCapture), m_inferenceThread(inferenceThread)
  {
  }

  void startCapture();

  cv::Mat & getFrame();

private:
  cv::VideoCapture& m_videoCapture;
  InferenceThread& m_inferenceThread;
  cv::Mat m_frame;
};

#endif /* VIDEOCAPTURETHREAD_H */

