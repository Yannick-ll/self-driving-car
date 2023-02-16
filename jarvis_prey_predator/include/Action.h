/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   Action.h
 * Author: yannick
 *
 * Created on 14 février 2023, 22:37
 */

#ifndef ACTION_H
#define ACTION_H

#include "Includes.h"

class Action {
public:
    Action(){
        motor_control_left = PwmMotorControl(std::move(pin_pwm_left),
                                      std::move(pin_direction_left_1),
                                      std::move(pin_direction_left_2));
        motor_control_right = PwmMotorControl(std::move(pin_pwm_right),
                                      std::move(pin_direction_right_1),
                                      std::move(pin_direction_right_2));
    }
    /*Action(cv::Mat& inferenceResult, nlohmann::json & m_jsonResult)
    : m_inferenceResult(inferenceResult),
    m_jsonResult(m_jsonResult){
    }*/
    void startAction();
    void setJsonAction(const nlohmann::json & jsonAction);
private:
    nlohmann::json m_jsonAction;
    std::unique_ptr<RaspberryPiGpioPwmPin> pin_pwm_left = std::make_unique<RaspberryPiGpioPwmPin>(1);
    std::unique_ptr<RaspberryPiGpioPin> pin_direction_left_1 = std::make_unique<RaspberryPiGpioPin>(4);
    std::unique_ptr<RaspberryPiGpioPin> pin_direction_left_2 = std::make_unique<RaspberryPiGpioPin>(5);
    
    std::unique_ptr<RaspberryPiGpioPwmPin> pin_pwm_right = std::make_unique<RaspberryPiGpioPwmPin>(2);
    std::unique_ptr<RaspberryPiGpioPin> pin_direction_right_1 = std::make_unique<RaspberryPiGpioPin>(6);
    std::unique_ptr<RaspberryPiGpioPin> pin_direction_right_2 = std::make_unique<RaspberryPiGpioPin>(7);
    
    PwmMotorControl motor_control_left, motor_control_right;
};


#endif /* ACTION_H */

