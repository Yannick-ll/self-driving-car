/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "Action.h"

void Action::startAction() {
    //std::cout << "m_jsonAction : " << m_jsonAction.dump(2) << "\n";
    Movement inferenceMovement;
    try{
        inferenceMovement = m_jsonAction;
        if(inferenceMovement.get_detection().get_conf() > 0){
            std::cout << std::endl << "====================================================== " << std::endl;
            std::cout << "m_jsonAction : " << m_jsonAction.dump(2) << "\n";
            JARVIS::ENUM::EnumCardinalPoint enumCardinalPoint = inferenceMovement.get_enumCardinalPoint();
            JARVIS::ENUM::EnumMovement enumMovement = inferenceMovement.get_enumMovement();       
            float left_speed = movement.get_left_speed();
            float right_speed = movement.get_right_speed();
            switch (enumCardinalPoint) {
                case JARVIS::ENUM::EnumCardinalPoint::NORTH:  
                    std::cout << "North " << std::endl;
                    left_speed = 50;
                    right_speed = 50;                   
                    break;
                case JARVIS::ENUM::EnumCardinalPoint::SOUTH:  
                    std::cout << "South " << std::endl;
                    left_speed = -50;
                    right_speed = -50;
                    break;
                case JARVIS::ENUM::EnumCardinalPoint::EAST:  
                    std::cout << "TURNING right" << std::endl;
                    left_speed = 50;
                    right_speed = 0;
                    break;
                case JARVIS::ENUM::EnumCardinalPoint::WEST:  
                    std::cout << "TURNING left" << std::endl;
                    left_speed = 0;
                    right_speed = 50;
                    break;
                case JARVIS::ENUM::EnumCardinalPoint::CONTINUE:  
                    break;
                default:
                    break;
            }
            
            switch (enumMovement) {
                case JARVIS::ENUM::EnumMovement::FORWARD:  
                    std::cout << "FORWARD" << std::endl;
                    left_speed = 50;
                    right_speed = 50;
                    break;
                case JARVIS::ENUM::EnumMovement::BACKWARD:  
                    std::cout << "BACKWARD" << std::endl;
                    left_speed = -50;
                    right_speed = -50;
                    break;
                case JARVIS::ENUM::EnumMovement::STOP: 
                    std::cout << "STOPPING both wheels !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                    left_speed = 0;
                    right_speed = 0;
                    std::cout << "Sleep 10 seconds..." << std::endl;
                    //sleep(10);
                    break;
                case JARVIS::ENUM::EnumMovement::CONTINUE: 
                    break;
                default:
                    break;
            }
            //motor_control_left.SetSpeed(left_speed);
            //motor_control_right.SetSpeed(right_speed);
            movement.set_left_speed(left_speed);
            movement.set_right_speed(right_speed);            
            
        }else{
            //std::cout << "ERROR : inferenceMovement.get_detection().get_conf() = " << inferenceMovement.get_detection().get_conf() << std::endl;
        }
    }catch (nlohmann::json::parse_error& e)
    {
        std::cout << "message: " << e.what() << '\n'
                  << "exception id: " << e.id << '\n'
                  << "byte position of error: " << e.byte << std::endl;
    }
}

void Action::setJsonAction(const nlohmann::json & jsonAction)
{
        m_jsonAction = jsonAction;
}
