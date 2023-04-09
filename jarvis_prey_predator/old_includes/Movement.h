/* 
 * File:   Movement.h
 * Author: yannick
 *
 * Created on 16 février 2023, 11:04
 */

#ifndef MOVEMENT_H
#define MOVEMENT_H

#include "utils/EnumCardinalPoint.h"
#include "utils/EnumMovement.h"
#include "Detection.h"


class Movement{
    private:        
        JARVIS::ENUM::EnumCardinalPoint enumCardinalPoint = JARVIS::ENUM::EnumCardinalPoint::CONTINUE;
        JARVIS::ENUM::EnumMovement enumMovement = JARVIS::ENUM::EnumMovement::CONTINUE;
        Detection detection;
        float seconds = 0;
        float left_speed = 0;
        float right_speed = 0;
        //float left_speed, right_speed;
        
    public :
        Movement()= default;
        virtual ~Movement()= default;

    public :   
        const JARVIS::ENUM::EnumCardinalPoint & get_enumCardinalPoint() const {return enumCardinalPoint;}
        JARVIS::ENUM::EnumCardinalPoint & get_enumCardinalPoint() {return enumCardinalPoint;}
        void set_enumCardinalPoint(const JARVIS::ENUM::EnumCardinalPoint &enumCardinalPoint){this->enumCardinalPoint = enumCardinalPoint;}
        
        const JARVIS::ENUM::EnumMovement & get_enumMovement() const {return enumMovement;}
        JARVIS::ENUM::EnumMovement & get_enumMovement() {return enumMovement;}
        void set_enumMovement(const JARVIS::ENUM::EnumMovement &enumMovement){this->enumMovement = enumMovement;}
        
        const Detection & get_detection() const {return detection;}
        Detection & get_detection() {return detection;}
        void set_detection(const Detection &detection){this->detection = detection;}
        
        const float & get_seconds() const {return seconds;}
        float & get_seconds() {return seconds;}
        void set_seconds(const float &seconds){this->seconds = seconds;}

        const float & get_left_speed() const {return left_speed;}
        float & get_left_speed() {return left_speed;}
        void set_left_speed(const float &left_speed){this->left_speed = left_speed;}
        
        const float & get_right_speed() const {return right_speed;}
        float & get_right_speed() {return right_speed;}
        void set_right_speed(const float &right_speed){this->right_speed = right_speed;}
        std::string to_string(){
            std::ostringstream s;
            s << "\n - seconds = " << seconds << " \n";
            s << "\n - left_speed = " << left_speed << " \n";
            return s.str();
        }
};


namespace nlohmann {
    void from_json(const json & j, Movement & x);
    void to_json(json & j, const Movement & x);

    inline void from_json(const json & j, Movement& x) {
        if(j.contains("enumCardinalPoint")){
            std::string strEnumCardinalPoint = j.at("enumCardinalPoint").get< std::string >();
            JARVIS::ENUM::EnumCardinalPoint enumCardinalPoint = 
                    JARVIS::ENUM::EnumCardinalPoint::_from_string(strEnumCardinalPoint.c_str());
            x.set_enumCardinalPoint(enumCardinalPoint);
        }
        if(j.contains("enumMovement")){
            std::string strEnumMovement = j.at("enumMovement").get< std::string >();
            JARVIS::ENUM::EnumMovement enumMovement = 
                    JARVIS::ENUM::EnumMovement::_from_string(strEnumMovement.c_str());            
            x.set_enumMovement(enumMovement);
        }
        if(j.contains("detection")) x.set_detection(j.at("detection").get<Detection>());               
        if(j.contains("seconds")) x.set_seconds(j.at("seconds").get<float>());
        if(j.contains("left_speed")) x.set_left_speed(j.at("left_speed").get<float>());
        if(j.contains("right_speed")) x.set_right_speed(j.at("right_speed").get<float>());
    }
//JARVIS::ENUM::EnumCardinalPoint::_from_string(name.c_str());

    inline void to_json(json & j, const Movement & x) {
        j = json::object();
        j["enumCardinalPoint"] = x.get_enumCardinalPoint()._to_string();
        j["enumMovement"] = x.get_enumMovement()._to_string();
        j["detection"] = x.get_detection();
        j["seconds"] = x.get_seconds();
        j["left_speed"] = x.get_left_speed();
        j["right_speed"] = x.get_right_speed();
   }
}

#endif /* MOVEMENT_H */

