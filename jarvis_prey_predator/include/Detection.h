/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   Detection.h
 * Author: yannick
 *
 * Created on 15 février 2023, 18:47
 */

#ifndef DETECTION_H
#define DETECTION_H

#include "Includes.h"

class Detection{
    private:        
        cv::Rect box;
        float conf;        
        int classId;

    public :
        Detection()= default;
        virtual ~Detection()= default;

    public :   
        const cv::Rect & get_box() const {return box;}
        cv::Rect & get_box() {return box;}
        void set_box(const cv::Rect &box){this->box = box;}
        
        const float & get_conf() const {return conf;}
        float & get_conf() {return conf;}
        void set_conf(const float &conf){this->conf = conf;}

        const int & get_classId() const {return classId;}
        int & get_classId() {return classId;}
        void set_classId(const int &classId){this->classId = classId;}

        std::string to_string(){
            std::ostringstream s;
            s << "\n - classId = " << classId << " \n";
            s << "\n - conf = " << conf << " \n";
            return s.str();
        }
};

/*namespace nlohmann {
    void from_json(const json & j, Detection & x);
    void to_json(json & j, const Detection & x);

    inline void from_json(const json & j, Detection& x) {
        if(exists(j,"box")) x.set_box(j.at("box").get<cv::Rect>());
        if(exists(j,"conf")) x.set_conf(j.at("conf").get<float>());
        if(exists(j,"classId")) x.set_classId(j.at("classId").get<int>());
    }

    inline void to_json(json & j, const Detection & x) {
        j = json::object();
        j["box"] = x.get_box();
        j["conf"] = x.get_conf();
        j["classId"] = x.get_classId();
    }
}*/

#endif /* DETECTION_H */

