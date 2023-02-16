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
#include "Point.h"

class Detection{
    private:        
        cv::Rect box;
        float conf;        
        int classId;
        JARVIS::Point start_point, end_point;

    public :
        Detection()= default;
        virtual ~Detection()= default;

    public :   
        const cv::Rect & get_box() const {return box;}
        cv::Rect & get_box() {return box;}
        void set_box(const cv::Rect &box){
            this->box = box;
            start_point.set_x(box.x);
            start_point.set_y(box.y);
            end_point.set_x(box.x+box.width);
            end_point.set_y(box.y+box.height);
        }
        void set_box(const JARVIS::Point &pTopLeft, const JARVIS::Point &pBottomRight){
            box = cv::Rect(pTopLeft.get_x(), pTopLeft.get_y(), 
                    std::abs(pBottomRight.get_x() - pTopLeft.get_x()), std::abs(pBottomRight.get_y() - pTopLeft.get_y()));
        }
        
        const JARVIS::Point & get_start_point() const {return start_point;}
        JARVIS::Point & get_start_point() {return start_point;}
        void set_start_point(const JARVIS::Point &start_point){this->start_point = start_point;}
        
        const JARVIS::Point & get_end_point() const {return end_point;}
        JARVIS::Point & get_end_point() {return end_point;}
        void set_end_point(const JARVIS::Point &end_point){this->end_point = end_point;}
        
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

namespace nlohmann {
    void from_json(const json & j, Detection & x);
    void to_json(json & j, const Detection & x);

    inline void from_json(const json & j, Detection& x) {
        if(j.contains("end_point")) x.set_start_point(j.at("start_point").get<JARVIS::Point>());
        if(j.contains("end_point")) x.set_end_point(j.at("end_point").get<JARVIS::Point>());
        if(j.contains("conf")) x.set_conf(j.at("conf").get<float>());
        if(j.contains("classId")) x.set_classId(j.at("classId").get<int>());
    }

    inline void to_json(json & j, const Detection & x) {
        j = json::object();
        j["start_point"] = x.get_start_point();
        j["end_point"] = x.get_end_point();
        j["conf"] = x.get_conf();
        j["classId"] = x.get_classId();
   }
}

#endif /* DETECTION_H */

