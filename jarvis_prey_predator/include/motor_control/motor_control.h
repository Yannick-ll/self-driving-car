#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <memory>

#include <gpio.h>
#include <gpio_pwm.h>

class PwmMotorControl
{
public:
    PwmMotorControl(){}
    PwmMotorControl(std::unique_ptr<GpioPwmPin> pin_pwm,
                    std::unique_ptr<GpioPin> pin_direction_1,
                    std::unique_ptr<GpioPin> pin_direction_2);

    void SetSpeed(int8_t motor_speed);

    const int8_t & get_motor_speed() const {return motor_speed;}
    int8_t & get_motor_speed() {return motor_speed;}
    void set_motor_speed(const int8_t &motor_speed){this->motor_speed = motor_speed;}

private:
    int8_t motor_speed;
    std::unique_ptr<GpioPwmPin> pin_pwm_;
    std::unique_ptr<GpioPin> pin_direction_1_;
    std::unique_ptr<GpioPin> pin_direction_2_;

    uint8_t pin_pwm_value_;
    uint8_t pin_direction_1_value_;
    uint8_t pin_direction_2_value_;
};

#endif
