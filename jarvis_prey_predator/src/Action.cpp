/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "Action.h"

void Action::startAction() {
    std::cout << "m_jsonAction : " << m_jsonAction.dump(2) << "\n";
}

void Action::setJsonAction(const nlohmann::json & jsonAction)
{
        m_jsonAction = jsonAction;
}
