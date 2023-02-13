/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.cc to edit this template
 */

#include "ActionThread.h"

void ActionThread::startAction() {
    while (true) {
        std::unique_lock<std::mutex> lock_frameMutex(m_jsonActionMutex);
        std::cout << "m_jsonAction : " << m_jsonAction.dump(2) << "\n";
        lock_frameMutex.unlock();        
    }
}

void ActionThread::setJsonAction(const nlohmann::json & jsonAction)
{
        std::unique_lock<std::mutex> lock(m_jsonActionMutex);
        m_jsonAction = jsonAction;
        lock.unlock();
        //m_frameCondition.notify_one();
}
