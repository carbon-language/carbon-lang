//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <condition_variable>

// class condition_variable;

// ~condition_variable();

#include <condition_variable>
#include <mutex>
#include <thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::condition_variable* cv;
std::mutex m;
typedef std::unique_lock<std::mutex> Lock;

bool f_ready = false;
bool g_ready = false;

void f()
{
    Lock lk(m);
    f_ready = true;
    cv->notify_one();
    delete cv;
}

void g()
{
    Lock lk(m);
    g_ready = true;
    cv->notify_one();
    while (!f_ready)
        cv->wait(lk);
}

int main(int, char**)
{
    cv = new std::condition_variable;
    std::thread th2 = support::make_test_thread(g);
    Lock lk(m);
    while (!g_ready)
        cv->wait(lk);
    lk.unlock();
    std::thread th1 = support::make_test_thread(f);
    th1.join();
    th2.join();

  return 0;
}
