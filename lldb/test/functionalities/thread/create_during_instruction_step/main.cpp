//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <thread>

std::atomic<bool> flag(false);

void do_nothing()
{
    while (flag)
        ;
}

int main ()
{
    // Instruction-level stepping over a creation of the first thread takes a very long time, so
    // we give the threading machinery a chance to initialize all its data structures.
    // This way, stepping over the second thread will be much faster.
    std::thread dummy(do_nothing);
    dummy.join();

    // Make sure the new thread does not exit before we get a chance to notice the main thread stopped
    flag = true;

    std::thread thread(do_nothing); // Set breakpoint here
    flag = false; // Release the new thread.
    thread.join();
    return 0;
}
