//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which two threads are stopped
// at a breakpoint and the debugger issues a step-out command.

#include <pthread.h>
#include <atomic>

// Note that although hogging the CPU while waiting for a variable to change
// would be terrible in production code, it's great for testing since it
// avoids a lot of messy context switching to get multiple threads synchronized.
#define do_nothing()

#define pseudo_barrier_wait(bar) \
    --bar;                       \
    while (bar > 0)              \
        do_nothing();

#define pseudo_barrier_init(bar, count) (bar = count)

std::atomic_int g_barrier;

volatile int g_test = 0;

void step_out_of_here() {
  g_test += 5; // Set breakpoint here
}

void *
thread_func (void *input)
{
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);

    // Do something
    step_out_of_here(); // Expect to stop here after step-out (clang)

    // Return
    return NULL;  // Expect to stop here after step-out (icc and gcc)
}

int main ()
{
    pthread_t thread_1;
    pthread_t thread_2;

    // Don't let either thread do anything until they're both ready.
    pseudo_barrier_init(g_barrier, 2);

    // Create two threads
    pthread_create (&thread_1, NULL, thread_func, NULL);
    pthread_create (&thread_2, NULL, thread_func, NULL);

    // Wait for the threads to finish
    pthread_join(thread_1, NULL);
    pthread_join(thread_2, NULL);

    return 0;
}
