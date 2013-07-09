//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which a watchpoint will be hit
// while a breakpoint is being handled in another thread.  The expected result is
// that the watchpoint in the second thread will be hit while the breakpoint handler
// in the first thread is trying to stop all threads.

#include <atomic>
#include <vector>
using namespace std;

#include <pthread.h>
#include <semaphore.h>

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

// Note that although hogging the CPU while waiting for a variable to change
// would be terrible in production code, it's great for testing since it
// avoids a lot of messy context switching to get multiple threads synchronized.
#define do_nothing()

#define pseudo_barrier_wait(bar) \
    --bar;                       \
    while (bar > 0)              \
        do_nothing();

#define pseudo_barrier_init(bar, count) (bar = count)

typedef std::vector<std::pair<unsigned, void*(*)(void*)> > action_counts;
typedef std::vector<pthread_t> thread_vector;

std::atomic_int g_barrier;
int g_breakpoint = 0;
int g_sigusr1_count = 0;
std::atomic_int g_watchme;

//sem_t g_signal_semaphore;

struct action_args {
  int delay;
};

// Perform any extra actions required by thread 'input' arg
void do_action_args(void *input) {
    if (input) {
      action_args *args = static_cast<action_args*>(input);
      sleep(args->delay);
    }
}

void *
breakpoint_func (void *input)
{
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    // Do something
    g_breakpoint++;       // Set breakpoint here
    return 0;
}

void *
signal_func (void *input) {
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    // Generate a user-defined signal to current process
    kill(getpid(), SIGUSR1);

    // wait for notification the signal handler was executed
    //sem_wait(&g_signal_semaphore);

    return 0;
}

void *
watchpoint_func (void *input) {
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    g_watchme += 1;     // watchpoint triggers here
    return 0;
}

void *
crash_func (void *input) {
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    int *a = 0;
    *a = 5; // crash happens here
    return 0;
}

void sigusr1_handler(int sig) {
    //sem_post(&g_signal_semaphore);
    if (sig == SIGUSR1)
        g_sigusr1_count += 1; // Break here in signal handler
}

/// Register a simple function for to handle signal
void register_signal_handler(int signal, void (*handler)(int))
{
    sigset_t empty_sigset;
    sigemptyset(&empty_sigset);

    struct sigaction action;
    action.sa_sigaction = 0;
    action.sa_mask = empty_sigset;
    action.sa_flags = 0;
    action.sa_handler = handler;
    sigaction(SIGUSR1, &action, 0);

    //sem_init(&g_signal_semaphore, 0, 0);
}

void start_threads(thread_vector& threads,
                   action_counts& actions,
                   void* args = 0) {
    action_counts::iterator b = actions.begin(), e = actions.end();
    for(action_counts::iterator i = b; i != e; ++i) {
        for(unsigned count = 0; count < i->first; ++count) {
            pthread_t t;
            pthread_create(&t, 0, i->second, args);
            threads.push_back(t);
        }
    }
}

int main ()
{
    g_watchme = 0;

    unsigned num_breakpoint_threads = 1;
    unsigned num_watchpoint_threads = 0;
    unsigned num_signal_threads = 0;
    unsigned num_crash_threads = 1;

    unsigned num_delay_breakpoint_threads = 0;
    unsigned num_delay_watchpoint_threads = 0;
    unsigned num_delay_signal_threads = 0;
    unsigned num_delay_crash_threads = 0;

    // Break here and adjust num_[breakpoint|watchpoint|signal|crash]_threads
    unsigned total_threads = num_breakpoint_threads \
                             + num_watchpoint_threads \
                             + num_signal_threads \
                             + num_crash_threads \
                             + num_delay_breakpoint_threads \
                             + num_delay_watchpoint_threads \
                             + num_delay_signal_threads \
                             + num_delay_crash_threads;

    // Don't let either thread do anything until they're both ready.
    pseudo_barrier_init(g_barrier, total_threads);

    thread_vector threads;

    action_counts actions;
    actions.push_back(std::make_pair(num_breakpoint_threads, breakpoint_func));
    actions.push_back(std::make_pair(num_watchpoint_threads, watchpoint_func));
    actions.push_back(std::make_pair(num_signal_threads, signal_func));
    actions.push_back(std::make_pair(num_crash_threads, crash_func));

    action_counts delay_actions;
    actions.push_back(std::make_pair(num_delay_breakpoint_threads, breakpoint_func));
    actions.push_back(std::make_pair(num_delay_watchpoint_threads, watchpoint_func));
    actions.push_back(std::make_pair(num_delay_signal_threads, signal_func));
    actions.push_back(std::make_pair(num_delay_crash_threads, crash_func));

    register_signal_handler(SIGUSR1, sigusr1_handler);

    // Create threads that handle instant actions
    start_threads(threads, actions);

    // Create threads that handle delayed actions
    action_args delay_arg;
    delay_arg.delay = 1;
    start_threads(threads, delay_actions, &delay_arg);

    // Join all threads
    typedef std::vector<pthread_t>::iterator thread_iterator;
    for(thread_iterator t = threads.begin(); t != threads.end(); ++t)
        pthread_join(*t, 0);

    // Break here and verify one thread is active.
    return 0;
}
