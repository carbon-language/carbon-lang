#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <iostream>
#include <cassert>

void f1()
{
    std::exit(0);
}

struct Mutex
{
    unsigned state = 0;
    Mutex() = default;
    ~Mutex() = default;
    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;

    void lock()
    {
    if (++state == 2)
        throw 1;  // this throw should end up calling terminate()
    }

    void unlock() {}
};

Mutex mut;
std::condition_variable_any cv;

void
signal_me()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    cv.notify_one();
}

int
main()
{
    std::set_terminate(f1);
    try
    {
        std::thread(signal_me).detach();
        mut.lock();
        cv.wait_for(mut, std::chrono::milliseconds(250));
    }
    catch (...) {}
    assert(false);
}
