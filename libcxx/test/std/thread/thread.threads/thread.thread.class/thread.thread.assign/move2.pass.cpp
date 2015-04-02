//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// NOTE: std::terminate is called so the destructors are not invoked and the
// memory is not freed. This will cause ASAN to fail.
// XFAIL: asan

// NOTE: TSAN will report this test as leaking a thread.
// XFAIL: tsan

// <thread>

// class thread

// thread& operator=(thread&& t);

#include <thread>
#include <exception>
#include <cstdlib>
#include <cassert>

class G
{
    int alive_;
public:
    static int n_alive;
    static bool op_run;

    G() : alive_(1) {++n_alive;}
    G(const G& g) : alive_(g.alive_) {++n_alive;}
    ~G() {alive_ = 0; --n_alive;}

    void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        op_run = true;
    }

    void operator()(int i, double j)
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        assert(i == 5);
        assert(j == 5.5);
        op_run = true;
    }
};

int G::n_alive = 0;
bool G::op_run = false;

void f1()
{
    std::exit(0);
}

int main()
{
    std::set_terminate(f1);
    {
        G g;
        std::thread t0(g, 5, 5.5);
        std::thread::id id = t0.get_id();
        std::thread t1;
        t0 = std::move(t1);
        assert(false);
    }
}
