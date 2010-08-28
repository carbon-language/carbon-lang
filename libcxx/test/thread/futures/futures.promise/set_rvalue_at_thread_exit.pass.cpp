//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class promise<R>

// void promise::set_value_at_thread_exit(R&& r);

#include <future>
#include <memory>
#include <cassert>

#ifdef _LIBCPP_MOVE

void func(std::promise<std::unique_ptr<int>>& p)
{
    p.set_value_at_thread_exit(std::unique_ptr<int>(new int(5)));
}

#endif  // _LIBCPP_MOVE

int main()
{
#ifdef _LIBCPP_MOVE
    {
        std::promise<std::unique_ptr<int>> p;
        std::future<std::unique_ptr<int>> f = p.get_future();
        std::thread(func, std::move(p)).detach();
        assert(*f.get() == 5);
    }
#endif  // _LIBCPP_MOVE
}
