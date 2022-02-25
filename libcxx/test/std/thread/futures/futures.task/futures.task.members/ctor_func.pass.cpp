//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <future>

// class packaged_task<R(ArgTypes...)>

// template <class F>
//     explicit packaged_task(F&& f);

#include <future>
#include <cassert>

#include "test_macros.h"

class A
{
    long data_;

public:
    static int n_moves;
    static int n_copies;

    explicit A(long i) : data_(i) {}
    A(A&& a) : data_(a.data_) {++n_moves; a.data_ = -1;}
    A(const A& a) : data_(a.data_) {++n_copies;}

    long operator()(long i, long j) const {return data_ + i + j;}
};

int A::n_moves = 0;
int A::n_copies = 0;

int func(int i) { return i; }

int main(int, char**)
{
    {
        std::packaged_task<double(int, char)> p(A(5));
        assert(p.valid());
        std::future<double> f = p.get_future();
        p(3, 97);
        assert(f.get() == 105.0);
        assert(A::n_copies == 0);
        assert(A::n_moves > 0);
    }
    A::n_copies = 0;
    A::n_copies = 0;
    {
        A a(5);
        std::packaged_task<double(int, char)> p(a);
        assert(p.valid());
        std::future<double> f = p.get_future();
        p(3, 97);
        assert(f.get() == 105.0);
        assert(A::n_copies > 0);
        assert(A::n_moves > 0);
    }
    {
        std::packaged_task<int(int)> p(&func);
        assert(p.valid());
        std::future<int> f = p.get_future();
        p(4);
        assert(f.get() == 4);
    }
    {
        std::packaged_task<int(int)> p(func);
        assert(p.valid());
        std::future<int> f = p.get_future();
        p(4);
        assert(f.get() == 4);
    }

  return 0;
}
