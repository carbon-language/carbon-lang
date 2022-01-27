//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "deleter_types.h"

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    virtual ~A() {--count;}
};

int A::count = 0;

struct B
    : public A
{
    static int count;
    B() {++count;}
    B(const B& other) : A(other) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

int main(int, char**)
{
    {
    const std::unique_ptr<A, Deleter<A> > p1(new A);
    const std::unique_ptr<A, Deleter<A> > p2(new A);
    assert(!(p1 == p2));
    assert(p1 != p2);
    }
    {
    const std::unique_ptr<A, Deleter<A> > p1(new A);
    const std::unique_ptr<B, Deleter<B> > p2(new B);
    assert(!(p1 == p2));
    assert(p1 != p2);
    }
    {
    const std::unique_ptr<A[], Deleter<A[]> > p1(new A[3]);
    const std::unique_ptr<A[], Deleter<A[]> > p2(new A[3]);
    assert(!(p1 == p2));
    assert(p1 != p2);
    }
    {
    const std::unique_ptr<A[], Deleter<A[]> > p1(new A[3]);
    const std::unique_ptr<B[], Deleter<B[]> > p2(new B[3]);
    assert(!(p1 == p2));
    assert(p1 != p2);
    }
    {
    const std::unique_ptr<A, Deleter<A> > p1;
    const std::unique_ptr<A, Deleter<A> > p2;
    assert(p1 == p2);
    assert(!(p1 != p2));
    }
    {
    const std::unique_ptr<A, Deleter<A> > p1;
    const std::unique_ptr<B, Deleter<B> > p2;
    assert(p1 == p2);
    assert(!(p1 != p2));
    }

  return 0;
}
