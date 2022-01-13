//===---------------------- catch_class_03.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
    This test checks that adjustedPtr is correct as there exist offsets in this
    object for the various subobjects, all of which have a unique id_ to
    check against.
*/

// UNSUPPORTED: no-exceptions

// FIXME: GCC doesn't allow turning off the warning for exceptions being caught
//        by earlier handlers, which this test is exercising. We have to disable
//        warnings altogether to remove the error.
//        See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97675.
// ADDITIONAL_COMPILE_FLAGS: -Wno-error

#include <exception>
#include <stdlib.h>
#include <assert.h>

// Clang emits  warnings about exceptions of type 'Child' being caught by
// an earlier handler of type 'Base'. Congrats clang, you've just
// diagnosed the behavior under test.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wexceptions"
#endif

struct B
{
    static int count;
    int id_;
    explicit B(int id) : id_(id) {count++;}
    B(const B& a) : id_(a.id_) {count++;}
    ~B() {count--;}
};

int B::count = 0;

struct C1
    : B
{
    static int count;
    int id_;
    explicit C1(int id) : B(id-2), id_(id) {count++;}
    C1(const C1& a) : B(a.id_-2), id_(a.id_) {count++;}
    ~C1() {count--;}
};

int C1::count = 0;

struct C2
    : B
{
    static int count;
    int id_;
    explicit C2(int id) : B(id-2), id_(id) {count++;}
    C2(const C2& a) : B(a.id_-2), id_(a.id_) {count++;}
    ~C2() {count--;}
};

int C2::count = 0;

struct A
    : C1, C2
{
    static int count;
    int id_;
    explicit A(int id) : C1(id-1), C2(id-2), id_(id) {count++;}
    A(const A& a) : C1(a.id_-1), C2(a.id_-2), id_(a.id_) {count++;}
    ~A() {count--;}
};

int A::count = 0;

void f1()
{
    assert(A::count == 0);
    assert(C1::count == 0);
    assert(C2::count == 0);
    assert(B::count == 0);
    A a(5);
    assert(A::count == 1);
    assert(C1::count == 1);
    assert(C2::count == 1);
    assert(B::count == 2);

    assert(a.id_ == 5);
    assert(static_cast<C1&>(a).id_ == 4);
    assert(static_cast<C2&>(a).id_ == 3);
    assert(static_cast<B&>(static_cast<C1&>(a)).id_ == 2);
    assert(static_cast<B&>(static_cast<C2&>(a)).id_ == 1);
    throw a;
    assert(false);
}

void f2()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f1();
        assert(false);
    }
    catch (const A& a)  // can catch A
    {
        assert(a.id_ == 5);
        assert(static_cast<const C1&>(a).id_ == 4);
        assert(static_cast<const C2&>(a).id_ == 3);
        assert(static_cast<const B&>(static_cast<const C1&>(a)).id_ == 2);
        assert(static_cast<const B&>(static_cast<const C2&>(a)).id_ == 1);
        throw;
    }
    catch (const C1&)
    {
        assert(false);
    }
    catch (const C2&)
    {
        assert(false);
    }
    catch (const B&)
    {
        assert(false);
    }
}

void f3()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f2();
        assert(false);
    }
    catch (const B& a)  // can not catch B (ambiguous base)
    {
        assert(false);
    }
    catch (const C1& c1)  // can catch C1
    {
        assert(c1.id_ == 4);
        assert(static_cast<const B&>(c1).id_ == 2);
        throw;
    }
    catch (const C2&)
    {
        assert(false);
    }
}

void f4()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f3();
        assert(false);
    }
    catch (const B& a)  // can not catch B (ambiguous base)
    {
        assert(false);
    }
    catch (const C2& c2)  // can catch C2
    {
        assert(c2.id_ == 3);
        assert(static_cast<const B&>(c2).id_ == 1);
        throw;
    }
    catch (const C1&)
    {
        assert(false);
    }
}

int main(int, char**)
{
    try
    {
        f4();
        assert(false);
    }
    catch (...)
    {
    }
    assert(A::count == 0);
    assert(C1::count == 0);
    assert(C2::count == 0);
    assert(B::count == 0);

    return 0;
}
