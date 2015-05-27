//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test move

// UNSUPPORTED: c++98, c++03

#include <utility>
#include <cassert>

int copy_ctor = 0;
int move_ctor = 0;

class A
{
public:

    A(const A&) {++copy_ctor;}
    A& operator=(const A&);

    A(A&&) {++move_ctor;}
    A& operator=(A&&);

    A() {}
};

A source() {return A();}
const A csource() {return A();}

void test(A) {}

int main()
{
    A a;
    const A ca = A();

    assert(copy_ctor == 0);
    assert(move_ctor == 0);

    A a2 = a;
    assert(copy_ctor == 1);
    assert(move_ctor == 0);

    A a3 = std::move(a);
    assert(copy_ctor == 1);
    assert(move_ctor == 1);

    A a4 = ca;
    assert(copy_ctor == 2);
    assert(move_ctor == 1);

    A a5 = std::move(ca);
    assert(copy_ctor == 3);
    assert(move_ctor == 1);
}
