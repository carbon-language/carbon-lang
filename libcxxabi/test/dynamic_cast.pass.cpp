//===------------------------- dynamic_cast.pass.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>

// This test explicitly tests dynamic cast with types that have inaccessible
// bases.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Winaccessible-base"
#endif

typedef char Pad1[43981];
typedef char Pad2[34981];
typedef char Pad3[93481];
typedef char Pad4[13489];
typedef char Pad5[81349];
typedef char Pad6[34819];
typedef char Pad7[3489];

namespace t1
{

// PR33425
struct C3 { virtual ~C3() {} Pad1 _; };
struct C5 : protected virtual C3 { Pad2 _; };
struct C6 : virtual C5 { Pad3 _; };
struct C7 : virtual C3 { Pad4 _; };
struct C9 : C6, C7 { Pad5 _; };

C9 c9;
C3 *c3 = &c9;

void test()
{
    assert(dynamic_cast<C3*>(c3) == static_cast<C3*>(&c9));
    assert(dynamic_cast<C5*>(c3) == static_cast<C5*>(&c9));
    assert(dynamic_cast<C6*>(c3) == static_cast<C6*>(&c9));
    assert(dynamic_cast<C7*>(c3) == static_cast<C7*>(&c9));
    assert(dynamic_cast<C9*>(c3) == static_cast<C9*>(&c9));
}

}  // t1

namespace t2
{

// PR33425
struct Src { virtual ~Src() {} Pad1 _; };
struct Mask : protected virtual Src { Pad2 _; };
struct Dest : Mask { Pad3 _; };
struct Root : Dest, virtual Src { Pad4 _; };

Root root;
Src *src = &root;

void test()
{
    assert(dynamic_cast<Src*>(src) == static_cast<Src*>(&root));
    assert(dynamic_cast<Mask*>(src) == static_cast<Mask*>(&root));
    assert(dynamic_cast<Dest*>(src) == static_cast<Dest*>(&root));
    assert(dynamic_cast<Root*>(src) == static_cast<Root*>(&root));
}

}  // t2

namespace t3
{

// PR33487
struct Class1 { virtual ~Class1() {} Pad1 _; };
struct Shared : virtual Class1 { Pad2 _; };
struct Class6 : virtual Shared { Pad3 _; };
struct Left : Class6 { Pad4 _; };
struct Right : Class6 { Pad5 _; };
struct Main : Left, Right { Pad6 _; };

Main m;
Class1 *c1 = &m;

void test()
{
    assert(dynamic_cast<Class1*>(c1) == static_cast<Class1*>(&m));
    assert(dynamic_cast<Shared*>(c1) == static_cast<Shared*>(&m));
    assert(dynamic_cast<Class6*>(c1) == 0);
    assert(dynamic_cast<Left*>(c1) == static_cast<Left*>(&m));
    assert(dynamic_cast<Right*>(c1) == static_cast<Right*>(&m));
    assert(dynamic_cast<Main*>(c1) == static_cast<Main*>(&m));
}

}  // t3

int main()
{
    t1::test();
    t2::test();
    t3::test();
}
