//===------------------------- dynamic_cast5.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>

namespace t1
{

struct A1
{
    char _[43981];
    virtual ~A1() {}

    A1* getA1() {return this;}
};

struct A2
{
    char _[34981];
    virtual ~A2() {}

    A2* getA2() {return this;}
};

struct A3
    : public virtual A1,
      private A2
{
    char _[93481];
    virtual ~A3() {}

    A1* getA1() {return A1::getA1();}
    A2* getA2() {return A2::getA2();}
    A3* getA3() {return this;}
};

struct A4
    : public A3,
      public A2
{
    char _[13489];
    virtual ~A4() {}

    t1::A1* getA1() {return A3::getA1();}
    A2* getA2() {return A3::getA2();}
    A3* getA3() {return A3::getA3();}
    A4* getA4() {return this;}
};

struct A5
    : public A4,
      public A3
{
    char _[13489];
    virtual ~A5() {}

    t1::A1* getA1() {return A4::getA1();}
    A2* getA2() {return A4::getA2();}
    A3* getA3() {return A4::getA3();}
    A4* getA4() {return A4::getA4();}
    A5* getA5() {return this;}
};

void test()
{
    A1 a1;
    A2 a2;
    A3 a3;
    A4 a4;
    A5 a5;

    assert(dynamic_cast<A1*>(a1.getA1()) == a1.getA1());
    assert(dynamic_cast<A1*>(a2.getA2()) == 0);
    assert(dynamic_cast<A1*>(a3.getA1()) == a3.getA1());
    assert(dynamic_cast<A1*>(a3.getA2()) == 0);
    assert(dynamic_cast<A1*>(a3.getA3()) == a3.getA1());
    assert(dynamic_cast<A1*>(a4.getA1()) == a4.getA1());
    assert(dynamic_cast<A1*>(a4.getA2()) == 0);
    assert(dynamic_cast<A1*>(a4.getA3()) == a4.getA1());
    assert(dynamic_cast<A1*>(a4.getA4()) == a4.getA1());
    assert(dynamic_cast<A1*>(a5.getA1()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA2()) == 0);
    assert(dynamic_cast<A1*>(a5.getA3()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA4()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA5()) == a5.getA1());

    assert(dynamic_cast<A2*>(a1.getA1()) == 0);
    assert(dynamic_cast<A2*>(a2.getA2()) == a2.getA2());
    assert(dynamic_cast<A2*>(a3.getA1()) == 0);
    assert(dynamic_cast<A2*>(a3.getA2()) == a3.getA2());
//    assert(dynamic_cast<A2*>(a3.getA3()) == 0);  // cast to private base
    assert(dynamic_cast<A2*>(a4.getA1()) == 0);
    assert(dynamic_cast<A2*>(a4.getA2()) == a4.getA2());
//    assert(dynamic_cast<A2*>(a4.getA3()) == 0);  // cast to private base
//    assert(dynamic_cast<A2*>(a4.getA4()) == 0);  // cast to ambiguous base
    assert(dynamic_cast<A2*>(a5.getA1()) == 0);
    assert(dynamic_cast<A2*>(a5.getA2()) == a5.getA2());
//    assert(dynamic_cast<A2*>(a5.getA3()) == 0);  // cast to private base
//    assert(dynamic_cast<A2*>(a5.getA4()) == 0);  // cast to ambiguous base
//    assert(dynamic_cast<A2*>(a5.getA5()) == 0);  // cast to ambiguous base

    assert(dynamic_cast<A3*>(a1.getA1()) == 0);
    assert(dynamic_cast<A3*>(a2.getA2()) == 0);
    assert(dynamic_cast<A3*>(a3.getA1()) == a3.getA3());
    assert(dynamic_cast<A3*>(a3.getA2()) == 0);
    assert(dynamic_cast<A3*>(a3.getA3()) == a3.getA3());
    assert(dynamic_cast<A3*>(a4.getA1()) == a4.getA3());
    assert(dynamic_cast<A3*>(a4.getA2()) == 0);
    assert(dynamic_cast<A3*>(a4.getA3()) == a4.getA3());
    assert(dynamic_cast<A3*>(a4.getA4()) == a4.getA3());
    assert(dynamic_cast<A3*>(a5.getA1()) == 0);
    assert(dynamic_cast<A3*>(a5.getA2()) == 0);
    assert(dynamic_cast<A3*>(a5.getA3()) == a5.getA3());
    assert(dynamic_cast<A3*>(a5.getA4()) == a5.getA3());
//    assert(dynamic_cast<A3*>(a5.getA5()) == 0);  // cast to ambiguous base

    assert(dynamic_cast<A4*>(a1.getA1()) == 0);
    assert(dynamic_cast<A4*>(a2.getA2()) == 0);
    assert(dynamic_cast<A4*>(a3.getA1()) == 0);
    assert(dynamic_cast<A4*>(a3.getA2()) == 0);
    assert(dynamic_cast<A4*>(a3.getA3()) == 0);
    assert(dynamic_cast<A4*>(a4.getA1()) == a4.getA4());
    assert(dynamic_cast<A4*>(a4.getA2()) == 0);
    assert(dynamic_cast<A4*>(a4.getA3()) == a4.getA4());
    assert(dynamic_cast<A4*>(a4.getA4()) == a4.getA4());
    assert(dynamic_cast<A4*>(a5.getA1()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA2()) == 0);
    assert(dynamic_cast<A4*>(a5.getA3()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA4()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA5()) == a5.getA4());

    assert(dynamic_cast<A5*>(a1.getA1()) == 0);
    assert(dynamic_cast<A5*>(a2.getA2()) == 0);
    assert(dynamic_cast<A5*>(a3.getA1()) == 0);
    assert(dynamic_cast<A5*>(a3.getA2()) == 0);
    assert(dynamic_cast<A5*>(a3.getA3()) == 0);
    assert(dynamic_cast<A5*>(a4.getA1()) == 0);
    assert(dynamic_cast<A5*>(a4.getA2()) == 0);
    assert(dynamic_cast<A5*>(a4.getA3()) == 0);
    assert(dynamic_cast<A5*>(a4.getA4()) == 0);
    assert(dynamic_cast<A5*>(a5.getA1()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA2()) == 0);
    assert(dynamic_cast<A5*>(a5.getA3()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA4()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA5()) == a5.getA5());
}

}  // t1

namespace t2
{

struct A1
{
    char _[43981];
    virtual ~A1() {}

    A1* getA1() {return this;}
};

struct A2
{
    char _[34981];
    virtual ~A2() {}

    A2* getA2() {return this;}
};

struct A3
    : public virtual A1,
      public A2
{
    char _[93481];
    virtual ~A3() {}

    A1* getA1() {return A1::getA1();}
    A2* getA2() {return A2::getA2();}
    A3* getA3() {return this;}
};

struct A4
    : public A3,
      public A2
{
    char _[13489];
    virtual ~A4() {}

    t2::A1* getA1() {return A3::getA1();}
    A2* getA2() {return A3::getA2();}
    A3* getA3() {return A3::getA3();}
    A4* getA4() {return this;}
};

struct A5
    : public A4,
      public A3
{
    char _[13489];
    virtual ~A5() {}

    t2::A1* getA1() {return A4::getA1();}
    A2* getA2() {return A4::getA2();}
    A3* getA3() {return A4::getA3();}
    A4* getA4() {return A4::getA4();}
    A5* getA5() {return this;}
};

void test()
{
    A1 a1;
    A2 a2;
    A3 a3;
    A4 a4;
    A5 a5;

    assert(dynamic_cast<A1*>(a1.getA1()) == a1.getA1());
    assert(dynamic_cast<A1*>(a2.getA2()) == 0);
    assert(dynamic_cast<A1*>(a3.getA1()) == a3.getA1());
    assert(dynamic_cast<A1*>(a3.getA2()) == a3.getA1());
    assert(dynamic_cast<A1*>(a3.getA3()) == a3.getA1());
    assert(dynamic_cast<A1*>(a4.getA1()) == a4.getA1());
    assert(dynamic_cast<A1*>(a4.getA2()) == a4.getA1());
    assert(dynamic_cast<A1*>(a4.getA3()) == a4.getA1());
    assert(dynamic_cast<A1*>(a4.getA4()) == a4.getA1());
    assert(dynamic_cast<A1*>(a5.getA1()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA2()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA3()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA4()) == a5.getA1());
    assert(dynamic_cast<A1*>(a5.getA5()) == a5.getA1());

    assert(dynamic_cast<A2*>(a1.getA1()) == 0);
    assert(dynamic_cast<A2*>(a2.getA2()) == a2.getA2());
    assert(dynamic_cast<A2*>(a3.getA1()) == a3.getA2());
    assert(dynamic_cast<A2*>(a3.getA2()) == a3.getA2());
    assert(dynamic_cast<A2*>(a3.getA3()) == a3.getA2());
    assert(dynamic_cast<A2*>(a4.getA1()) == 0);
    assert(dynamic_cast<A2*>(a4.getA2()) == a4.getA2());
    assert(dynamic_cast<A2*>(a4.getA3()) == a4.getA2());
//    assert(dynamic_cast<A2*>(a4.getA4()) == 0);  // cast to ambiguous base
    assert(dynamic_cast<A2*>(a5.getA1()) == 0);
    assert(dynamic_cast<A2*>(a5.getA2()) == a5.getA2());
    assert(dynamic_cast<A2*>(a5.getA3()) == a5.getA2());
//    assert(dynamic_cast<A2*>(a5.getA4()) == 0);  // cast to ambiguous base
//    assert(dynamic_cast<A2*>(a5.getA5()) == 0);  // cast to ambiguous base

    assert(dynamic_cast<A3*>(a1.getA1()) == 0);
    assert(dynamic_cast<A3*>(a2.getA2()) == 0);
    assert(dynamic_cast<A3*>(a3.getA1()) == a3.getA3());
    assert(dynamic_cast<A3*>(a3.getA2()) == a3.getA3());
    assert(dynamic_cast<A3*>(a3.getA3()) == a3.getA3());
    assert(dynamic_cast<A3*>(a4.getA1()) == a4.getA3());
    assert(dynamic_cast<A3*>(a4.getA2()) == a4.getA3());
    assert(dynamic_cast<A3*>(a4.getA3()) == a4.getA3());
    assert(dynamic_cast<A3*>(a4.getA4()) == a4.getA3());
    assert(dynamic_cast<A3*>(a5.getA1()) == 0);
    assert(dynamic_cast<A3*>(a5.getA2()) == a5.getA3());
    assert(dynamic_cast<A3*>(a5.getA3()) == a5.getA3());
    assert(dynamic_cast<A3*>(a5.getA4()) == a5.getA3());
//    assert(dynamic_cast<A3*>(a5.getA5()) == 0);  // cast to ambiguous base

    assert(dynamic_cast<A4*>(a1.getA1()) == 0);
    assert(dynamic_cast<A4*>(a2.getA2()) == 0);
    assert(dynamic_cast<A4*>(a3.getA1()) == 0);
    assert(dynamic_cast<A4*>(a3.getA2()) == 0);
    assert(dynamic_cast<A4*>(a3.getA3()) == 0);
    assert(dynamic_cast<A4*>(a4.getA1()) == a4.getA4());
    assert(dynamic_cast<A4*>(a4.getA2()) == a4.getA4());
    assert(dynamic_cast<A4*>(a4.getA3()) == a4.getA4());
    assert(dynamic_cast<A4*>(a4.getA4()) == a4.getA4());
    assert(dynamic_cast<A4*>(a5.getA1()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA2()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA3()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA4()) == a5.getA4());
    assert(dynamic_cast<A4*>(a5.getA5()) == a5.getA4());

    assert(dynamic_cast<A5*>(a1.getA1()) == 0);
    assert(dynamic_cast<A5*>(a2.getA2()) == 0);
    assert(dynamic_cast<A5*>(a3.getA1()) == 0);
    assert(dynamic_cast<A5*>(a3.getA2()) == 0);
    assert(dynamic_cast<A5*>(a3.getA3()) == 0);
    assert(dynamic_cast<A5*>(a4.getA1()) == 0);
    assert(dynamic_cast<A5*>(a4.getA2()) == 0);
    assert(dynamic_cast<A5*>(a4.getA3()) == 0);
    assert(dynamic_cast<A5*>(a4.getA4()) == 0);
    assert(dynamic_cast<A5*>(a5.getA1()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA2()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA3()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA4()) == a5.getA5());
    assert(dynamic_cast<A5*>(a5.getA5()) == a5.getA5());
}

}  // t2

int main()
{
    t1::test();
    t2::test();
}
