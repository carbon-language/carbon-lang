// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

// NOTE on the "CURRENT" prefix: some things are mangled incorrectly as of
// writing. If you find a CURRENT-test that fails with your patch, please test
// if your patch has actually fixed a problem in the mangler and replace the
// corresponding CORRECT line with a CHECK.
// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck -check-prefix CURRENT %s

void f1(const char* a, const char* b) {}
// CHECK: "\01?f1@@YAXPBD0@Z"

void f2(const char* a, char* b) {}
// CHECK: "\01?f2@@YAXPBDPAD@Z"

void f3(int a, const char* b, const char* c) {}
// CHECK: "\01?f3@@YAXHPBD0@Z"

const char *f4(const char* a, const char* b) { return 0; }
// CHECK: "\01?f4@@YAPBDPBD0@Z"

void f5(char const* a, unsigned int b, char c, void const* d, char const* e, unsigned int f) {}
// CHECK: "\01?f5@@YAXPBDIDPBX0I@Z"

void f6(bool a, bool b) {}
// CHECK: "\01?f6@@YAX_N0@Z"

void f7(int a, int* b, int c, int* d, bool e, bool f, bool* g) {}
// CHECK: "\01?f7@@YAXHPAHH0_N1PA_N@Z"

// FIXME: tests for more than 10 types?

struct S {
  void mbb(bool a, bool b) {}
};

void g1(struct S a) {}
// CHECK: "\01?g1@@YAXUS@@@Z"

void g2(struct S a, struct S b) {}
// CHECK: "\01?g2@@YAXUS@@0@Z"

void g3(struct S a, struct S b, struct S* c, struct S* d) {}
// CHECK: "\01?g3@@YAXUS@@0PAU1@1@Z"

void g4(const char* a, struct S* b, const char* c, struct S* d) {
// CHECK: "\01?g4@@YAXPBDPAUS@@01@Z"
  b->mbb(false, false);
// CHECK: "\01?mbb@S@@QAEX_N0@Z"
}

// Make sure that different aliases of built-in types end up mangled as the
// built-ins.
typedef unsigned int uintptr_t;
typedef unsigned int size_t;
void *h(size_t a, uintptr_t b) { return 0; }
// CHECK: "\01?h@@YAPAXII@Z"

// Function pointers might be mangled in a complex way.
typedef void (*VoidFunc)();
typedef int* (*PInt3Func)(int* a, int* b);

void h1(const char* a, const char* b, VoidFunc c, VoidFunc d) {}
// CHECK: "\01?h1@@YAXPBD0P6AXXZ1@Z"

void h2(void (*f_ptr)(void *), void *arg) {}
// CHECK: "\01?h2@@YAXP6AXPAX@Z0@Z"

PInt3Func h3(PInt3Func x, PInt3Func y, int* z) { return 0; }
// CHECK: "\01?h3@@YAP6APAHPAH0@ZP6APAH00@Z10@Z"

namespace PR13207 {
class A {};
class B {};
class C {};

template<class X>
class F {};
template<class X>
class I {};
template<class X, class Y>
class J {};
template<class X, class Y, class Z>
class K {};

class L {
 public:
  void foo(I<A> x) {}
};
// CHECK: "\01?foo@L@PR13207@@QAEXV?$I@VA@PR13207@@@2@@Z"

void call_l_foo(L* l) { l->foo(I<A>()); }

void foo(I<A> x) {}
// CHECK: "\01?foo@PR13207@@YAXV?$I@VA@PR13207@@@1@@Z"
void foo2(I<A> x, I<A> y) { }
// CHECK "\01?foo2@PR13207@@YAXV?$I@VA@PR13207@@@1@0@Z"
void bar(J<A,B> x) {}
// CHECK: "\01?bar@PR13207@@YAXV?$J@VA@PR13207@@VB@2@@1@@Z"
void spam(K<A,B,C> x) {}
// CHECK: "\01?spam@PR13207@@YAXV?$K@VA@PR13207@@VB@2@VC@2@@1@@Z"

// The following CURRENT line is here to improve the precision of the "scanning
// from here" reports of FileCheck.
// CURRENT: "\01?spam@PR13207@@YAXV?$K@VA@PR13207@@VB@2@VC@2@@1@@Z"

// The tests below currently fail:
void baz(K<char, F<char>, I<char> >) {}
// CURRENT: "\01?baz@PR13207@@YAXV?$K@DV?$F@D@PR13207@@V?$I@D@1@@1@@Z"
// CORRECT: "\01?baz@PR13207@@YAXV?$K@DV?$F@D@PR13207@@V?$I@D@2@@1@@Z"
void qux(K<char, I<char>, I<char> >) {}
// CURRENT: "\01?qux@PR13207@@YAXV?$K@DV?$I@D@PR13207@@V?$I@D@1@@1@@Z"
// CORRECT: "\01?qux@PR13207@@YAXV?$K@DV?$I@D@PR13207@@V12@@1@@Z

namespace NA {
class X {};
template<class T> class Y {};
void foo(Y<X> x) {}
// CHECK: "\01?foo@NA@PR13207@@YAXV?$Y@VX@NA@PR13207@@@12@@Z"
}

namespace NB {
class X {};
template<class T> class Y {};
void foo(Y<NA::X> x) {}
// CHECK: "\01?foo@NB@PR13207@@YAXV?$Y@VX@NA@PR13207@@@12@@Z"

void bar(NA::Y<X> x) {}
// CHECK: "\01?bar@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@NA@2@@Z"

void spam(NA::Y<NA::X> x) {}
// CHECK: "\01?spam@NB@PR13207@@YAXV?$Y@VX@NA@PR13207@@@NA@2@@Z"

// The tests below currently fail:
void foobar(NA::Y<Y<X> > a, Y<Y<X> >) {}
// CURRENT: "\01?foobar@NB@PR13207@@YAXV?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@12@@Z"
// CORRECT: "\01?foobar@NB@PR13207@@YAXV?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V312@@Z"

void foobarspam(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >) {}
// CURRENT: "\01?foobarspam@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@12@@Z"
// CORRECT: "\01?foobarspam@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@@Z"

void foobarbaz(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >, Y<Y<X> > c) {}
// CURRENT: "\01?foobarbaz@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@12@2@Z"
// CORRECT: "\01?foobarbaz@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@2@Z"

void foobarbazqux(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >, Y<Y<X> > c , NA::Y<Y<Y<X> > > d) {}
// CURRENT: "\01?foobarbazqux@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@12@2V?$Y@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NB@PR13207@@@32@@Z"
// CORRECT: "\01?foobarbazqux@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@2V?$Y@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NB@PR13207@@@52@@Z"
}

namespace NC {
class X {};
template<class T> class Y {};

void foo(Y<NB::X> x) {}
// CHECK: "\01?foo@NC@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@@Z"

void foobar(NC::Y<NB::Y<NA::Y<NA::X> > > x) {}
// CHECK: "\01?foobar@NC@PR13207@@YAXV?$Y@V?$Y@V?$Y@VX@NA@PR13207@@@NA@PR13207@@@NB@PR13207@@@12@@Z"
}
}
