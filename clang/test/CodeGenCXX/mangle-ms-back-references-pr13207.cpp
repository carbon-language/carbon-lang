// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

// NOTE on the "CURRENT" prefix: some things are mangled incorrectly as of
// writing. If you find a CURRENT-test that fails with your patch, please test
// if your patch has actually fixed a problem in the mangler and replace the
// corresponding CORRECT line with a CHECK.
// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck -check-prefix CURRENT %s

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
