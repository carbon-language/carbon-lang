// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
#include <typeinfo>

// CHECK: _ZTS1B = constant
// CHECK: _ZTS1A = weak_odr constant
// CHECK: _ZTI1A = weak_odr constant
// CHECK: _ZTI1B = constant
// CHECK: _ZTSP1C = internal constant
// CHECK: _ZTS1C = internal constant
// CHECK: _ZTI1C = internal constant
// CHECK: _ZTIP1C = internal constant
// CHECK: _ZTSPP1C = internal constant
// CHECK: _ZTIPP1C = internal constant
// CHECK: _ZTSM1Ci = internal constant
// CHECK: _ZTIM1Ci = internal constant
// CHECK: _ZTSPM1Ci = internal constant
// CHECK: _ZTIPM1Ci = internal constant
// CHECK: _ZTSM1CS_ = internal constant
// CHECK: _ZTIM1CS_ = internal constant
// CHECK: _ZTSM1CPS_ = internal constant
// CHECK: _ZTIM1CPS_ = internal constant
// CHECK: _ZTSM1A1C = internal constant
// CHECK: _ZTIM1A1C = internal constant
// CHECK: _ZTSM1AP1C = internal constant
// CHECK: _ZTIM1AP1C = internal constant

// CHECK: _ZTSN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTIN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTSPN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTIPN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTSFN12_GLOBAL__N_11DEvE = internal constant
// CHECK: _ZTIFN12_GLOBAL__N_11DEvE = internal constant
// CHECK: _ZTSFvN12_GLOBAL__N_11DEE = internal constant
// CHECK: _ZTIFvN12_GLOBAL__N_11DEE = internal constant

// CHECK: _ZTSPFvvE = weak_odr constant
// CHECK: _ZTSFvvE = weak_odr constant
// CHECK: _ZTIFvvE = weak_odr
// CHECK: _ZTIPFvvE = weak_odr constant

// CHECK: _ZTSN12_GLOBAL__N_11EE = internal constant
// CHECK: _ZTIN12_GLOBAL__N_11EE = internal constant

// A has no key function, so its RTTI data should be weak_odr.
struct A { };

// B has a key function defined in the translation unit, so the RTTI data should
// be emitted in this translation unit and have external linkage.
struct B : A {
  virtual void f();
};
void B::f() { }

// C is an incomplete class type, so any direct or indirect pointer types should have 
// internal linkage, as should the type info for C itself.
struct C;

void t1() {
  (void)typeid(C*);
  (void)typeid(C**);
  (void)typeid(int C::*);
  (void)typeid(int C::**);
  (void)typeid(C C::*);
  (void)typeid(C *C::*);
  (void)typeid(C A::*);
  (void)typeid(C* A::*);
}

namespace {
  // D is inside an anonymous namespace, so all type information related to D should have
  // internal linkage.
  struct D { };
  
  // E is also inside an anonymous namespace.
  enum E { };
  
};

const D getD();

const std::type_info &t2() {
  (void)typeid(const D);
  (void)typeid(D *);
  (void)typeid(D (*)());
  (void)typeid(void (*)(D));
  // The exception specification is not part of the RTTI descriptor, so it should not have
  // internal linkage.
  (void)typeid(void (*)() throw (D));
  
  (void)typeid(E);
  
  // CHECK: _ZTIN12_GLOBAL__N_11DE to
  return typeid(getD());  
}
