// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

namespace {
  struct A {
    virtual void f() { }
  };
}

void f() { A b; }

struct B {
  B();
  virtual void f();
};

B::B() { }

struct C {
  C();
  virtual void f() { } 
};

C::C() { } 

struct D {
  virtual void f();
};

void D::f() { }

static struct : D { } e;

template<typename T>
struct E {
  virtual ~E();
};

template<typename T> E<T>::~E() { }

template<>
struct E<char> {
  virtual void anchor();
};

void E<char>::anchor() { }

template struct E<short>;
extern template struct E<int>;

void use_E() {
  E<int> ei;
  (void)ei;
  E<long> el;
  (void)el;
}

// B has a key function that is not defined in this translation unit so its vtable
// has external linkage.
// CHECK: @_ZTV1B = external constant

// C has no key function, so its vtable should have weak_odr linkage.
// CHECK: @_ZTS1C = weak_odr constant
// CHECK: @_ZTI1C = weak_odr constant
// CHECK: @_ZTV1C = weak_odr constant

// D has a key function that is defined in this translation unit so its vtable is
// defined in the translation unit.
// CHECK: @_ZTS1D = constant
// CHECK: @_ZTI1D = constant
// CHECK: @_ZTV1D = constant

// E<char> is an explicit specialization with a key function defined
// in this translation unit, so its vtable should have external
// linkage.
// CHECK: @_ZTV1EIcE = constant

// E<short> is an explicit template instantiation with a key function
// defined in this translation unit, so its vtable should have
// weak_odr linkage.
// CHECK: @_ZTV1EIsE = weak_odr constant

// E<long> is an implicit template instantiation with a key function
// defined in this translation unit, so its vtable should have
// weak_odr linkage.
// CHECK: @_ZTV1EIlE = weak_odr constant

// The anonymous struct for e has no linkage, so the vtable should have
// internal linkage.
// CHECK: @"_ZTS3$_0" = internal constant
// CHECK: @"_ZTI3$_0" = internal constant
// CHECK: @"_ZTV3$_0" = internal constant

// E<int> is an explicit template instantiation declaration. It has a
// key function that is not instantiation, so we should only reference
// its vtable, not define it.
// CHECK: @_ZTV1EIiE = external constant

// The A vtable should have internal linkage since it is inside an anonymous 
// namespace.
// CHECK: @_ZTSN12_GLOBAL__N_11AE = internal constant
// CHECK: @_ZTIN12_GLOBAL__N_11AE = internal constant
// CHECK: @_ZTVN12_GLOBAL__N_11AE = internal constant


