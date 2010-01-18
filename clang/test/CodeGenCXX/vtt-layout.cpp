// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

namespace Test1 {
struct A { };

struct B : virtual A { 
  virtual void f();
};

void B::f() { } 
}

// Test1::B should just have a single entry in its VTT, which points to the vtable.
// CHECK: @_ZTTN5Test11BE = constant [1 x i8*] [i8* bitcast (i8** getelementptr inbounds ([4 x i8*]* @_ZTVN5Test11BE, i64 0, i64 3) to i8*)]

namespace Test2 {
  struct A { };

  struct B : A { virtual void f(); };
  struct C : virtual B { };

  C c;
}

// Check that we don't add a secondary virtual pointer for Test2::A, since Test2::A doesn't have any virtual member functions or bases.
// CHECK: @_ZTTN5Test21CE = weak_odr constant [2 x i8*] [i8* bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN5Test21CE, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN5Test21CE, i64 0, i64 4) to i8*)] ; <[2 x i8*]*> [#uses=0]
