// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// Test1::B should just have a single entry in its VTT, which points to the vtable.
namespace Test1 {
struct A { };

struct B : virtual A { 
  virtual void f();
};

void B::f() { } 
}

// Check that we don't add a secondary virtual pointer for Test2::A, since Test2::A doesn't have any virtual member functions or bases.
namespace Test2 {
  struct A { };

  struct B : A { virtual void f(); };
  struct C : virtual B { };

  C c;
}

// This is the sample from the C++ Itanium ABI, p2.6.2.
namespace Test3 {
  class A1 { int i; };
  class A2 { int i; virtual void f(); };
  class V1 : public A1, public A2 { int i; };
  class B1 { int i; };
  class B2 { int i; };
  class V2 : public B1, public B2, public virtual V1 { int i; };
  class V3 {virtual void g(); };
  class C1 : public virtual V1 { int i; };
  class C2 : public virtual V3, virtual V2 { int i; };
  class X1 { int i; };
  class C3 : public X1 { int i; };
  class D : public C1, public C2, public C3 { int i;  };
  
  D d;
}

// This is the sample from the C++ Itanium ABI, p2.6.2, with the change suggested
// (making A2 a virtual base of V1)
namespace Test4 {
  class A1 { int i; };
  class A2 { int i; virtual void f(); };
  class V1 : public A1, public virtual A2 { int i; };
  class B1 { int i; };
  class B2 { int i; };
  class V2 : public B1, public B2, public virtual V1 { int i; };
  class V3 {virtual void g(); };
  class C1 : public virtual V1 { int i; };
  class C2 : public virtual V3, virtual V2 { int i; };
  class X1 { int i; };
  class C3 : public X1 { int i; };
  class D : public C1, public C2, public C3 { int i;  };
  
  D d;
}

// CHECK: @_ZTTN5Test11BE = constant [1 x i8*] [i8* bitcast (i8** getelementptr inbounds ([4 x i8*]* @_ZTVN5Test11BE, i64 0, i64 3) to i8*)]
// CHECK: @_ZTTN5Test41DE = weak_odr constant [19 x i8*] [i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTVN5Test41DE, i64 0, i64 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*]* @_ZTCN5Test41DE0_NS_2C1E, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*]* @_ZTCN5Test41DE0_NS_2C1E, i64 0, i64 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*]* @_ZTCN5Test41DE0_NS_2C1E, i64 0, i64 10) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i64 0, i64 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i64 0, i64 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i64 0, i64 12) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i64 0, i64 15) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTCN5Test41DE16_NS_2C2E, i64 0, i64 18) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTVN5Test41DE, i64 0, i64 17) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTVN5Test41DE, i64 0, i64 20) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTVN5Test41DE, i64 0, i64 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTVN5Test41DE, i64 0, i64 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([25 x i8*]* @_ZTVN5Test41DE, i64 1, i64 0) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*]* @_ZTCN5Test41DE40_NS_2V1E, i64 0, i64 3) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*]* @_ZTCN5Test41DE40_NS_2V1E, i64 0, i64 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*]* @_ZTCN5Test41DE72_NS_2V2E, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*]* @_ZTCN5Test41DE72_NS_2V2E, i64 0, i64 7) to i8*), i8* bitcast (i8** getelementptr inbounds ([11 x i8*]* @_ZTCN5Test41DE72_NS_2V2E, i64 0, i64 10) to i8*)] 
// CHECK: @_ZTTN5Test31DE = weak_odr constant [13 x i8*] [i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTVN5Test31DE, i64 0, i64 5) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*]* @_ZTCN5Test31DE0_NS_2C1E, i64 0, i64 3) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*]* @_ZTCN5Test31DE0_NS_2C1E, i64 0, i64 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i64 0, i64 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i64 0, i64 6) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i64 0, i64 10) to i8*), i8* bitcast (i8** getelementptr inbounds ([14 x i8*]* @_ZTCN5Test31DE16_NS_2C2E, i64 0, i64 13) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTVN5Test31DE, i64 0, i64 15) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTVN5Test31DE, i64 0, i64 11) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTVN5Test31DE, i64 0, i64 11) to i8*), i8* bitcast (i8** getelementptr inbounds ([19 x i8*]* @_ZTVN5Test31DE, i64 1, i64 0) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*]* @_ZTCN5Test31DE64_NS_2V2E, i64 0, i64 3) to i8*), i8* bitcast (i8** getelementptr inbounds ([7 x i8*]* @_ZTCN5Test31DE64_NS_2V2E, i64 0, i64 6) to i8*)] 
// CHECK: @_ZTTN5Test21CE = weak_odr constant [2 x i8*] [i8* bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN5Test21CE, i64 0, i64 4) to i8*), i8* bitcast (i8** getelementptr inbounds ([5 x i8*]* @_ZTVN5Test21CE, i64 0, i64 4) to i8*)] 
