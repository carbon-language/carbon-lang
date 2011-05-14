// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

namespace Test1 {

// Check that we don't initialize the vtable pointer in A::~A(), since the destructor body is trivial.
struct A {
  virtual void f();
  ~A();
};

// CHECK: define void @_ZN5Test11AD2Ev
// CHECK-NOT: store i8** getelementptr inbounds ([3 x i8*]* @_ZTVN5Test11AE, i64 0, i64 2), i8***
A::~A() 
{
}

}

namespace Test2 {

// Check that we do initialize the vtable pointer in A::~A() since the destructor body isn't trivial.
struct A {
  virtual void f();
  ~A();
};

// CHECK: define void @_ZN5Test21AD2Ev
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTVN5Test21AE, i64 0, i64 2), i8***
A::~A() {
  f();
}

}

namespace Test3 {

// Check that we don't initialize the vtable pointer in A::~A(), since the destructor body is trivial
// and Field's destructor body is also trivial.
struct Field {
  ~Field() { }
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK: define void @_ZN5Test31AD2Ev
// CHECK-NOT: store i8** getelementptr inbounds ([3 x i8*]* @_ZTVN5Test31AE, i64 0, i64 2), i8***
A::~A() {
  
}

}

namespace Test4 {

// Check that we do initialize the vtable pointer in A::~A(), since Field's destructor body
// isn't trivial.

void f();

struct Field {
  ~Field() { f(); }
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK: define void @_ZN5Test41AD2Ev
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTVN5Test41AE, i64 0, i64 2), i8***
A::~A()
{
}

}

namespace Test5 {

// Check that we do initialize the vtable pointer in A::~A(), since Field's destructor isn't
// available in this translation unit.

struct Field {
  ~Field();
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK: define void @_ZN5Test51AD2Ev
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTVN5Test51AE, i64 0, i64 2), i8***
A::~A()
{
}

}