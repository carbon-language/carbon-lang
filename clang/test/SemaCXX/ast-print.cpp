// RUN: %clang_cc1 -ast-print %s | FileCheck %s

// CHECK: r;
// CHECK-NEXT: (r->method());
struct MyClass
{
    void method() {}
};

struct Reference
{
    MyClass* object;
    MyClass* operator ->() { return object; }
};

void test1() {
    Reference r;
    (r->method());
}

// CHECK: if (int a = 1)
// CHECK:  while (int a = 1)
// CHECK:  switch (int a = 1)

void test2()
{
    if (int a = 1) { }
    while (int a = 1) { }
    switch (int a = 1) { }
}

// CHECK: new (1) int;
void *operator new (typeof(sizeof(1)), int, int = 2);
void test3() {
  new (1) int;
}

// CHECK: new X;
struct X {
  void *operator new (typeof(sizeof(1)), int = 2);
};
void test4() { new X; }

// CHECK: for (int i = 2097, j = 42; false;)
void test5() {
  for (int i = 2097, j = 42; false;) {}
}

// CHECK: test6fn((int &)y);
void test6fn(int& x);
void test6() {
    unsigned int y = 0;
    test6fn((int&)y);
}

// CHECK: S s( 1, 2 );

template <class S> void test7()
{
    S s( 1,2 );
}


// CHECK: t.~T();

template <typename T> void test8(T t) { t.~T(); }


// CHECK:      enum E {
// CHECK-NEXT:  A,
// CHECK-NEXT:  B,
// CHECK-NEXT:  C
// CHECK-NEXT:  };
// CHECK-NEXT: {{^[ ]+}}E a = A;

struct test9
{
    void f()
    {
        enum E { A, B, C };
        E a = A;
    }
};

namespace test10 {
  namespace M {
    template<typename T>
    struct X {
      enum { value };
    };
  }
}

typedef int INT;

// CHECK: test11
// CHECK-NEXT: return test10::M::X<INT>::value;
int test11() {
  return test10::M::X<INT>::value;
}


struct DefaultArgClass
{
  DefaultArgClass(int a = 1) {}
};

struct NoArgClass
{
  NoArgClass() {}
};

// CHECK: test12
// CHECK-NEXT: DefaultArgClass useDefaultArg;
// CHECK-NEXT: DefaultArgClass overrideDefaultArg(1);
// CHECK-NEXT: NoArgClass noArg;
void test12() {
  DefaultArgClass useDefaultArg;
  DefaultArgClass overrideDefaultArg(1);
  NoArgClass noArg;
}

