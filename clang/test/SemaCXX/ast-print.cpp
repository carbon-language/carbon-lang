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
  DefaultArgClass(int a, int b, int c = 1) {}
};

struct NoArgClass
{
  NoArgClass() {}
};

struct VirualDestrClass
{
  VirualDestrClass(int arg);
  virtual ~VirualDestrClass();
};

struct ConstrWithCleanupsClass
{
  ConstrWithCleanupsClass(const VirualDestrClass& cplx = VirualDestrClass(42));
};

// CHECK: test12
// CHECK-NEXT: DefaultArgClass useDefaultArg;
// CHECK-NEXT: DefaultArgClass overrideDefaultArg(1);
// CHECK-NEXT: DefaultArgClass(1, 2);
// CHECK-NEXT: DefaultArgClass(1, 2, 3);
// CHECK-NEXT: NoArgClass noArg;
// CHECK-NEXT: ConstrWithCleanupsClass cwcNoArg;
// CHECK-NEXT: ConstrWithCleanupsClass cwcOverrideArg(48);
// CHECK-NEXT: ConstrWithCleanupsClass cwcExplicitArg(VirualDestrClass(56));
void test12() {
  DefaultArgClass useDefaultArg;
  DefaultArgClass overrideDefaultArg(1);
  DefaultArgClass tempWithDefaultArg = DefaultArgClass(1, 2);
  DefaultArgClass tempWithExplictArg = DefaultArgClass(1, 2, 3);
  NoArgClass noArg;
  ConstrWithCleanupsClass cwcNoArg;
  ConstrWithCleanupsClass cwcOverrideArg(48);
  ConstrWithCleanupsClass cwcExplicitArg(VirualDestrClass(56));
}

// CHECK: void test13() {
// CHECK:   _Atomic(int) i;
// CHECK:   __c11_atomic_init(&i, 0);
// CHECK:   __c11_atomic_load(&i, 0);
// CHECK: }
void test13() {
  _Atomic(int) i;
  __c11_atomic_init(&i, 0);
  __c11_atomic_load(&i, 0);
}


// CHECK: void test14() {
// CHECK:     struct X {
// CHECK:         union {
// CHECK:             int x;
// CHECK:         } x;
// CHECK:     };
// CHECK: }
void test14() {
  struct X { union { int x; } x; };
}


// CHECK: float test15() {
// CHECK:     return __builtin_asinf(1.F);
// CHECK: }
// CHECK-NOT: extern "C"
float test15() {
  return __builtin_asinf(1.0F);
}
