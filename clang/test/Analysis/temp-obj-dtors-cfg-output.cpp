// RUN: %clang_cc1 -analyze -cfg-dump -cfg-add-implicit-dtors -cfg-add-initializers %s 2>&1 | FileCheck %s
// XPASS: *

class A {
public:
  A() {}
  ~A() {}

  static A make() { return A(); }

  operator bool() { return false; }
  operator int() { return 0; }
};

class B {
public:
  B() {}
  ~B() {}

  operator bool() { return true; }
  operator int() { return 1; }
  operator A() { return A(); }
};

void foo(int);
void foo(bool);
void foo(const A&);

void test_binary() {
  int a = int(A()) + int(B());
  foo(int(A()) + int(B()));
  int b;
}

void test_and() {
  bool a = A() && B();
  foo(A() && B());
  int b;
}

void test_or() {
  bool a = A() || B();
  foo(A() || B());
  int b;
}

void test_cond() {
  A a = B() ? A() : A(B());
  if (B()) { foo(0); } else { foo(0); }
  int b;
}

void test_cond_cref() {
  const A& a = B() ? A() : A(B());
  foo(B() ? A() : A(B()));
  int b;
}

void test_cond_implicit() {
  A a = A() ?: A();
  int b;
}

void test_cond_implicit_cref() {
  const A& a = A() ?: A();
  foo(A() ?: A());
  int b;
}

void test_copy_init() {
  A a = A();
  int b;
}

void test_cref_init() {
  const A& a = A();
  foo(A());
  int b;
}

void test_call_copy_init() {
  A a = A::make();
  int b;
}

void test_call_cref_init() {
  const A& a = A::make();
  foo(A::make());
  int b;
}

void test_assign() {
  int a;
  a = A();
  int b;
}

class TestCtorInits {
  int a;
  int b;
public:
  TestCtorInits();
};

TestCtorInits::TestCtorInits()
  : a(int(A()) + int(B()))
  , b() {}

// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: A()
// CHECK:      2: [B1.1].operator int()
// CHECK:      3: B()
// CHECK:      4: [B1.3].operator int()
// CHECK:      5: int a = int(A().operator int()) + int(B().operator int());
// CHECK:      6: ~B() (Temporary object destructor)
// CHECK:      7: ~A() (Temporary object destructor)
// CHECK:      8: A()
// CHECK:      9: [B1.8].operator int()
// CHECK:     10: B()
// CHECK:     11: [B1.10].operator int()
// CHECK:     12: foo(int([B1.9]) + int([B1.11]))
// CHECK:     13: ~B() (Temporary object destructor)
// CHECK:     14: ~A() (Temporary object destructor)
// CHECK:     15: int b;
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B10 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B8
// CHECK: [ B1 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: int b;
// CHECK:    Predecessors (2): B2 B3
// CHECK:    Successors (1): B0
// CHECK: [ B2 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B3
// CHECK:    Successors (1): B1
// CHECK: [ B3 ]
// CHECK:      1: [B4.3] && [B5.2]
// CHECK:      2: foo([B3.1])
// CHECK:      T: [B4.3] && ...
// CHECK:    Predecessors (2): B5 B4
// CHECK:    Successors (2): B2 B1
// CHECK: [ B4 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: A()
// CHECK:      3: [B4.2].operator _Bool()
// CHECK:      T: [B4.3] && ...
// CHECK:    Predecessors (2): B6 B7
// CHECK:    Successors (2): B5 B3
// CHECK: [ B5 ]
// CHECK:      1: B()
// CHECK:      2: [B5.1].operator _Bool()
// CHECK:    Predecessors (1): B4
// CHECK:    Successors (1): B3
// CHECK: [ B6 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B4
// CHECK: [ B7 ]
// CHECK:      1: [B8.2] && [B9.2]
// CHECK:      2: bool a = A().operator _Bool() && B().operator _Bool();
// CHECK:      T: [B8.2] && ...
// CHECK:    Predecessors (2): B9 B8
// CHECK:    Successors (2): B6 B4
// CHECK: [ B8 ]
// CHECK:      1: A()
// CHECK:      2: [B8.1].operator _Bool()
// CHECK:      T: [B8.2] && ...
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (2): B9 B7
// CHECK: [ B9 ]
// CHECK:      1: B()
// CHECK:      2: [B9.1].operator _Bool()
// CHECK:    Predecessors (1): B8
// CHECK:    Successors (1): B7
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B10 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B8
// CHECK: [ B1 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: int b;
// CHECK:    Predecessors (2): B2 B3
// CHECK:    Successors (1): B0
// CHECK: [ B2 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B3
// CHECK:    Successors (1): B1
// CHECK: [ B3 ]
// CHECK:      1: [B4.3] || [B5.2]
// CHECK:      2: foo([B3.1])
// CHECK:      T: [B4.3] || ...
// CHECK:    Predecessors (2): B5 B4
// CHECK:    Successors (2): B1 B2
// CHECK: [ B4 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: A()
// CHECK:      3: [B4.2].operator _Bool()
// CHECK:      T: [B4.3] || ...
// CHECK:    Predecessors (2): B6 B7
// CHECK:    Successors (2): B3 B5
// CHECK: [ B5 ]
// CHECK:      1: B()
// CHECK:      2: [B5.1].operator _Bool()
// CHECK:    Predecessors (1): B4
// CHECK:    Successors (1): B3
// CHECK: [ B6 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B4
// CHECK: [ B7 ]
// CHECK:      1: [B8.2] || [B9.2]
// CHECK:      2: bool a = A().operator _Bool() || B().operator _Bool();
// CHECK:      T: [B8.2] || ...
// CHECK:    Predecessors (2): B9 B8
// CHECK:    Successors (2): B4 B6
// CHECK: [ B8 ]
// CHECK:      1: A()
// CHECK:      2: [B8.1].operator _Bool()
// CHECK:      T: [B8.2] || ...
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (2): B7 B9
// CHECK: [ B9 ]
// CHECK:      1: B()
// CHECK:      2: [B9.1].operator _Bool()
// CHECK:    Predecessors (1): B8
// CHECK:    Successors (1): B7
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B11 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B10
// CHECK: [ B1 ]
// CHECK:      1: int b;
// CHECK:      2: [B7.2].~A() (Implicit destructor)
// CHECK:    Predecessors (2): B2 B3
// CHECK:    Successors (1): B0
// CHECK: [ B2 ]
// CHECK:      1: foo(0)
// CHECK:    Predecessors (1): B4
// CHECK:    Successors (1): B1
// CHECK: [ B3 ]
// CHECK:      1: foo(0)
// CHECK:    Predecessors (1): B4
// CHECK:    Successors (1): B1
// CHECK: [ B4 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:      2: B()
// CHECK:      3: [B4.2].operator _Bool()
// CHECK:      4: ~B() (Temporary object destructor)
// CHECK:      T: if [B4.3]
// CHECK:    Predecessors (2): B5 B6
// CHECK:    Successors (2): B3 B2
// CHECK: [ B5 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B4
// CHECK: [ B6 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:      3: ~A() (Temporary object destructor)
// CHECK:      4: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B4
// CHECK: [ B7 ]
// CHECK:      1: [B10.2] ? [B8.2] : [B9.3]
// CHECK:      2: A a = B().operator _Bool() ? A() : A(B().operator A());
// CHECK:      T: [B10.2] ? ... : ...
// CHECK:    Predecessors (2): B8 B9
// CHECK:    Successors (2): B5 B6
// CHECK: [ B8 ]
// CHECK:      1: A()
// CHECK:      2: [B8.1] (BindTemporary)
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (1): B7
// CHECK: [ B9 ]
// CHECK:      1: B()
// CHECK:      2: [B9.1].operator A()
// CHECK:      3: A([B9.2]) (BindTemporary)
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (1): B7
// CHECK: [ B10 ]
// CHECK:      1: B()
// CHECK:      2: [B10.1].operator _Bool()
// CHECK:      T: [B10.2] ? ... : ...
// CHECK:    Predecessors (1): B11
// CHECK:    Successors (2): B8 B9
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B14 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B13
// CHECK: [ B1 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:      2: int b;
// CHECK:      3: [B10.2].~A() (Implicit destructor)
// CHECK:    Predecessors (2): B2 B3
// CHECK:    Successors (1): B0
// CHECK: [ B2 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:    Predecessors (1): B4
// CHECK:    Successors (1): B1
// CHECK: [ B3 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:      3: ~A() (Temporary object destructor)
// CHECK:      4: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B4
// CHECK:    Successors (1): B1
// CHECK: [ B4 ]
// CHECK:      1: [B7.3] ? [B5.2] : [B6.3]
// CHECK:      2: foo([B4.1])
// CHECK:      T: [B7.3] ? ... : ...
// CHECK:    Predecessors (2): B5 B6
// CHECK:    Successors (2): B2 B3
// CHECK: [ B5 ]
// CHECK:      1: A()
// CHECK:      2: [B5.1] (BindTemporary)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B4
// CHECK: [ B6 ]
// CHECK:      1: B()
// CHECK:      2: [B6.1].operator A()
// CHECK:      3: A([B6.2]) (BindTemporary)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B4
// CHECK: [ B7 ]
// CHECK:      1: ~B() (Temporary object destructor)
// CHECK:      2: B()
// CHECK:      3: [B7.2].operator _Bool()
// CHECK:      T: [B7.3] ? ... : ...
// CHECK:    Predecessors (2): B8 B9
// CHECK:    Successors (2): B5 B6
// CHECK: [ B8 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (1): B7
// CHECK: [ B9 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:      3: ~B() (Temporary object destructor)
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (1): B7
// CHECK: [ B10 ]
// CHECK:      1: [B13.2] ? [B11.2] : [B12.3]
// CHECK:      2: const A &a = B().operator _Bool() ? A() : A(B().operator A());
// CHECK:      T: [B13.2] ? ... : ...
// CHECK:    Predecessors (2): B11 B12
// CHECK:    Successors (2): B8 B9
// CHECK: [ B11 ]
// CHECK:      1: A()
// CHECK:      2: [B11.1] (BindTemporary)
// CHECK:    Predecessors (1): B13
// CHECK:    Successors (1): B10
// CHECK: [ B12 ]
// CHECK:      1: B()
// CHECK:      2: [B12.1].operator A()
// CHECK:      3: A([B12.2]) (BindTemporary)
// CHECK:    Predecessors (1): B13
// CHECK:    Successors (1): B10
// CHECK: [ B13 ]
// CHECK:      1: B()
// CHECK:      2: [B13.1].operator _Bool()
// CHECK:      T: [B13.2] ? ... : ...
// CHECK:    Predecessors (1): B14
// CHECK:    Successors (2): B11 B12
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B6 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B5
// CHECK: [ B1 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: int b;
// CHECK:      3: [B3.2].~A() (Implicit destructor)
// CHECK:    Predecessors (2): B3 B2
// CHECK:    Successors (1): B0
// CHECK: [ B2 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:    Predecessors (1): B3
// CHECK:    Successors (1): B1
// CHECK: [ B3 ]
// CHECK:      1: [B5.2] ?: [B4.2]
// CHECK:      2: A a = A().operator _Bool() ?: A();
// CHECK:      T: [B5.2] ? ... : ...
// CHECK:    Predecessors (2): B5 B4
// CHECK:    Successors (2): B1 B2
// CHECK: [ B4 ]
// CHECK:      1: A()
// CHECK:      2: [B4.1] (BindTemporary)
// CHECK:    Predecessors (1): B5
// CHECK:    Successors (1): B3
// CHECK: [ B5 ]
// CHECK:      1: A()
// CHECK:      2: [B5.1].operator _Bool()
// CHECK:      T: [B5.2] ? ... : ...
// CHECK:    Predecessors (1): B6
// CHECK:    Successors (2): B3 B4
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B10 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B9
// CHECK: [ B1 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: int b;
// CHECK:      3: [B7.2].~A() (Implicit destructor)
// CHECK:    Predecessors (2): B3 B2
// CHECK:    Successors (1): B0
// CHECK: [ B2 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: ~A() (Temporary object destructor)
// CHECK:    Predecessors (1): B3
// CHECK:    Successors (1): B1
// CHECK: [ B3 ]
// CHECK:      1: [B5.3] ?: [B4.2]
// CHECK:      2: foo([B3.1])
// CHECK:      T: [B5.3] ? ... : ...
// CHECK:    Predecessors (2): B5 B4
// CHECK:    Successors (2): B1 B2
// CHECK: [ B4 ]
// CHECK:      1: A()
// CHECK:      2: [B4.1] (BindTemporary)
// CHECK:    Predecessors (1): B5
// CHECK:    Successors (1): B3
// CHECK: [ B5 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:      2: A()
// CHECK:      3: [B5.2].operator _Bool()
// CHECK:      T: [B5.3] ? ... : ...
// CHECK:    Predecessors (2): B7 B6
// CHECK:    Successors (2): B3 B4
// CHECK: [ B6 ]
// CHECK:      1: ~A() (Temporary object destructor)
// CHECK:    Predecessors (1): B7
// CHECK:    Successors (1): B5
// CHECK: [ B7 ]
// CHECK:      1: [B9.2] ?: [B8.2]
// CHECK:      2: const A &a = A().operator _Bool() ?: A();
// CHECK:      T: [B9.2] ? ... : ...
// CHECK:    Predecessors (2): B9 B8
// CHECK:    Successors (2): B5 B6
// CHECK: [ B8 ]
// CHECK:      1: A()
// CHECK:      2: [B8.1] (BindTemporary)
// CHECK:    Predecessors (1): B9
// CHECK:    Successors (1): B7
// CHECK: [ B9 ]
// CHECK:      1: A()
// CHECK:      2: [B9.1].operator _Bool()
// CHECK:      T: [B9.2] ? ... : ...
// CHECK:    Predecessors (1): B10
// CHECK:    Successors (2): B7 B8
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: A()
// CHECK:      2: A a = A();
// CHECK:      3: ~A() (Temporary object destructor)
// CHECK:      4: int b;
// CHECK:      5: [B1.2].~A() (Implicit destructor)
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: A()
// CHECK:      2: const A &a = A();
// CHECK:      3: A()
// CHECK:      4: foo([B1.3])
// CHECK:      5: ~A() (Temporary object destructor)
// CHECK:      6: int b;
// CHECK:      7: [B1.2].~A() (Implicit destructor)
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: A::make()
// CHECK:      2: A a = A::make();
// CHECK:      3: ~A() (Temporary object destructor)
// CHECK:      4: int b;
// CHECK:      5: [B1.2].~A() (Implicit destructor)
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: A::make()
// CHECK:      2: const A &a = A::make();
// CHECK:      3: A::make()
// CHECK:      4: foo([B1.3])
// CHECK:      5: ~A() (Temporary object destructor)
// CHECK:      6: int b;
// CHECK:      7: [B1.2].~A() (Implicit destructor)
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: int a;
// CHECK:      2: A()
// CHECK:      3: [B1.2].operator int()
// CHECK:      4: a = [B1.3]
// CHECK:      5: ~A() (Temporary object destructor)
// CHECK:      6: int b;
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
// CHECK: [ B2 (ENTRY) ]
// CHECK:    Predecessors (0):
// CHECK:    Successors (1): B1
// CHECK: [ B1 ]
// CHECK:      1: A()
// CHECK:      2: [B1.1].operator int()
// CHECK:      3: B()
// CHECK:      4: [B1.3].operator int()
// CHECK:      5: a(int([B1.2]) + int([B1.4])) (Member initializer)
// CHECK:      6: ~B() (Temporary object destructor)
// CHECK:      7: ~A() (Temporary object destructor)
// CHECK:      8: b(/*implicit*/int()) (Member initializer)
// CHECK:    Predecessors (1): B2
// CHECK:    Successors (1): B0
// CHECK: [ B0 (EXIT) ]
// CHECK:    Predecessors (1): B1
// CHECK:    Successors (0):
