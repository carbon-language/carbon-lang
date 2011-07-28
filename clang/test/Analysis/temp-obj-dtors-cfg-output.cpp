// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCFG -cfg-add-implicit-dtors -cfg-add-initializers %s 2>&1 | FileCheck %s
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

// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: A()
// CHECK:       2: [B1.1] (BindTemporary)
// CHECK:       3: [B1.2].operator int
// CHECK:       4: [B1.3]()
// CHECK:       5: [B1.4]
// CHECK:       6: int([B1.5])
// CHECK:       7: B()
// CHECK:       8: [B1.7] (BindTemporary)
// CHECK:       9: [B1.8].operator int
// CHECK:      10: [B1.9]()
// CHECK:      11: [B1.10]
// CHECK:      12: int([B1.11])
// CHECK:      13: [B1.6] + [B1.12]
// CHECK:      14: int a = int(A().operator int()) + int(B().operator int());
// CHECK:      15: ~B() (Temporary object destructor)
// CHECK:      16: ~A() (Temporary object destructor)
// CHECK:      17: A()
// CHECK:      18: [B1.17] (BindTemporary)
// CHECK:      19: [B1.18].operator int
// CHECK:      20: [B1.19]()
// CHECK:      21: [B1.20]
// CHECK:      22: int([B1.21])
// CHECK:      23: B()
// CHECK:      24: [B1.23] (BindTemporary)
// CHECK:      25: [B1.24].operator int
// CHECK:      26: [B1.25]()
// CHECK:      27: [B1.26]
// CHECK:      28: int([B1.27])
// CHECK:      29: [B1.22] + [B1.28]
// CHECK:      30: foo
// CHECK:      31: [B1.30]
// CHECK:      32: [B1.31]([B1.29])
// CHECK:      33: ~B() (Temporary object destructor)
// CHECK:      34: ~A() (Temporary object destructor)
// CHECK:      35: int b;
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B10 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B8
// CHECK:  [ B1 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: int b;
// CHECK:     Predecessors (2): B2 B3
// CHECK:     Successors (1): B0
// CHECK:  [ B2 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B3
// CHECK:     Successors (1): B1
// CHECK:  [ B3 ]
// CHECK:       1: [B4.5] && [B5.4]
// CHECK:       2: foo
// CHECK:       3: [B3.2]
// CHECK:       4: [B3.3]([B3.1])
// CHECK:       T: [B4.5] && ...
// CHECK:     Predecessors (2): B5 B4
// CHECK:     Successors (2): B2 B1
// CHECK:  [ B4 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: A()
// CHECK:       3: [B4.2] (BindTemporary)
// CHECK:       4: [B4.3].operator _Bool
// CHECK:       5: [B4.4]()
// CHECK:       T: [B4.5] && ...
// CHECK:     Predecessors (2): B6 B7
// CHECK:     Successors (2): B5 B3
// CHECK:  [ B5 ]
// CHECK:       1: B()
// CHECK:       2: [B5.1] (BindTemporary)
// CHECK:       3: [B5.2].operator _Bool
// CHECK:       4: [B5.3]()
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B3
// CHECK:  [ B6 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B7 ]
// CHECK:       1: [B8.4] && [B9.4]
// CHECK:       2: bool a = A().operator _Bool() && B().operator _Bool();
// CHECK:       T: [B8.4] && ...
// CHECK:     Predecessors (2): B9 B8
// CHECK:     Successors (2): B6 B4
// CHECK:  [ B8 ]
// CHECK:       1: A()
// CHECK:       2: [B8.1] (BindTemporary)
// CHECK:       3: [B8.2].operator _Bool
// CHECK:       4: [B8.3]()
// CHECK:       T: [B8.4] && ...
// CHECK:     Predecessors (1): B10
// CHECK:     Successors (2): B9 B7
// CHECK:  [ B9 ]
// CHECK:       1: B()
// CHECK:       2: [B9.1] (BindTemporary)
// CHECK:       3: [B9.2].operator _Bool
// CHECK:       4: [B9.3]()
// CHECK:     Predecessors (1): B8
// CHECK:     Successors (1): B7
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B10 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B8
// CHECK:  [ B1 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: int b;
// CHECK:     Predecessors (2): B2 B3
// CHECK:     Successors (1): B0
// CHECK:  [ B2 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B3
// CHECK:     Successors (1): B1
// CHECK:  [ B3 ]
// CHECK:       1: [B4.5] || [B5.4]
// CHECK:       2: foo
// CHECK:       3: [B3.2]
// CHECK:       4: [B3.3]([B3.1])
// CHECK:       T: [B4.5] || ...
// CHECK:     Predecessors (2): B5 B4
// CHECK:     Successors (2): B1 B2
// CHECK:  [ B4 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: A()
// CHECK:       3: [B4.2] (BindTemporary)
// CHECK:       4: [B4.3].operator _Bool
// CHECK:       5: [B4.4]()
// CHECK:       T: [B4.5] || ...
// CHECK:     Predecessors (2): B6 B7
// CHECK:     Successors (2): B3 B5
// CHECK:  [ B5 ]
// CHECK:       1: B()
// CHECK:       2: [B5.1] (BindTemporary)
// CHECK:       3: [B5.2].operator _Bool
// CHECK:       4: [B5.3]()
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B3
// CHECK:  [ B6 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B7 ]
// CHECK:       1: [B8.4] || [B9.4]
// CHECK:       2: bool a = A().operator _Bool() || B().operator _Bool();
// CHECK:       T: [B8.4] || ...
// CHECK:     Predecessors (2): B9 B8
// CHECK:     Successors (2): B4 B6
// CHECK:  [ B8 ]
// CHECK:       1: A()
// CHECK:       2: [B8.1] (BindTemporary)
// CHECK:       3: [B8.2].operator _Bool
// CHECK:       4: [B8.3]()
// CHECK:       T: [B8.4] || ...
// CHECK:     Predecessors (1): B10
// CHECK:     Successors (2): B7 B9
// CHECK:  [ B9 ]
// CHECK:       1: B()
// CHECK:       2: [B9.1] (BindTemporary)
// CHECK:       3: [B9.2].operator _Bool
// CHECK:       4: [B9.3]()
// CHECK:     Predecessors (1): B8
// CHECK:     Successors (1): B7
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B11 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B10
// CHECK:  [ B1 ]
// CHECK:       1: int b;
// CHECK:       2: [B7.4].~A() (Implicit destructor)
// CHECK:     Predecessors (2): B2 B3
// CHECK:     Successors (1): B0
// CHECK:  [ B2 ]
// CHECK:       1: 0
// CHECK:       2: foo
// CHECK:       3: [B2.2]
// CHECK:       4: [B2.3]([B2.1])
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B3 ]
// CHECK:       1: 0
// CHECK:       2: foo
// CHECK:       3: [B3.2]
// CHECK:       4: [B3.3]([B3.1])
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B4 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:       2: B()
// CHECK:       3: [B4.2] (BindTemporary)
// CHECK:       4: [B4.3].operator _Bool
// CHECK:       5: [B4.4]()
// CHECK:       6: ~B() (Temporary object destructor)
// CHECK:       T: if [B4.5]
// CHECK:     Predecessors (2): B5 B6
// CHECK:     Successors (2): B3 B2
// CHECK:  [ B5 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B6 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:       3: ~A() (Temporary object destructor)
// CHECK:       4: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B7 ]
// CHECK:       1: [B10.4] ? [B8.5] : [B9.13]
// CHECK:       2: [B7.1]
// CHECK:       3: [B7.2]
// CHECK:       4: A a = B().operator _Bool() ? A() : A(B().operator A());
// CHECK:       T: [B10.4] ? ... : ...
// CHECK:     Predecessors (2): B8 B9
// CHECK:     Successors (2): B5 B6
// CHECK:  [ B8 ]
// CHECK:       1: A()
// CHECK:       2: [B8.1] (BindTemporary)
// CHECK:       3: [B8.2]
// CHECK:       4: [B8.3]
// CHECK:       5: [B8.4] (BindTemporary)
// CHECK:     Predecessors (1): B10
// CHECK:     Successors (1): B7
// CHECK:  [ B9 ]
// CHECK:       1: B()
// CHECK:       2: [B9.1] (BindTemporary)
// CHECK:       3: [B9.2].operator A
// CHECK:       4: [B9.3]()
// CHECK:       5: [B9.4] (BindTemporary)
// CHECK:       6: [B9.5]
// CHECK:       7: [B9.6]
// CHECK:       8: [B9.7]
// CHECK:       9: [B9.8] (BindTemporary)
// CHECK:      10: A([B9.9])
// CHECK:      11: [B9.10]
// CHECK:      12: [B9.11]
// CHECK:      13: [B9.12] (BindTemporary)
// CHECK:     Predecessors (1): B10
// CHECK:     Successors (1): B7
// CHECK:  [ B10 ]
// CHECK:       1: B()
// CHECK:       2: [B10.1] (BindTemporary)
// CHECK:       3: [B10.2].operator _Bool
// CHECK:       4: [B10.3]()
// CHECK:       T: [B10.4] ? ... : ...
// CHECK:     Predecessors (1): B11
// CHECK:     Successors (2): B8 B9
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B14 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B13
// CHECK:  [ B1 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:       2: int b;
// CHECK:       3: [B10.4].~A() (Implicit destructor)
// CHECK:     Predecessors (2): B2 B3
// CHECK:     Successors (1): B0
// CHECK:  [ B2 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B3 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:       3: ~A() (Temporary object destructor)
// CHECK:       4: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B4 ]
// CHECK:       1: [B7.5] ? [B5.5] : [B6.13]
// CHECK:       2: [B4.1]
// CHECK:       3: [B4.2]
// CHECK:       4: foo
// CHECK:       5: [B4.4]
// CHECK:       6: [B4.5]([B4.3])
// CHECK:       T: [B7.5] ? ... : ...
// CHECK:     Predecessors (2): B5 B6
// CHECK:     Successors (2): B2 B3
// CHECK:  [ B5 ]
// CHECK:       1: A()
// CHECK:       2: [B5.1] (BindTemporary)
// CHECK:       3: [B5.2]
// CHECK:       4: [B5.3]
// CHECK:       5: [B5.4] (BindTemporary)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B6 ]
// CHECK:       1: B()
// CHECK:       2: [B6.1] (BindTemporary)
// CHECK:       3: [B6.2].operator A
// CHECK:       4: [B6.3]()
// CHECK:       5: [B6.4] (BindTemporary)
// CHECK:       6: [B6.5]
// CHECK:       7: [B6.6]
// CHECK:       8: [B6.7]
// CHECK:       9: [B6.8] (BindTemporary)
// CHECK:      10: A([B6.9])
// CHECK:      11: [B6.10]
// CHECK:      12: [B6.11]
// CHECK:      13: [B6.12] (BindTemporary)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B7 ]
// CHECK:       1: ~B() (Temporary object destructor)
// CHECK:       2: B()
// CHECK:       3: [B7.2] (BindTemporary)
// CHECK:       4: [B7.3].operator _Bool
// CHECK:       5: [B7.4]()
// CHECK:       T: [B7.5] ? ... : ...
// CHECK:     Predecessors (2): B8 B9
// CHECK:     Successors (2): B5 B6
// CHECK:  [ B8 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B10
// CHECK:     Successors (1): B7
// CHECK:  [ B9 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:       3: ~B() (Temporary object destructor)
// CHECK:     Predecessors (1): B10
// CHECK:     Successors (1): B7
// CHECK:  [ B10 ]
// CHECK:       1: [B13.4] ? [B11.5] : [B12.13]
// CHECK:       2: [B10.1]
// CHECK:       3: [B10.2]
// CHECK:       4: const A &a = B().operator _Bool() ? A() : A(B().operator A());
// CHECK:       T: [B13.4] ? ... : ...
// CHECK:     Predecessors (2): B11 B12
// CHECK:     Successors (2): B8 B9
// CHECK:  [ B11 ]
// CHECK:       1: A()
// CHECK:       2: [B11.1] (BindTemporary)
// CHECK:       3: [B11.2]
// CHECK:       4: [B11.3]
// CHECK:       5: [B11.4] (BindTemporary)
// CHECK:     Predecessors (1): B13
// CHECK:     Successors (1): B10
// CHECK:  [ B12 ]
// CHECK:       1: B()
// CHECK:       2: [B12.1] (BindTemporary)
// CHECK:       3: [B12.2].operator A
// CHECK:       4: [B12.3]()
// CHECK:       5: [B12.4] (BindTemporary)
// CHECK:       6: [B12.5]
// CHECK:       7: [B12.6]
// CHECK:       8: [B12.7]
// CHECK:       9: [B12.8] (BindTemporary)
// CHECK:      10: A([B12.9])
// CHECK:      11: [B12.10]
// CHECK:      12: [B12.11]
// CHECK:      13: [B12.12] (BindTemporary)
// CHECK:     Predecessors (1): B13
// CHECK:     Successors (1): B10
// CHECK:  [ B13 ]
// CHECK:       1: B()
// CHECK:       2: [B13.1] (BindTemporary)
// CHECK:       3: [B13.2].operator _Bool
// CHECK:       4: [B13.3]()
// CHECK:       T: [B13.4] ? ... : ...
// CHECK:     Predecessors (1): B14
// CHECK:     Successors (2): B11 B12
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B8 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B7
// CHECK:  [ B1 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: int b;
// CHECK:       3: [B4.4].~A() (Implicit destructor)
// CHECK:     Predecessors (2): B2 B3
// CHECK:     Successors (1): B0
// CHECK:  [ B2 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B3 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B4 ]
// CHECK:       1: [B7.2] ?: [B6.5]
// CHECK:       2: [B4.1]
// CHECK:       3: [B4.2]
// CHECK:       4: A a = A() ?: A();
// CHECK:       T: [B7.5] ? ... : ...
// CHECK:     Predecessors (2): B5 B6
// CHECK:     Successors (2): B2 B3
// CHECK:  [ B5 ]
// CHECK:       1: [B7.3]
// CHECK:       2: [B7.3]
// CHECK:       3: [B5.2]
// CHECK:       4: [B5.3]
// CHECK:       5: [B5.4] (BindTemporary)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B6 ]
// CHECK:       1: A()
// CHECK:       2: [B6.1] (BindTemporary)
// CHECK:       3: [B6.2]
// CHECK:       4: [B6.3]
// CHECK:       5: [B6.4] (BindTemporary)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B7 ]
// CHECK:       1: A()
// CHECK:       2: [B7.1] (BindTemporary)
// CHECK:       3: 
// CHECK:       4: [B7.3].operator _Bool
// CHECK:       5: [B7.4]()
// CHECK:       T: [B7.5] ? ... : ...
// CHECK:     Predecessors (1): B8
// CHECK:     Successors (2): B5 B6
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B13 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B12
// CHECK:  [ B1 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: int b;
// CHECK:       3: [B9.4].~A() (Implicit destructor)
// CHECK:     Predecessors (2): B2 B3
// CHECK:     Successors (1): B0
// CHECK:  [ B2 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B3 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B4
// CHECK:     Successors (1): B1
// CHECK:  [ B4 ]
// CHECK:       1: [B7.3] ?: [B6.5]
// CHECK:       2: [B4.1]
// CHECK:       3: [B4.2]
// CHECK:       4: foo
// CHECK:       5: [B4.4]
// CHECK:       6: [B4.5]([B4.3])
// CHECK:       T: [B7.6] ? ... : ...
// CHECK:     Predecessors (2): B5 B6
// CHECK:     Successors (2): B2 B3
// CHECK:  [ B5 ]
// CHECK:       1: [B7.4]
// CHECK:       2: [B7.4]
// CHECK:       3: [B5.2]
// CHECK:       4: [B5.3]
// CHECK:       5: [B5.4] (BindTemporary)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B6 ]
// CHECK:       1: A()
// CHECK:       2: [B6.1] (BindTemporary)
// CHECK:       3: [B6.2]
// CHECK:       4: [B6.3]
// CHECK:       5: [B6.4] (BindTemporary)
// CHECK:     Predecessors (1): B7
// CHECK:     Successors (1): B4
// CHECK:  [ B7 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:       2: A()
// CHECK:       3: [B7.2] (BindTemporary)
// CHECK:       4: 
// CHECK:       5: [B7.4].operator _Bool
// CHECK:       6: [B7.5]()
// CHECK:       T: [B7.6] ? ... : ...
// CHECK:     Predecessors (2): B9 B8
// CHECK:     Successors (2): B5 B6
// CHECK:  [ B8 ]
// CHECK:       1: ~A() (Temporary object destructor)
// CHECK:     Predecessors (1): B9
// CHECK:     Successors (1): B7
// CHECK:  [ B9 ]
// CHECK:       1: [B12.2] ?: [B11.5]
// CHECK:       2: [B9.1]
// CHECK:       3: [B9.2]
// CHECK:       4: const A &a = A() ?: A();
// CHECK:       T: [B12.5] ? ... : ...
// CHECK:     Predecessors (2): B10 B11
// CHECK:     Successors (2): B7 B8
// CHECK:  [ B10 ]
// CHECK:       1: [B12.3]
// CHECK:       2: [B12.3]
// CHECK:       3: [B10.2]
// CHECK:       4: [B10.3]
// CHECK:       5: [B10.4] (BindTemporary)
// CHECK:     Predecessors (1): B12
// CHECK:     Successors (1): B9
// CHECK:  [ B11 ]
// CHECK:       1: A()
// CHECK:       2: [B11.1] (BindTemporary)
// CHECK:       3: [B11.2]
// CHECK:       4: [B11.3]
// CHECK:       5: [B11.4] (BindTemporary)
// CHECK:     Predecessors (1): B12
// CHECK:     Successors (1): B9
// CHECK:  [ B12 ]
// CHECK:       1: A()
// CHECK:       2: [B12.1] (BindTemporary)
// CHECK:       3: 
// CHECK:       4: [B12.3].operator _Bool
// CHECK:       5: [B12.4]()
// CHECK:       T: [B12.5] ? ... : ...
// CHECK:     Predecessors (1): B13
// CHECK:     Successors (2): B10 B11
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: A()
// CHECK:       2: [B1.1] (BindTemporary)
// CHECK:       3: [B1.2]
// CHECK:       4: [B1.3]
// CHECK:       5: A a = A();
// CHECK:       6: ~A() (Temporary object destructor)
// CHECK:       7: int b;
// CHECK:       8: [B1.5].~A() (Implicit destructor)
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: A()
// CHECK:       2: [B1.1] (BindTemporary)
// CHECK:       3: [B1.2]
// CHECK:       4: [B1.3]
// CHECK:       5: const A &a = A();
// CHECK:       6: A()
// CHECK:       7: [B1.6] (BindTemporary)
// CHECK:       8: [B1.7]
// CHECK:       9: [B1.8]
// CHECK:      10: foo
// CHECK:      11: [B1.10]
// CHECK:      12: [B1.11]([B1.9])
// CHECK:      13: ~A() (Temporary object destructor)
// CHECK:      14: int b;
// CHECK:      15: [B1.5].~A() (Implicit destructor)
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: A::make
// CHECK:       2: [B1.1]
// CHECK:       3: [B1.2]()
// CHECK:       4: [B1.3] (BindTemporary)
// CHECK:       5: [B1.4]
// CHECK:       6: [B1.5]
// CHECK:       7: A a = A::make();
// CHECK:       8: ~A() (Temporary object destructor)
// CHECK:       9: int b;
// CHECK:      10: [B1.7].~A() (Implicit destructor)
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: A::make
// CHECK:       2: [B1.1]
// CHECK:       3: [B1.2]()
// CHECK:       4: [B1.3] (BindTemporary)
// CHECK:       5: [B1.4]
// CHECK:       6: [B1.5]
// CHECK:       7: const A &a = A::make();
// CHECK:       8: A::make
// CHECK:       9: [B1.8]
// CHECK:      10: [B1.9]()
// CHECK:      11: [B1.10] (BindTemporary)
// CHECK:      12: [B1.11]
// CHECK:      13: [B1.12]
// CHECK:      14: foo
// CHECK:      15: [B1.14]
// CHECK:      16: [B1.15]([B1.13])
// CHECK:      17: ~A() (Temporary object destructor)
// CHECK:      18: int b;
// CHECK:      19: [B1.7].~A() (Implicit destructor)
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: int a;
// CHECK:       2: A()
// CHECK:       3: [B1.2] (BindTemporary)
// CHECK:       4: [B1.3].operator int
// CHECK:       5: [B1.4]()
// CHECK:       6: a
// CHECK:       7: [B1.6] = [B1.5]
// CHECK:       8: ~A() (Temporary object destructor)
// CHECK:       9: int b;
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):
// CHECK:  [ B2 (ENTRY) ]
// CHECK:     Predecessors (0):
// CHECK:     Successors (1): B1
// CHECK:  [ B1 ]
// CHECK:       1: A()
// CHECK:       2: [B1.1] (BindTemporary)
// CHECK:       3: [B1.2].operator int
// CHECK:       4: [B1.3]()
// CHECK:       5: [B1.4]
// CHECK:       6: int([B1.5])
// CHECK:       7: B()
// CHECK:       8: [B1.7] (BindTemporary)
// CHECK:       9: [B1.8].operator int
// CHECK:      10: [B1.9]()
// CHECK:      11: [B1.10]
// CHECK:      12: int([B1.11])
// CHECK:      13: [B1.6] + [B1.12]
// CHECK:      14: a([B1.13]) (Member initializer)
// CHECK:      15: ~B() (Temporary object destructor)
// CHECK:      16: ~A() (Temporary object destructor)
// CHECK:      17: /*implicit*/int()
// CHECK:      18: b([B1.17]) (Member initializer)
// CHECK:     Predecessors (1): B2
// CHECK:     Successors (1): B0
// CHECK:  [ B0 (EXIT) ]
// CHECK:     Predecessors (1): B1
// CHECK:     Successors (0):

