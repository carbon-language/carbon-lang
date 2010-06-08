// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin10 | FileCheck %s

struct S {
  virtual ~S() { }
};

// PR5706
// Make sure this doesn't crash; the mangling doesn't matter because the name
// doesn't have linkage.
static struct : S { } obj8;

void f() {
  // Make sure this doesn't crash; the mangling doesn't matter because the
  // generated vtable/etc. aren't modifiable (although it would be nice for
  // codesize to make it consistent inside inline functions).
  static struct : S { } obj8;
}

inline int f2() {
  // FIXME: We don't mangle the names of a or x correctly!
  static struct { int a() { static int x; return ++x; } } obj;
  return obj.a();
}

int f3() { return f2(); }

struct A {
  typedef struct { int x; } *ptr;
  ptr m;
  int a() {
    static struct x {
      // FIXME: We don't mangle the names of a or x correctly!
      int a(ptr A::*memp) { static int x; return ++x; }
    } a;
    return a.a(&A::m);
  }
};

int f4() { return A().a(); }

int f5() {
  static union {
    int a;
  };
  
  // CHECK: _ZZ2f5vE1a
  return a;
}

int f6() {
  static union {
    union {
      int : 1;
    };
    int b;
  };
  
  // CHECK: _ZZ2f6vE1b
  return b;
}

int f7() {
  static union {
    union {
      int b;
    } a;
  };
  
  // CHECK: _ZZ2f7vE1a
  return a.b;
}
