// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic.h > %T/basic.h
// RUN: cpp11-migrate -use-nullptr %t.cpp -- -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: FileCheck -input-file=%T/basic.h %S/Inputs/basic.h

#include "Inputs/basic.h"

const unsigned int g_null = 0;
#define NULL 0

void test_assignment() {
  int *p1 = 0;
  // CHECK: int *p1 = nullptr;
  p1 = 0;
  // CHECK: p1 = nullptr;

  int *p2 = NULL;
  // CHECK: int *p2 = nullptr;

  p2 = p1;
  // CHECK: p2 = p1;

  const int null = 0;
  int *p3 = null;
  // CHECK: int *p3 = nullptr;

  p3 = NULL;
  // CHECK: p3 = nullptr;

  int *p4 = p3;
  // CHECK: int *p4 = p3;

  p4 = null;
  // CHECK: p4 = nullptr;

  int i1 = 0;
  // CHECK: int i1 = 0;

  int i2 = NULL;
  // CHECK: int i2 = NULL;

  int i3 = null;
  // CHECK: int i3 = null;

  int *p5, *p6, *p7;
  p5 = p6 = p7 = NULL;
  // CHECK: p5 = p6 = p7 = nullptr;
}

struct Foo {
  Foo(int *p = NULL) : m_p1(p) {}
  // CHECK: Foo(int *p = nullptr) : m_p1(p) {}

  void bar(int *p = 0) {}
  // CHECK: void bar(int *p = nullptr) {}

  void baz(int i = 0) {}
  // CHECK: void baz(int i = 0) {}

  int *m_p1;
  static int *m_p2;
};

int *Foo::m_p2 = NULL;
// CHECK: int *Foo::m_p2 = nullptr;

template <typename T>
struct Bar {
  Bar(T *p) : m_p(p) {
    m_p = static_cast<T*>(NULL);
    // CHECK: m_p = nullptr;

    m_p = static_cast<T*>(reinterpret_cast<int*>((void*)NULL));
    // CHECK: m_p = nullptr;

    m_p = static_cast<T*>(p ? p : static_cast<void*>(g_null));
    // CHECK: m_p = static_cast<T*>(p ? p : nullptr);

    T *p2 = static_cast<T*>(reinterpret_cast<int*>((void*)NULL));
    // CHECK: T *p2 = nullptr;

    m_p = NULL;
    // CHECK: m_p = nullptr;

    int i = static_cast<int>(0.f);
    // CHECK: int i = static_cast<int>(0.f);
    T *i2 = static_cast<int>(0.f);
    // CHECK: T *i2 = nullptr;
  }

  T *m_p;
};

struct Baz {
  Baz() : i(0) {}
  int i;
};

void test_cxx_cases() {
  Foo f(g_null);
  // CHECK: Foo f(nullptr);

  f.bar(NULL);
  // CHECK: f.bar(nullptr);

  f.baz(g_null);
  // CHECK: f.baz(g_null);

  f.m_p1 = 0;
  // CHECK: f.m_p1 = nullptr;

  Bar<int> b(g_null);
  // CHECK: Bar<int> b(nullptr);

  Baz b2;
  int Baz::*memptr(0);
  // CHECK: int Baz::*memptr(nullptr);

  memptr = 0;
  // CHECK: memptr = nullptr;
}

void test_function_default_param1(void *p = 0);
// CHECK: void test_function_default_param1(void *p = nullptr);

void test_function_default_param2(void *p = NULL);
// CHECK: void test_function_default_param2(void *p = nullptr);

void test_function_default_param3(void *p = g_null);
// CHECK: void test_function_default_param3(void *p = nullptr);

void test_function(int *p) {}
// CHECK: void test_function(int *p) {}

void test_function_no_ptr_param(int i) {}

void test_function_call() {
  test_function(0);
  // CHECK: test_function(nullptr);

  test_function(NULL);
  // CHECK: test_function(nullptr);

  test_function(g_null);
  // CHECK: test_function(nullptr);

  test_function_no_ptr_param(0);
  // CHECK: test_function_no_ptr_param(0);
}

char *test_function_return1() {
  return 0;
  // CHECK: return nullptr;
}

void *test_function_return2() {
  return NULL;
  // CHECK: return nullptr;
}

long *test_function_return3() {
  return g_null;
  // CHECK: return nullptr;
}

int test_function_return4() {
  return 0;
  // CHECK: return 0;
}

int test_function_return5() {
  return NULL;
  // CHECK: return NULL;
}

int test_function_return6() {
  return g_null;
  // CHECK: return g_null;
}

// Test function-like macros where the parameter to the macro (expression)
// results in a nullptr.
void __dummy_assert_fail() {}

void test_function_like_macro1() {
  // This tests that the CastSequenceVisitor is working properly.
#define my_assert(expr) \
  ((expr) ? static_cast<void>(expr) : __dummy_assert_fail())

  int *p;
  my_assert(p != 0);
  // CHECK: my_assert(p != nullptr);
#undef my_assert
}

void test_function_like_macro2() {
  // This tests that the implicit cast is working properly.
#define my_macro(expr) \
  (expr)

  int *p;
  my_macro(p != 0);
  // CHECK: my_macro(p != nullptr);
#undef my_macro
}

// Test parentheses expressions resulting in a nullptr.
int *test_parentheses_expression1() {
  return(0);
  // CHECK: return(nullptr);
}

int *test_parentheses_expression2() {
  return(int(0.f));
  // CHECK: return(nullptr);
}

int *test_nested_parentheses_expression() {
  return((((0))));
  // CHECK: return((((nullptr))));
}

void *test_parentheses_explicit_cast() {
  return(static_cast<void*>(0));
  // CHECK: return(nullptr);
}

void *test_parentheses_explicit_cast_sequence1() {
  return(static_cast<void*>(static_cast<int*>((void*)NULL)));
  // CHECK: return(nullptr);
}

void *test_parentheses_explicit_cast_sequence2() {
  return(static_cast<void*>(reinterpret_cast<int*>((float*)int(0.f))));
  // CHECK: return(nullptr);
}
