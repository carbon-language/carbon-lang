// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-nullptr %t.cpp -- -I %S
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t2.cpp
// RUN: cpp11-migrate -use-nullptr -user-null-macros=MY_NULL %t2.cpp -- -I %S
// RUN: FileCheck -check-prefix=USER-SUPPLIED-NULL -input-file=%t2.cpp %s

#define NULL 0
// CHECK: #define NULL 0

void dummy(int*) {}
void side_effect() {}

#define MACRO_EXPANSION_HAS_NULL \
  void foo() { \
    dummy(0); \
    dummy(NULL); \
    side_effect(); \
  }
  // CHECK: void foo() { \
  // CHECK-NEXT:   dummy(0); \
  // CHECK-NEXT:   dummy(NULL); \

MACRO_EXPANSION_HAS_NULL;
// CHECK: MACRO_EXPANSION_HAS_NULL;
#undef MACRO_EXPANSION_HAS_NULL


void test_macro_expansion1() {
#define MACRO_EXPANSION_HAS_NULL \
  dummy(NULL); \
  side_effect();
  // CHECK: dummy(NULL); \
  // CHECK-NEXT: side_effect();

  MACRO_EXPANSION_HAS_NULL;
  // CHECK: MACRO_EXPANSION_HAS_NULL;

#undef MACRO_EXPANSION_HAS_NULL
}

void test_macro_expansion2() {
#define MACRO_EXPANSION_HAS_NULL \
  dummy(NULL); \
  side_effect();
  // CHECK: dummy(NULL); \
  // CHECK-NEXT: side_effect();

#define OUTER_MACRO \
  MACRO_EXPANSION_HAS_NULL; \
  side_effect();

  OUTER_MACRO;
  // CHECK: OUTER_MACRO;

#undef OUTER_MACRO
#undef MACRO_EXPANSION_HAS_NULL
}

void test_macro_expansion3() {
#define MY_NULL NULL
  int *p = MY_NULL;
  // CHECK: int *p = MY_NULL;
  // USER-SUPPLIED-NULL: int *p = nullptr;
#undef MY_NULL
}

// Test function-like macros where the parameter to the macro (expression)
// results in a nullptr.
void __dummy_assert_fail() {}

void test_function_like_macro1() {
  // This tests that the CastSequenceVisitor is working properly.
#define my_assert(expr) \
  ((expr) ? static_cast<void>(expr) : __dummy_assert_fail())
  // CHECK: ((expr) ? static_cast<void>(expr) : __dummy_assert_fail())

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


