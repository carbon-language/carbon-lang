// RUN: %clang --analyze -std=c++0x %s -Xclang -verify -o /dev/null

void test_static_assert() {
  static_assert(sizeof(void *) == sizeof(void*), "test_static_assert");
}

void test_analyzer_working() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}

// Test that pointer-to-member functions don't cause the analyzer
// to crash.
struct RDar10243398 {
  void bar(int x);
};

typedef void (RDar10243398::*RDar10243398MemberFn)(int x);

void test_rdar10243398(RDar10243398 *p) {
  RDar10243398MemberFn q = &RDar10243398::bar;
  ((*p).*(q))(1);
}
