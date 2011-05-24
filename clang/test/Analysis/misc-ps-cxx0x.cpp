// RUN: %clang --analyze -std=c++0x %s -Xclang -verify

void test_static_assert() {
  static_assert(sizeof(void *) == sizeof(void*), "test_static_assert");
}

void test_analyzer_working() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}

