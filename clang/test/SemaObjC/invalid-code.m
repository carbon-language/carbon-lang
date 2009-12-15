// RUN: %clang_cc1 %s -fsyntax-only -verify

// rdar://6124613
void test1() {
  void *p = @1; // expected-error {{unexpected '@' in program}}
}

