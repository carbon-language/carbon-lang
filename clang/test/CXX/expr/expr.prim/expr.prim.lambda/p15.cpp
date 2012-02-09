// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

class NonCopyable {
  NonCopyable(const NonCopyable&);
};

void capture_by_ref(NonCopyable nc, NonCopyable &ncr) {
  int array[3];
  (void)[&nc] () -> void {}; // expected-error{{lambda expressions are not supported yet}}
  (void)[&ncr] () -> void {}; // expected-error{{lambda expressions are not supported yet}}
  (void)[&array] () -> void {}; // expected-error{{lambda expressions are not supported yet}}
}
