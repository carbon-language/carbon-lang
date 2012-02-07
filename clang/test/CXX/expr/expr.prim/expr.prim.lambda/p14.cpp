// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify

class NonCopyable {
  NonCopyable(const NonCopyable&); // expected-note 2 {{implicitly declared private here}}
};

void capture_by_copy(NonCopyable nc, NonCopyable &ncr) {
  // FIXME: error messages should talk about capture
  [nc] { }; // expected-error{{field of type 'NonCopyable' has private copy constructor}} \
             // expected-error{{lambda expressions are not supported yet}}
  [ncr] { }; // expected-error{{field of type 'NonCopyable' has private copy constructor}} \
             // expected-error{{lambda expressions are not supported yet}}
}

// FIXME: arrays!
