// RUN: %clang_cc1 -std=c++11 -verify %s

using size_t = decltype(sizeof(int));
void operator "" wibble(const char *); // expected-warning {{preempted}}
void operator "" wibble(const char *, size_t); // expected-warning {{preempted}}

template<typename T>
void f() {
  // A program containing a reserved ud-suffix is ill-formed.
  // FIXME: Reject these for the right reason.
  123wibble; // expected-error {{suffix 'wibble'}}
  123.0wibble; // expected-error {{suffix 'wibble'}}
  ""wibble; // expected-warning {{unused}}
  R"x("hello")x"wibble; // expected-warning {{unused}}
}
