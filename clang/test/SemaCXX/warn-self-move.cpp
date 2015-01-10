// RUN: %clang_cc1 -fsyntax-only -Wself-move -std=c++11 -verify %s

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
}
}

void int_test() {
  int x = 5;
  x = std::move(x);  // expected-warning{{explicitly moving}}
  (x) = std::move(x);  // expected-warning{{explicitly moving}}

  using std::move;
  x = move(x);  // expected-warning{{explicitly moving}}
}

int global;
void global_int_test() {
  global = std::move(global);  // expected-warning{{explicitly moving}}
  (global) = std::move(global);  // expected-warning{{explicitly moving}}

  using std::move;
  global = move(global);  // expected-warning{{explicitly moving}}
}

class field_test {
  int x;
  field_test(field_test&& other) {
    x = std::move(x);  // expected-warning{{explicitly moving}}
    x = std::move(other.x);
    other.x = std::move(x);
    other.x = std::move(other.x);  // expected-warning{{explicitly moving}}
  }
};
