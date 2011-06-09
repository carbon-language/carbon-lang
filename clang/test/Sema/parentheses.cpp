// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fixit %s -o - | %clang_cc1 -Wparentheses -Werror -

bool someConditionFunc();

void conditional_op(int x, int y, bool b) {
  (void)(x + someConditionFunc() ? 1 : 2); // expected-warning {{?: has lower precedence than +}} \
                                           // expected-note {{place parentheses around the ?: expression to evaluate it first}} \
                                           // expected-note {{place parentheses around the + expression to silence this warning}}

  (void)(x - b ? 1 : 2); // expected-warning {{?: has lower precedence than -}} \
                         // expected-note {{place parentheses around the ?: expression to evaluate it first}} \
                         // expected-note {{place parentheses around the - expression to silence this warning}}

  (void)(x * (x == y) ? 1 : 2); // expected-warning {{?: has lower precedence than *}} \
                                // expected-note {{place parentheses around the ?: expression to evaluate it first}} \
                                // expected-note {{place parentheses around the * expression to silence this warning}}
}

class Stream {
public:
  operator int();
  Stream &operator<<(int);
  Stream &operator<<(const char*);
};

void f(Stream& s, bool b) {
  (void)(s << b ? "foo" : "bar"); // expected-warning {{?: has lower precedence than <<}} \
                                  // expected-note {{place parentheses around the ?: expression to evaluate it first}} \
                                  // expected-note {{place parentheses around the << expression to silence this warning}}
}
