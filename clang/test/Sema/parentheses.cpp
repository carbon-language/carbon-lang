// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

bool someConditionFunc();

void conditional_op(int x, int y, bool b) {
  (void)(x + someConditionFunc() ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                           // expected-note {{place parentheses around the '+' expression to silence this warning}} \
                                           // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:33-[[@LINE-4]]:33}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:41-[[@LINE-6]]:41}:")"

  (void)(x - b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '-'}} \
                         // expected-note {{place parentheses around the '-' expression to silence this warning}} \
                         // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:15-[[@LINE-4]]:15}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:23-[[@LINE-6]]:23}:")"

  (void)(x * (x == y) ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '*'}} \
                                // expected-note {{place parentheses around the '*' expression to silence this warning}} \
                                // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:22}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:30-[[@LINE-6]]:30}:")"
}

class Stream {
public:
  operator int();
  Stream &operator<<(int);
  Stream &operator<<(const char*);
  Stream &operator>>(int);
  Stream &operator>>(const char*);
};

void f(Stream& s, bool b) {
  (void)(s << b ? "foo" : "bar"); // expected-warning {{operator '?:' has lower precedence than '<<'}} \
                                  // expected-note {{place parentheses around the '<<' expression to silence this warning}} \
                                  // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:32-[[@LINE-6]]:32}:")"

  (void)(s << 5 == 1); // expected-warning {{overloaded operator << has lower precedence than comparison operator}} \
                       // expected-note {{place parentheses around comparison expression to evaluate it first}} \
                       // expected-note {{place parentheses around the '<<' expression to silence this warning}}

  (void)(s >> 5 == 1); // expected-warning {{overloaded operator >> has lower precedence than comparison operator}} \
                       // expected-note {{place parentheses around comparison expression to evaluate it first}} \
                       // expected-note {{place parentheses around the '>>' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:21-[[@LINE-4]]:21}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:")"
}

struct S {
  operator int() { return 42; }
  friend S operator+(const S &lhs, bool) { return S(); }
};

void test(S *s, bool (S::*m_ptr)()) {
  (void)(*s + true ? "foo" : "bar"); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                     // expected-note {{place parentheses around the '+' expression to silence this warning}} \
                                     // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:19-[[@LINE-4]]:19}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:35-[[@LINE-6]]:35}:")"

  (void)((*s + true) ? "foo" : "bar"); // No warning.

  // Don't crash on unusual member call expressions.
  (void)((s->*m_ptr)() ? "foo" : "bar");
}

void test(int a, int b, int c) {
  (void)(a >> b + c); // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} \
                         expected-note {{place parentheses around the '+' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:")"

  (void)(a - b << c); // expected-warning {{operator '<<' has lower precedence than '-'; '-' will be evaluated first}} \
                         expected-note {{place parentheses around the '-' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:15}:")"

  Stream() << b + c;
  Stream() >> b + c; // expected-warning {{operator '>>' has lower precedence than '+'; '+' will be evaluated first}} \
                        expected-note {{place parentheses around the '+' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:")"
}

namespace PR15628 {
  struct BlockInputIter {
    void* operator++(int);
    void* operator--(int);
  };

  void test(BlockInputIter i) {
    (void)(i++ ? true : false); // no-warning
    (void)(i-- ? true : false); // no-warning
  }
}
