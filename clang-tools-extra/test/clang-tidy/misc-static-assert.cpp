// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-static-assert %t
// REQUIRES: shell

void abort() {}
#ifdef NDEBUG
#define assert(x) 1
#else
#define assert(x)                                                              \
  if (!(x))                                                                    \
  abort()
#endif

#define ZERO_MACRO 0

#define my_macro() assert(0 == 1)
// CHECK-FIXES: #define my_macro() assert(0 == 1)

constexpr bool myfunc(int a, int b) { return a * b == 0; }

class A {
public:
  bool method() { return true; }
};

class B {
public:
  constexpr bool method() { return true; }
};

template <class T> void doSomething(T t) {
  assert(myfunc(1, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be replaced by static_assert() [misc-static-assert]
  // CHECK-FIXES: {{^  }}static_assert(myfunc(1, 2), "");

  assert(t.method());
  // CHECK-FIXES: {{^  }}assert(t.method());
}

int main() {
  my_macro();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}my_macro();

  assert(myfunc(1, 2) && (3 == 4));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(myfunc(1, 2) && (3 == 4), "");

  int x = 1;
  assert(x == 0);
  // CHECK-FIXES: {{^  }}assert(x == 0);

  A a;
  B b;

  doSomething<A>(a);
  doSomething<B>(b);

  assert(false);
  // CHECK-FIXES: {{^  }}assert(false);

  assert(ZERO_MACRO);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(ZERO_MACRO, "");

  assert(0 && "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(0 && "Don't report me!");

  assert(false && "Don't report me!");
  // CHECK-FIXES: {{^  }}assert(false && "Don't report me!");

  assert(ZERO_MACRO && "Report me!");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(ZERO_MACRO , "Report me!");

  assert(10==5 && "Report me!");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: found assert() that could be
  // CHECK-FIXES: {{^  }}static_assert(10==5 , "Report me!");

  return 0;
}
