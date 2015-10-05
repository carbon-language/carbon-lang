// RUN: %python %S/check_clang_tidy.py %s cert-dcl50-cpp %t

// Variadic function definitions are diagnosed.
void f1(int, ...) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: do not define a C-style variadic function; consider using a function parameter pack or currying instead [cert-dcl50-cpp]

// Variadic function *declarations* are not diagnosed.
void f2(int, ...); // ok

// Function parameter packs are good, however.
template <typename Arg, typename... Ts>
void f3(Arg F, Ts... Rest) {}

struct S {
  void f(int, ...); // ok
  void f1(int, ...) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: do not define a C-style variadic function; consider using a function parameter pack or currying instead
};
