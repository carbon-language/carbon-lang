// RUN: %python %S/check_clang_tidy.py %s cppcoreguidelines-pro-type-const-cast %t

const int *i;
int *j;
void f() { j = const_cast<int *>(i); }
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use const_cast [cppcoreguidelines-pro-type-const-cast]
