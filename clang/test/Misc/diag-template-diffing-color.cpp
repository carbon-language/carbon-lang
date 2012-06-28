// RUN: %clang_cc1 -fsyntax-only -fcolor-diagnostics %s 2>&1 | FileCheck %s
// XFAIL: cygwin,mingw32,win32
template<typename> struct foo {};
void func(foo<int>);
int main() {
  func(foo<double>());
}
// CHECK: {{.*}}candidate function not viable: no known conversion from 'foo<{{.}}[0;1;36mdouble{{.}}[0m>' to 'foo<{{.}}[0;1;36mint{{.}}[0m>' for 1st argument{{.}}[0m
