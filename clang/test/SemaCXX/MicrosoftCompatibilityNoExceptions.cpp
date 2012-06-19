// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-compatibility

// PR13153
namespace std {}
class type_info {};
void f() {
  (void)typeid(int);
}
