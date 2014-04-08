// RUN: %clang_cc1 -std=c++11 -triple i686-pc-win32 %s -fms-extensions -fno-rtti -emit-llvm -o - | FileCheck %s

// Similar to predefined-expr.cpp, but not as exhaustive, since it's basically
// equivalent to __PRETTY_FUNCTION__.

extern "C" int printf(const char *, ...);

void freeFunc(int *, char) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK: private unnamed_addr constant [{{.*}} x i8] c"void __cdecl freeFunc(int *, char)\00"

struct TopLevelClass {
  void topLevelMethod(int *, char);
};
void TopLevelClass::topLevelMethod(int *, char) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK: private unnamed_addr constant [{{.*}} x i8] c"void __thiscall TopLevelClass::topLevelMethod(int *, char)\00"

namespace NS {
struct NamespacedClass {
  void namespacedMethod(int *, char);
};
void NamespacedClass::namespacedMethod(int *, char) {
  printf("__FUNCSIG__ %s\n\n", __FUNCSIG__);
}
// CHECK: private unnamed_addr constant [{{.*}} x i8] c"void __thiscall NS::NamespacedClass::namespacedMethod(int *, char)\00"
}
