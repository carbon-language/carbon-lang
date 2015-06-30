// RUN: %clang_cc1 -triple i686-windows-gnu -fms-compatibility -g -emit-llvm %s -o - \
// RUN:    | FileCheck %s

struct __declspec(dllexport) s {
  static const unsigned int ui = 0;
};

// CHECK: ![[SCOPE:[0-9]+]] = distinct !DICompileUnit(
// CHECK: !DIGlobalVariable(name: "ui", linkageName: "_ZN1s2uiE", scope: ![[SCOPE]],
// CHECK-SAME:              variable: i32* @_ZN1s2uiE

