// RUN: rm -rf %t
// RUN: %clang_cc1 -emit-llvm -o - -fmodule-cache-path %t -fmodules -F %S/Inputs -I %S/Inputs %s | FileCheck %s

@import autolink.sub2;

int f() {
  return autolink_sub2();
}

@import autolink;

int g() {
  return autolink;
}

@import Module.SubFramework;
const char *get_module_subframework() {
  return module_subframework;
}

@import DependsOnModule.SubFramework;
float *get_module_subframework_dep() {
  return sub_framework;
}

@import NoUmbrella;
int use_no_umbrella() {
  return no_umbrella_A;
}

// CHECK: !llvm.module.linkoptions = !{![[AUTOLINK_FRAMEWORK:[0-9]+]], ![[AUTOLINK:[0-9]+]], ![[DEPENDSONMODULE:[0-9]+]], ![[MODULE:[0-9]+]], ![[NOUMBRELLA:[0-9]+]]}
// CHECK: ![[AUTOLINK_FRAMEWORK]] = metadata !{metadata !"-framework", metadata !"autolink_framework"}
// CHECK: ![[AUTOLINK]] = metadata !{metadata !"-lautolink"}
// CHECK: ![[DEPENDSONMODULE]] = metadata !{metadata !"-framework", metadata !"DependsOnModule"}
// CHECK: ![[MODULE]] = metadata !{metadata !"-framework", metadata !"Module"}
// CHECK: ![[NOUMBRELLA]] = metadata !{metadata !"-framework", metadata !"NoUmbrella"}
