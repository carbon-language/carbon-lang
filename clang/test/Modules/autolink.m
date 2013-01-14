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

@import NoUmbrella;
int use_no_umbrella() {
  return no_umbrella_A;
}

// CHECK: !llvm.link.libraries = !{![[AUTOLINK:[0-9]+]], ![[AUTOLINK_FRAMEWORK:[0-9]+]], ![[MODULE:[0-9]+]], ![[NOUMBRELLA:[0-9]+]]}
// CHECK: ![[AUTOLINK]] = metadata !{metadata !"autolink", i1 false}
// CHECK: ![[AUTOLINK_FRAMEWORK]] = metadata !{metadata !"autolink_framework", i1 true}
// CHECK: ![[MODULE]] = metadata !{metadata !"Module", i1 true}
// CHECK: ![[NOUMBRELLA]] = metadata !{metadata !"NoUmbrella", i1 true}
