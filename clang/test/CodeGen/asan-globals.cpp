// RUN: echo "global:*blacklisted_global*" > %t.blacklist
// RUN: %clang_cc1 -fsanitize=address -fsanitize-blacklist=%t.blacklist -emit-llvm -o - %s | FileCheck %s
// RUN: echo "src:%s" > %t.blacklist-src
// RUN: %clang_cc1 -fsanitize=address -fsanitize-blacklist=%t.blacklist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=BLACKLIST-SRC
// REQUIRES: shell

int global;
int dyn_init_global = global;
int blacklisted_global;

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
}

// CHECK: !llvm.asan.globals = !{![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[BLACKLISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// CHECK: ![[GLOBAL]] = metadata !{{{.*}} metadata ![[GLOBAL_LOC:[0-9]+]], metadata !"global", i1 false, i1 false}
// CHECK: ![[GLOBAL_LOC]] = metadata !{metadata !"{{.*}}asan-globals.cpp", i32 7, i32 5}
// CHECK: ![[DYN_INIT_GLOBAL]] = metadata !{{{.*}} metadata ![[DYN_INIT_LOC:[0-9]+]], metadata !"dyn_init_global", i1 true, i1 false}
// CHECK: ![[DYN_INIT_LOC]] = metadata !{metadata !"{{.*}}asan-globals.cpp", i32 8, i32 5}
// CHECK: ![[BLACKLISTED_GLOBAL]] = metadata !{{{.*}}, null, null, i1 false, i1 true}
// CHECK: ![[STATIC_VAR]] = metadata !{{{.*}} metadata ![[STATIC_LOC:[0-9]+]], metadata !"static_var", i1 false, i1 false}
// CHECK: ![[STATIC_LOC]] = metadata !{metadata !"{{.*}}asan-globals.cpp", i32 12, i32 14}
// CHECK: ![[LITERAL]] = metadata !{{{.*}} metadata ![[LITERAL_LOC:[0-9]+]], metadata !"<string literal>", i1 false, i1 false}
// CHECK: ![[LITERAL_LOC]] = metadata !{metadata !"{{.*}}asan-globals.cpp", i32 13, i32 25}

// BLACKLIST-SRC: !llvm.asan.globals = !{![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[BLACKLISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// BLACKLIST-SRC: ![[GLOBAL]] = metadata !{{{.*}} null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[DYN_INIT_GLOBAL]] = metadata !{{{.*}} null, null, i1 true, i1 true}
// BLACKLIST-SRC: ![[BLACKLISTED_GLOBAL]] = metadata !{{{.*}}, null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[STATIC_VAR]] = metadata !{{{.*}} null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[LITERAL]] = metadata !{{{.*}} null, null, i1 false, i1 true}
