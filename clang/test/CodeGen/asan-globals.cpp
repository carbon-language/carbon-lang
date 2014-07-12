// RUN: echo "global:*blacklisted_global*" > %t.blacklist
// RUN: %clang_cc1 -fsanitize=address -fsanitize-blacklist=%t.blacklist -emit-llvm -o - %s | FileCheck %s
// RUN: echo "src:%s" > %t.blacklist-src
// RUN: %clang_cc1 -fsanitize=address -fsanitize-blacklist=%t.blacklist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=BLACKLIST-SRC
// REQUIRES: shell

int global;
// CHECK: [[GLOBAL_LOC:@.asan_loc_descr[0-9]*]] = private unnamed_addr constant {{.*}} i32 [[@LINE-1]], i32 5
// CHECK: [[GLOBAL_NAME:@.str[0-9]+]] = private unnamed_addr constant {{.*}} c"global\00"
int dyn_init_global = global;
// CHECK: [[DYN_INIT_LOC:@.asan_loc_descr[0-9]*]] = {{.*}} i32 [[@LINE-1]], i32 5
// CHECK: [[DYN_INIT_NAME:@.str[0-9]+]] = private unnamed_addr constant {{.*}} c"dyn_init_global\00"
int blacklisted_global;

void func() {
  static int static_var = 0;
  // CHECK: [[STATIC_LOC:@.asan_loc_descr[0-9]*]] = {{.*}} i32 [[@LINE-1]], i32 14
  // CHECK: [[STATIC_NAME:@.str[0-9]+]] = private unnamed_addr constant {{.*}} c"static_var\00"
  const char *literal = "Hello, world!";
  // CHECK: [[LITERAL_LOC:@.asan_loc_descr[0-9]*]] = {{.*}} i32 [[@LINE-1]], i32 25
  // CHECK: [[LITERAL_NAME:@.str[0-9]+]] = private unnamed_addr constant {{.*}} c"<string literal>\00"
}

// CHECK: !llvm.asan.globals = !{![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[BLACKLISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// CHECK: ![[GLOBAL]] = metadata !{{{.*}} [[GLOBAL_LOC]], {{.*}} [[GLOBAL_NAME]], i1 false, i1 false}
// CHECK: ![[DYN_INIT_GLOBAL]] = metadata !{{{.*}} [[DYN_INIT_LOC]], {{.*}} [[DYN_INIT_NAME]], i1 true, i1 false}
// CHECK: ![[BLACKLISTED_GLOBAL]] = metadata !{{{.*}}, null, null, i1 false, i1 true}
// CHECK: ![[STATIC_VAR]] = metadata !{{{.*}} [[STATIC_LOC]], {{.*}} [[STATIC_NAME]], i1 false, i1 false}
// CHECK: ![[LITERAL]] = metadata !{{{.*}} [[LITERAL_LOC]], {{.*}} [[LITERAL_NAME]], i1 false, i1 false}

// BLACKLIST-SRC: !llvm.asan.globals = !{![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[BLACKLISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// BLACKLIST-SRC: ![[GLOBAL]] = metadata !{{{.*}} null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[DYN_INIT_GLOBAL]] = metadata !{{{.*}} null, null, i1 true, i1 true}
// BLACKLIST-SRC: ![[BLACKLISTED_GLOBAL]] = metadata !{{{.*}}, null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[STATIC_VAR]] = metadata !{{{.*}} null, null, i1 false, i1 true}
// BLACKLIST-SRC: ![[LITERAL]] = metadata !{{{.*}} null, null, i1 false, i1 true}
