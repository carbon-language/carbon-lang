// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple x86_64-linux-gnu \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple powerpc64le-linux-gnu \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple arm-linux-gnueabi \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple thumbv7-linux-gnueabi \
// RUN:   -mstack-protector-guard-offset=1024 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -mstack-protector-guard=sysreg -triple aarch64-linux-gnu \
// RUN:   -mstack-protector-guard-offset=1024 -mstack-protector-guard-reg=sp_el0 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=AARCH64
void foo(int*);
void bar(int x) {
  int baz[x];
  foo(baz);
}

// CHECK: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]]}
// CHECK: [[ATTR1]] = !{i32 1, !"stack-protector-guard", !"sysreg"}
// CHECK: [[ATTR2]] = !{i32 1, !"stack-protector-guard-offset", i32 1024}

// AARCH64: !llvm.module.flags = !{{{.*}}[[ATTR1:![0-9]+]], [[ATTR2:![0-9]+]], [[ATTR3:![0-9]+]]}
// AARCH64: [[ATTR1]] = !{i32 1, !"stack-protector-guard", !"sysreg"}
// AARCH64: [[ATTR2]] = !{i32 1, !"stack-protector-guard-reg", !"sp_el0"}
// AARCH64: [[ATTR3]] = !{i32 1, !"stack-protector-guard-offset", i32 1024}
