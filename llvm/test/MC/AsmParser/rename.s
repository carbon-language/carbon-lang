// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .size bar, . - bar
.Ltmp0:
       .size foo, .Ltmp0 - foo

// CHECK: .Ltmp0:
// CHECK: .size  bar, .Ltmp0-bar
// CHECK: .Ltmp01
// CHECK: .size foo, .Ltmp01-foo
