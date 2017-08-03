// RUN: %clang_cc1 -triple mips-mti--elf -emit-llvm -mrelocation-model static \
// RUN:            -target-feature +noabicalls -mllvm -mgpopt -mllvm \
// RUN:            -membedded-data=1 -muninit-const-in-rodata -o - %s | \
// RUN:   FileCheck %s

// REQUIRES: mips-registered-target

// Test that -muninit-const-in-rodata places constant uninitialized structures
// in the .rodata section rather than the commeon section.

// CHECK: @a = global [8 x i32] zeroinitializer, section "rodata", align 4
const int a[8];
