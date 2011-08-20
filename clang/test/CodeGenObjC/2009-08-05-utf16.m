// RUN: %clang_cc1 -emit-llvm -w -x objective-c %s -o - | FileCheck %s
// rdar://7095855 rdar://7115749

// CHECK: internal unnamed_addr constant [12 x i8]
void *P = @"iPod™";
