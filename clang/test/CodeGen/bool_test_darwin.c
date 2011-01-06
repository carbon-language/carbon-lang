// RUN: %clang_cc1 -triple powerpc-apple-darwin -emit-llvm -o - %s| FileCheck %s

int boolsize = sizeof(_Bool);
//CHECK: boolsize = global i32 4, align 4

