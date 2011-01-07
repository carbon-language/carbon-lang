// RUN: %clang_cc1 -triple powerpc-unknown-freebsd -emit-llvm -o - %s| FileCheck %s

#include <stdarg.h>

int va_list_size = sizeof(va_list);
// CHECK: va_list_size = global i32 12, align 4
