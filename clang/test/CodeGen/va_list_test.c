// RUN: %clang_cc1 -triple powerpc-unknown-freebsd -emit-llvm -o - %s| FileCheck -check-prefix=SVR4-CHECK %s

#include <stdarg.h>

int va_list_size = sizeof(va_list);
// SVR4-CHECK: va_list_size = global i32 12, align 4
