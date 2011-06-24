// RUN: %clang_cc1 -emit-llvm -g < %s | grep DW_TAG_member | grep \!3

struct A { int x; } a;
