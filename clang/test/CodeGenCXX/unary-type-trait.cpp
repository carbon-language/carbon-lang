// RUN: %clang_cc1 -emit-llvm-only -verify %s

bool a() { return __is_pod(int); }
