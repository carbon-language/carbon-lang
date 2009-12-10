// RUN: clang-cc -emit-llvm-only -verify %s

bool a() { return __is_pod(int); }
