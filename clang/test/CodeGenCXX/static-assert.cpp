// RUN: clang-cc %s -emit-llvm -o - -std=c++0x

static_assert(true, "");
