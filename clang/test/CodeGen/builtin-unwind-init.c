// RUN: clang-cc -emit-llvm < %s -o - | grep -F "llvm.eh.unwind.init"

int a() { __builtin_unwind_init(); }

