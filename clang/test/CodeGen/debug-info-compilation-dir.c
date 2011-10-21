// RUN: %clang_cc1 -fdebug-compilation-dir /nonsense -emit-llvm -g %s -o - | \
// RUN:   grep nonsense

// RUN: %clang_cc1 -emit-llvm -g %s -o - | grep %S
