// RUN: %clang     -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -target mips64-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -target mips64el-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// Verify that -fsanitize=memory invokes msan instrumentation.

int foo(int *a) { return *a; }
// CHECK: __msan_init
