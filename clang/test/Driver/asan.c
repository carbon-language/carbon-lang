// RUN: %clang     -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O1 -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O3 -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang     -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O1 -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O3 -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// Verify that -fsanitize={address,kernel-address} invoke ASan and KASan instrumentation.

int foo(int *a) { return *a; }
// CHECK-ASAN: __asan_init
// CHECK-KASAN: __asan_load4_noabort
