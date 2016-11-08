// Test API-related defines for various Android targets.
//
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target arm-linux-androideabi \
// RUN:   | FileCheck %s
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target arm-linux-androideabi19 \
// RUN:   | FileCheck %s -check-prefix=LEVEL19
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target arm-linux-androideabi20 \
// RUN:   | FileCheck %s -check-prefix=LEVEL20
//
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target aarch64-linux-android \
// RUN:   | FileCheck %s
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target aarch64-linux-android19 \
// RUN:   | FileCheck %s -check-prefix=LEVEL19
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target aarch64-linux-android20 \
// RUN:   | FileCheck %s -check-prefix=LEVEL20
//
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target i686-linux-android \
// RUN:   | FileCheck %s
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target i686-linux-android19 \
// RUN:   | FileCheck %s -check-prefix=LEVEL19
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target i686-linux-android20 \
// RUN:   | FileCheck %s -check-prefix=LEVEL20
//
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target x86_64-linux-android \
// RUN:   | FileCheck %s
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target x86_64-linux-android19 \
// RUN:   | FileCheck %s -check-prefix=LEVEL19
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target x86_64-linux-android20 \
// RUN:   | FileCheck %s -check-prefix=LEVEL20
//
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target mipsel-linux-android \
// RUN:   | FileCheck %s
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target mipsel-linux-android19 \
// RUN:   | FileCheck %s -check-prefix=LEVEL19
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target mipsel-linux-android20 \
// RUN:   | FileCheck %s -check-prefix=LEVEL20
//
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target mips64el-linux-android \
// RUN:   | FileCheck %s
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target mips64el-linux-android19 \
// RUN:   | FileCheck %s -check-prefix=LEVEL19
// RUN: %clang %s -emit-llvm -S -c -o - \
// RUN:     -target mips64el-linux-android20 \
// RUN:   | FileCheck %s -check-prefix=LEVEL20

// CHECK: __ANDROID__defined
// LEVEL19: __ANDROID__defined
// LEVEL20: __ANDROID__defined
#ifdef __ANDROID__
void __ANDROID__defined(void) {}
#endif

// CHECK-NOT: __ANDROID_API__defined
// LEVEL19: __ANDROID_API__defined
// LEVEL20: __ANDROID_API__defined
#ifdef __ANDROID_API__
void __ANDROID_API__defined(void) {}
int android_api = __ANDROID_API__;
#endif

// CHECK-NOT: __ANDROID_API__20
// LEVEL19-NOT: __ANDROID_API__20
// LEVEL20: __ANDROID_API__20
#if __ANDROID_API__ >= 20
void __ANDROID_API__20(void) {}
#endif
