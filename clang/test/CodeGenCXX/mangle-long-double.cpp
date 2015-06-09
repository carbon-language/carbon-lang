// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=POWER64-LINUX
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu   %s -emit-llvm -o - | FileCheck %s --check-prefix=POWER-LINUX
// RUN: %clang_cc1 -triple powerpc64-apple-darwin9     %s -emit-llvm -o - | FileCheck %s --check-prefix=POWER64-DARWIN
// RUN: %clang_cc1 -triple powerpc-apple-darwin9       %s -emit-llvm -o - | FileCheck %s --check-prefix=POWER-DARWIN

void f(long double) {}
// POWER64-LINUX:  _Z1fg
// POWER-LINUX:    _Z1fg
// POWER64-DARWIN: _Z1fe
// POWER-DARWIN:   _Z1fe
