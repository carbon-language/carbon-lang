// RUN: %clang_cc1 -triple armv7 -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -triple aarch64 -target-cpu cortex-a53 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple armv7 -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple aarch64 -target-cpu cortex-a57 -fsyntax-only -ffreestanding %s
// expected-no-diagnostics

#include <arm_acle.h>
