// RUN: %clang_cc1 -triple armv7-eabi -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -triple aarch64-eabi -target-cpu cortex-a53 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -triple thumbv7-windows -target-cpu cortex-a53 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple armv7-eabi -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple aarch64-eabi -target-cpu cortex-a57 -fsyntax-only -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple thumbv7-windows -target-cpu cortex-a15 -fsyntax-only -ffreestanding %s
// expected-no-diagnostics

#include <arm_acle.h>
