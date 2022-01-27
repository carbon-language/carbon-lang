// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple aarch64_be -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple i386 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple riscv32 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple riscv64 -fsyntax-only -verify=silence %s
// RUN: %clang_cc1 -triple ppc64le -fsyntax-only -verify %s

// silence-no-diagnostics

// expected-warning@+1 {{unknown attribute 'patchable_function_entry' ignored}}
[[gnu::patchable_function_entry(0)]] void f();
