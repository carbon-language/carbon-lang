// These should be valid cxx abi flags for these targets.
// RUN: %clang_cc1 -fc++-abi=itanium -triple x86_64-unknown-linux-gnu %s -S -emit-llvm -o /dev/null
// RUN: %clang_cc1 -fc++-abi=fuchsia -triple x86_64-unknown-fuchsia %s -S -emit-llvm -o /dev/null
// RUN: %clang_cc1 -fc++-abi=microsoft -triple x86_64-windows-msvc %s -S -emit-llvm -o /dev/null
