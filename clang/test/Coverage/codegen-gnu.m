// RUN: %clang_cc1 -triple i386-unknown-unknown -fobjc-exceptions -fobjc-runtime=gnu -emit-llvm -o %t %s

#include "objc-language-features.inc"
