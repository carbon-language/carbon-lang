// This test verifies that we produce target-cpu and target-features attributes
// on functions when they're different from the standard cpu and have written
// features.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-feature +avx | FileCheck %s -check-prefix=AVX-FEATURE
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-feature +avx | FileCheck %s -check-prefix=AVX-NO-CPU
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-feature +avx512f -target-feature +avx512er | FileCheck %s -check-prefix=TWO-AVX
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-cpu corei7 | FileCheck %s -check-prefix=CORE-CPU
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-cpu corei7 -target-feature +avx | FileCheck %s -check-prefix=CORE-CPU-AND-FEATURES
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-cpu x86-64 | FileCheck %s -check-prefix=X86-64-CPU-NOT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-cpu corei7-avx -target-feature -avx | FileCheck %s -check-prefix=AVX-MINUS-FEATURE

void foo() {}

// AVX-FEATURE: "target-features"="+avx"
// AVX-NO-CPU-NOT: target-cpu
// TWO-AVX: "target-features"="+avx512f,+avx512er"
// CORE-CPU: "target-cpu"="corei7"
// CORE-CPU-AND-FEATURES: "target-cpu"="corei7" "target-features"="+avx"
// X86-64-CPU-NOT: "target-cpu"
// AVX-MINUS-FEATURE: "target-features"="-avx"
