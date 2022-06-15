// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - -fcf-protection=return %s | FileCheck %s --check-prefix=RETURN
// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - -fcf-protection=branch %s | FileCheck %s --check-prefix=BRANCH
// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - -fcf-protection=full %s   | FileCheck %s --check-prefix=FULL
// RUN: %clang -target i386-unknown-unknown -o - -emit-llvm -S -fcf-protection=branch -mibt-seal -flto %s | FileCheck %s --check-prefixes=CFPROT,IBTSEAL
// RUN: %clang -target i386-unknown-unknown -o - -emit-llvm -S -fcf-protection=branch -flto %s | FileCheck %s --check-prefixes=CFPROT,NOIBTSEAL
// RUN: %clang -target i386-unknown-unknown -o - -emit-llvm -S -fcf-protection=branch -mibt-seal %s | FileCheck %s --check-prefixes=CFPROT,NOIBTSEAL

// RETURN: #define __CET__ 2
// BRANCH: #define __CET__ 1
// FULL: #define __CET__ 3
// CFPROT: "cf-protection-branch", i32 1
// IBTSEAL: "ibt-seal", i32 1
// NOIBTSEAL-NOT: "ibt-seal", i32 1
void foo() {}
