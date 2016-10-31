// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -O0  -S -o - -Wall -Werror | FileCheck %s
// This test checks basic inline assembly recognition of k0-k7 registers for avx512.

void test_basic_inline_asm_with_k_regs() {
    //CHECK: ## InlineAsm Start
    //CHECK: kandw %k1, %k2, %k3
    //CHECK: ## InlineAsm End
    asm("kandw %k1, %k2, %k3\t");
    //CHECK: ## InlineAsm Start
    //CHECK: kandw %k4, %k5, %k6
    //CHECK: ## InlineAsm End
    asm("kandw %k4, %k5, %k6\t");
    //CHECK: ## InlineAsm Start
    //CHECK: kandw %k7, %k0, %k1
    //CHECK: ## InlineAsm End
    asm("kandw %k7, %k0, %k1\t");
}
