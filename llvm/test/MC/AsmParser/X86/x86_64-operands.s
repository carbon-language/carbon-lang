// FIXME: Actually test that we get the expected results.
        
// RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

# CHECK: callq a
        callq a

        
