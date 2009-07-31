// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .data
// CHECK: a:
a:
        .long 0
// CHECK: b:
"b":
        .long 0
// CHECK: "a$b":
"a$b":
        .long 0

        .text
foo:    
// CHECK: val:"a$b"
        addl $24, "a$b"(%eax)    
// CHECK: val:"a$b" + 10
        addl $24, ("a$b" + 10)(%eax)
        
// CHECK: "b$c" = 10
"b$c" = 10
// CHECK: val:10
        addl "b$c", %eax
        
        
