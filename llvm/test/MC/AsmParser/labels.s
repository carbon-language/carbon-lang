// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

        .data:
// CHECK: a:
a:
        .long 0
// CHECK: b:
"b":
        .long 0
// FIXME(quoting): CHECK: a$b:
"a$b":
        .long 0

        .text:
foo:    
// FIXME(quoting): CHECK: val:a$b
        addl $24, "a$b"(%eax)    
// FIXME(quoting): CHECK: val:a$b + 10
        addl $24, ("a$b" + 10)(%eax)
        
// FIXME(quoting): CHECK: b$c = 10
"b$c" = 10
// FIXME(quoting): CHECK: val:10
        addl "b$c", %eax
        
        
