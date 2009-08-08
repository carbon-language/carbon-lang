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
        
        
// CHECK: set "a 0", 11
.set "a 0", 11
// CHECK: .long 11
        .long "a 0"

// XXCHCK: .section "a 1,a 2"
//.section "a 1", "a 2"

// CHECK: .globl "a 3"
        .globl "a 3"

// CHECK: .weak "a 4"
        .weak "a 4"

// CHECK: .desc "a 5",1
        .desc "a 5", 1

// CHECK: .comm "a 6",1
        .comm "a 6", 1

// CHECK: .lcomm "a 7",1
        .lcomm "a 7", 1

// CHECK: .lsym "a 8",1
        .lsym "a 8", 1
