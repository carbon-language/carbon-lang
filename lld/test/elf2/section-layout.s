# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld2 %t -o %tout
# RUN: llvm-readobj -sections %tout | FileCheck %s
# REQUIRES: x86

# Check that sections are laid out in the correct order.

.global _start
.text
_start:

.section h,""
.section g,"",@nobits
.section f,"aw",@nobits
.section e,"aw"
.section d,"ax",@nobits
.section c,"ax"
.section b,"a",@nobits
.section a,"a"

// CHECK: Name: a
// CHECK: Name: b
// CHECK: Name: c
// CHECK: Name: d
// CHECK: Name: e
// CHECK: Name: f
// CHECK: Name: h
// CHECK: Name: g
