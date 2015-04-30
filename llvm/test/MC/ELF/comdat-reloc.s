// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sd | FileCheck %s

  .text
  .globl  hello
  .type  hello,@function
hello:
  call  world
  ret

  .section  .text.world,"axG",@progbits,world,comdat
  .type  world,@function
world:
  call  doctor
  ret

// CHECK:  Name: .group
// CHECK-NOT: SectionData
// CHECK: SectionData
// CHECK-NEXT: 0000: 01000000 07000000 08000000

// CHECK: Index: 7
// CHECK-NEXT: Name: .text.world
// CHECK-NOT: Section {
// CHECK: SHF_GROUP

// CHECK: Index: 8
// CHECK-NEXT: Name: .rela.text.world
// CHECK-NOT: Section {
// CHECK: SHF_GROUP
