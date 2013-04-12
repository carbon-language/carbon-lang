; RUN: llc  %s -mtriple=armv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    llvm-readobj -s -sr -sd | FileCheck  -check-prefix=OBJ %s

target triple = "armv7-none-linux-gnueabi"

@a = external global i8

define arm_aapcs_vfpcc i32 @barf() nounwind {
entry:
  %0 = tail call arm_aapcs_vfpcc  i32 @foo(i8* @a) nounwind
  ret i32 %0
; OBJ:        Section {
; OBJ:          Name: .text
; OBJ:          Relocations [
; OBJ-NEXT:       0x4 R_ARM_MOVW_ABS_NC a
; OBJ-NEXT:       0x8 R_ARM_MOVT_ABS
; OBJ-NEXT:       0xC R_ARM_CALL foo
; OBJ-NEXT:     ]
; OBJ-NEXT:     SectionData (
; OBJ-NEXT:       0000: 00482DE9 000000E3 000040E3 FEFFFFEB
; OBJ-NEXT:       0010: 0088BDE8
; OBJ-NEXT:     )

}

declare arm_aapcs_vfpcc i32 @foo(i8*)

