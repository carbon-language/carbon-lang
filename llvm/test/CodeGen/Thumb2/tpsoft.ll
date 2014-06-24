; RUN: llc  %s -mtriple=thumbv7-linux-gnueabi -o - | \
; RUN:    FileCheck  -check-prefix=ELFASM %s
; RUN: llc  %s -mtriple=thumbebv7-linux-gnueabi -o - | \
; RUN:    FileCheck  -check-prefix=ELFASM %s
; RUN: llc  %s -mtriple=thumbv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    llvm-readobj -s -sd | FileCheck  -check-prefix=ELFOBJ -check-prefix=ELFOBJ-LE %s
; RUN: llc  %s -mtriple=thumbebv7-linux-gnueabi -filetype=obj -o - | \
; RUN:    llvm-readobj -s -sd | FileCheck  -check-prefix=ELFOBJ -check-prefix=ELFOBJ-BE %s

;; Make sure that bl __aeabi_read_tp is materialized and fixed up correctly
;; in the obj case.

@i = external thread_local global i32
@a = external global i8
@b = external global [10 x i8]

define arm_aapcs_vfpcc i32 @main() nounwind {
entry:
  %0 = load i32* @i, align 4
  switch i32 %0, label %bb2 [
    i32 12, label %bb
    i32 13, label %bb1
  ]

bb:                                               ; preds = %entry
  %1 = tail call arm_aapcs_vfpcc  i32 @foo(i8* @a) nounwind
  ret i32 %1
; ELFASM:       	bl	__aeabi_read_tp


; ELFOBJ:      Sections [
; ELFOBJ:        Section {
; ELFOBJ:          Name: .text
; ELFOBJ-LE:          SectionData (
;;;                  BL __aeabi_read_tp is ---------+
;;;                                                 V
; ELFOBJ-LE-NEXT:     0000: 2DE90048 0E487844 0168FFF7 FEFF4058
; ELFOBJ-BE:          SectionData (
;;;                  BL __aeabi_read_tp is ---------+
;;;                                                 V
; ELFOBJ-BE-NEXT:     0000: E92D4800 480E4478 6801F7FF FFFE5840


bb1:                                              ; preds = %entry
  %2 = tail call arm_aapcs_vfpcc  i32 @bar(i32* bitcast ([10 x i8]* @b to i32*)) nounwind
  ret i32 %2

bb2:                                              ; preds = %entry
  ret i32 -1
}

declare arm_aapcs_vfpcc i32 @foo(i8*)

declare arm_aapcs_vfpcc i32 @bar(i32*)
