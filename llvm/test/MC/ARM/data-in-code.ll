;; RUN: llc -O0 -mtriple=armv7-linux-gnueabi -filetype=obj %s -o - | \
;; RUN:   elf-dump | FileCheck -check-prefix=ARM %s

;; RUN: llc -O0 -mtriple=thumbv7-linux-gnueabi -filetype=obj %s -o - | \
;; RUN:   elf-dump --dump-section-data | FileCheck -check-prefix=TMB %s

;; Ensure that if a jump table is generated that it has Mapping Symbols
;; marking the data-in-code region.

define void @foo(i32* %ptr) nounwind ssp {
  %tmp = load i32* %ptr, align 4
  switch i32 %tmp, label %default [
    i32 11, label %bb0
    i32 10, label %bb1
    i32 8, label %bb2
    i32 4, label %bb3
    i32 2, label %bb4
    i32 6, label %bb5
    i32 9, label %bb6
    i32 15, label %bb7
    i32 1, label %bb8
    i32 3, label %bb9
    i32 5, label %bb10
    i32 30, label %bb11
    i32 31, label %bb12
    i32 13, label %bb13
    i32 14, label %bb14
    i32 20, label %bb15
    i32 19, label %bb16
    i32 17, label %bb17
    i32 18, label %bb18
    i32 21, label %bb19
    i32 22, label %bb20
    i32 16, label %bb21
    i32 24, label %bb22
    i32 25, label %bb23
    i32 26, label %bb24
    i32 27, label %bb25
    i32 28, label %bb26
    i32 23, label %bb27
    i32 12, label %bb28
  ]

default:
  br label %exit
bb0:
  br label %exit
bb1:
  br label %exit
bb2:
  br label %exit
bb3:
  br label %exit
bb4:
  br label %exit
bb5:
  br label %exit
bb6:
  br label %exit
bb7:
  br label %exit
bb8:
  br label %exit
bb9:
  br label %exit
bb10:
  br label %exit
bb11:
  br label %exit
bb12:
  br label %exit
bb13:
  br label %exit
bb14:
  br label %exit
bb15:
  br label %exit
bb16:
  br label %exit
bb17:
  br label %exit
bb18:
  br label %exit
bb19:
  br label %exit
bb20:
  br label %exit
bb21:
  br label %exit
bb22:
  br label %exit
bb23:
  br label %exit
bb24:
  br label %exit
bb25:
  br label %exit
bb26:
  br label %exit
bb27:
  br label %exit
bb28:
  br label %exit


exit:

  ret void
}

;; ARM:         # Symbol 2
;; ARM-NEXT:    $a
;; ARM-NEXT:   'st_value', 0x00000000
;; ARM-NEXT:   'st_size', 0x00000000
;; ARM-NEXT:   'st_bind', 0x0
;; ARM-NEXT:   'st_type', 0x0
;; ARM-NEXT:   'st_other'
;; ARM-NEXT:   'st_shndx', [[MIXED_SECT:0x[0-9a-f]+]]

;; ARM:         # Symbol 3
;; ARM-NEXT:    $a
;; ARM-NEXT:   'st_value', 0x000000ac
;; ARM-NEXT:   'st_size', 0x00000000
;; ARM-NEXT:   'st_bind', 0x0
;; ARM-NEXT:   'st_type', 0x0
;; ARM-NEXT:   'st_other'
;; ARM-NEXT:   'st_shndx', [[MIXED_SECT]]

;; ARM:         # Symbol 4
;; ARM-NEXT:    $d
;; ARM-NEXT:    'st_value', 0x00000000
;; ARM-NEXT:    'st_size', 0x00000000
;; ARM-NEXT:    'st_bind', 0x0
;; ARM-NEXT:    'st_type', 0x0

;; ARM:         # Symbol 5
;; ARM-NEXT:    $d
;; ARM-NEXT:   'st_value', 0x00000030
;; ARM-NEXT:   'st_size', 0x00000000
;; ARM-NEXT:   'st_bind', 0x0
;; ARM-NEXT:   'st_type', 0x0
;; ARM-NEXT:   'st_other'
;; ARM-NEXT:   'st_shndx', [[MIXED_SECT]]

;; ARM-NOT:     ${{[atd]}}

;; TMB:         # Symbol 3
;; TMB-NEXT:    $d
;; TMB-NEXT:   'st_value', 0x00000016
;; TMB-NEXT:   'st_size', 0x00000000
;; TMB-NEXT:   'st_bind', 0x0
;; TMB-NEXT:   'st_type', 0x0
;; TMB-NEXT:   'st_other'
;; TMB-NEXT:   'st_shndx', [[MIXED_SECT:0x[0-9a-f]+]]

;; TMB:         # Symbol 4
;; TMB-NEXT:    $t
;; TMB-NEXT:   'st_value', 0x00000000
;; TMB-NEXT:   'st_size', 0x00000000
;; TMB-NEXT:   'st_bind', 0x0
;; TMB-NEXT:   'st_type', 0x0
;; TMB-NEXT:   'st_other'
;; TMB-NEXT:   'st_shndx', [[MIXED_SECT]]

;; TMB:         # Symbol 5
;; TMB-NEXT:    $t
;; TMB-NEXT:   'st_value', 0x00000036
;; TMB-NEXT:   'st_size', 0x00000000
;; TMB-NEXT:   'st_bind', 0x0
;; TMB-NEXT:   'st_type', 0x0
;; TMB-NEXT:   'st_other'
;; TMB-NEXT:   'st_shndx', [[MIXED_SECT]]


;; TMB-NOT:     ${{[atd]}}

