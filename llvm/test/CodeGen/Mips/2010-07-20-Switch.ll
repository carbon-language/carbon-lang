; RUN: llc < %s -march=mips -relocation-model=static | FileCheck %s -check-prefix=STATIC-O32 
; RUN: llc < %s -march=mips -relocation-model=pic | FileCheck %s -check-prefix=PIC-O32 
; RUN: llc < %s -march=mips64 -relocation-model=pic -mcpu=mips64 -mattr=n64 | FileCheck %s -check-prefix=PIC-N64

define i32 @main() nounwind readnone {
entry:
  %x = alloca i32, align 4                        ; <i32*> [#uses=2]
  store volatile i32 2, i32* %x, align 4
  %0 = load volatile i32* %x, align 4             ; <i32> [#uses=1]
; STATIC-O32: lui $[[R0:[0-9]+]], %hi($JTI0_0)
; STATIC-O32: addiu ${{[0-9]+}}, $[[R0]], %lo($JTI0_0)
; STATIC-O32: sll ${{[0-9]+}}, ${{[0-9]+}}, 2
; PIC-O32: lw $[[R0:[0-9]+]], %got($JTI0_0)
; PIC-O32: addiu ${{[0-9]+}}, $[[R0]], %lo($JTI0_0)
; PIC-O32: sll ${{[0-9]+}}, ${{[0-9]+}}, 2
; PIC-O32: addu $[[R1:[0-9]+]], ${{[0-9]+}}, $gp
; PIC-O32: jr  $[[R1]]
; PIC-N64: daddiu $[[R2:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(main)))
; PIC-N64: ld $[[R0:[0-9]+]], %got_page($JTI0_0)
; PIC-N64: daddiu ${{[0-9]+}}, $[[R0]], %got_ofst($JTI0_0)
; PIC-N64: dsll ${{[0-9]+}}, ${{[0-9]+}}, 3
; PIC-N64: daddu $[[R1:[0-9]+]], ${{[0-9]+}}, $[[R2]]
; PIC-N64: jr  $[[R1]]
  switch i32 %0, label %bb4 [
    i32 0, label %bb5
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]

bb1:                                              ; preds = %entry
  ret i32 2

; CHECK: STATIC-O32: $BB0_2
bb2:                                              ; preds = %entry
  ret i32 0

bb3:                                              ; preds = %entry
  ret i32 3

bb4:                                              ; preds = %entry
  ret i32 4

bb5:                                              ; preds = %entry
  ret i32 1
}

; STATIC-O32: .align  2
; STATIC-O32: $JTI0_0:
; STATIC-O32: .4byte
; STATIC-O32: .4byte
; STATIC-O32: .4byte
; STATIC-O32: .4byte
; PIC-O32: .align  2
; PIC-O32: $JTI0_0:
; PIC-O32: .gpword
; PIC-O32: .gpword
; PIC-O32: .gpword 
; PIC-O32: .gpword 
; PIC-N64: .align  3
; PIC-N64: $JTI0_0:
; PIC-N64: .gpdword
; PIC-N64: .gpdword
; PIC-N64: .gpdword 
; PIC-N64: .gpdword 

