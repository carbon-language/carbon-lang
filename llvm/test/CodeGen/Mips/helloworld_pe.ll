; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0))
  ret i32 0

; 16: 	.set	mips16                  # @main

; 16:	li	$[[T1:[0-9]+]], %hi(_gp_disp)
; 16: 	addiu	$[[T2:[0-9]+]], $pc, %lo(_gp_disp)
; 16:	sll	$[[T3:[0-9]+]], $[[T1]], 16
; 16:	li	$2, 0
}

declare i32 @printf(i8*, ...)
