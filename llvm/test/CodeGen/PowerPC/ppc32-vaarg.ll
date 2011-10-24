; RUN: llc -O0 < %s | FileCheck %s
;ModuleID = 'test.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-freebsd9.0"

%struct.__va_list_tag = type { i8, i8, i16, i8*, i8* }

@var1 = common global i64 0, align 8
@var2 = common global double 0.0, align 8
@var3 = common global i32 0, align 4

define void @ppcvaargtest(%struct.__va_list_tag* %ap) nounwind {
 entry:
  %x = va_arg %struct.__va_list_tag* %ap, i64; Get from r5,r6
; CHECK: lbz 4, 0(3)
; CHECK-NEXT: rlwinm 5, 4, 0, 31, 31
; CHECK-NEXT: cmplwi 0, 5, 0
; CHECK-NEXT: addi 5, 4, 1
; CHECK-NEXT: stw 3, -4(1)
; CHECK-NEXT: stw 5, -8(1)
; CHECK-NEXT: stw 4, -12(1)
; CHECK-NEXT: bne 0, .LBB0_2
; CHECK-NEXT: # BB#1:                                 # %entry
; CHECK-NEXT: lwz 3, -12(1)
; CHECK-NEXT: stw 3, -8(1)
; CHECK-NEXT: .LBB0_2:                                # %entry
; CHECK-NEXT: lwz 3, -8(1)
; CHECK-NEXT: slwi 4, 3, 2
; CHECK-NEXT: lwz 5, -4(1)
; CHECK-NEXT: lwz 6, 4(5)
; CHECK-NEXT: lwz 7, 8(5)
; CHECK-NEXT: add 4, 7, 4
; CHECK-NEXT: cmpwi 0, 3, 8
; CHECK-NEXT: mfcr 0                          # cr0
; CHECK-NEXT: stw 0, -16(1)
; CHECK-NEXT: stw 3, -20(1)
; CHECK-NEXT: stw 4, -24(1)
; CHECK-NEXT: stw 6, -28(1)
; CHECK-NEXT: blt 0, .LBB0_4
; CHECK-NEXT: # BB#3:                                 # %entry
; CHECK-NEXT: lwz 3, -28(1)
; CHECK-NEXT: stw 3, -24(1)
; CHECK-NEXT: .LBB0_4:                                # %entry
; CHECK-NEXT: lwz 3, -24(1)
; CHECK-NEXT: lwz 4, -28(1)
; CHECK-NEXT: addi 5, 4, 4
; CHECK-NEXT: lwz 0, -16(1)
; CHECK-NEXT: mtcrf 128, 0
; CHECK-NEXT: stw 4, -32(1)
; CHECK-NEXT: stw 5, -36(1)
; CHECK-NEXT: stw 3, -40(1)
; CHECK-NEXT: blt 0, .LBB0_6
; CHECK-NEXT: # BB#5:                                 # %entry
; CHECK-NEXT: lwz 3, -36(1)
; CHECK-NEXT: stw 3, -32(1)
; CHECK-NEXT: .LBB0_6:                                # %entry
; CHECK-NEXT: lwz 3, -32(1)
; CHECK-NEXT: lwz 4, -20(1)
; CHECK-NEXT: addi 5, 4, 2
; CHECK-NEXT: lwz 6, -4(1)
; CHECK-NEXT: stb 5, 0(6)
; CHECK-NEXT: stw 3, 4(6)
  store i64 %x, i64* @var1, align 8
; CHECK-NEXT: lwz 3, -40(1)
; CHECK-NEXT: lwz 5, 0(3)
; CHECK-NEXT: lwz 7, 4(3)
; CHECK-NEXT: lis 8, var1@ha
; CHECK-NEXT: la 9, var1@l(8)
; CHECK-NEXT: stw 7, 4(9)
; CHECK-NEXT: stw 5, var1@l(8)
  %y = va_arg %struct.__va_list_tag* %ap, double; From f1
; CHECK-NEXT: lbz 5, 1(6)
; CHECK-NEXT: lwz 7, 4(6)
; CHECK-NEXT: lwz 8, 8(6)
; CHECK-NEXT: slwi 9, 5, 3
; CHECK-NEXT: add 8, 8, 9
; CHECK-NEXT: cmpwi 0, 5, 8
; CHECK-NEXT: addi 9, 7, 8
; CHECK-NEXT: mr 10, 7
; CHECK-NEXT: stw 9, -44(1)
; CHECK-NEXT: stw 7, -48(1)
; CHECK-NEXT: mfcr 0                          # cr0
; CHECK-NEXT: stw 0, -52(1)
; CHECK-NEXT: stw 5, -56(1)
; CHECK-NEXT: stw 10, -60(1)
; CHECK-NEXT: stw 8, -64(1)
; CHECK-NEXT: blt 0, .LBB0_8
; CHECK-NEXT: # BB#7:                                 # %entry
; CHECK-NEXT: lwz 3, -44(1)
; CHECK-NEXT: stw 3, -60(1)
; CHECK-NEXT: .LBB0_8:                                # %entry
; CHECK-NEXT: lwz 3, -60(1)
; CHECK-NEXT: lwz 4, -64(1)
; CHECK-NEXT: addi 4, 4, 32
; CHECK-NEXT: lwz 0, -52(1)
; CHECK-NEXT: mtcrf 128, 0
; CHECK-NEXT: stw 4, -68(1)
; CHECK-NEXT: stw 3, -72(1)
; CHECK-NEXT: blt 0, .LBB0_10
; CHECK-NEXT: # BB#9:                                 # %entry
; CHECK-NEXT: lwz 3, -48(1)
; CHECK-NEXT: stw 3, -68(1)
; CHECK-NEXT: .LBB0_10:                               # %entry
; CHECK-NEXT: lwz 3, -68(1)
; CHECK-NEXT: lwz 4, -56(1)
; CHECK-NEXT: addi 5, 4, 1
; CHECK-NEXT: lwz 6, -4(1)
; CHECK-NEXT: stb 5, 1(6)
; CHECK-NEXT: lwz 5, -72(1)
; CHECK-NEXT: stw 5, 4(6)
; CHECK-NEXT: lfd 0, 0(3)
  store double %y, double* @var2, align 8
; CHECK-NEXT: lis 3, var2@ha
; CHECK-NEXT: stfd 0, var2@l(3)
  %z = va_arg %struct.__va_list_tag* %ap, i32; From r7
; CHECK-NEXT: lbz 3, 0(6)
; CHECK-NEXT: lwz 5, 4(6)
; CHECK-NEXT: lwz 7, 8(6)
; CHECK-NEXT: slwi 8, 3, 2
; CHECK-NEXT: add 7, 7, 8
; CHECK-NEXT: cmpwi 0, 3, 8
; CHECK-NEXT: addi 8, 5, 4
; CHECK-NEXT: mr 9, 5
; CHECK-NEXT: stw 3, -76(1)
; CHECK-NEXT: stw 7, -80(1)
; CHECK-NEXT: stw 8, -84(1)
; CHECK-NEXT: stw 5, -88(1)
; CHECK-NEXT: stw 9, -92(1)
; CHECK-NEXT: mfcr 0                          # cr0
; CHECK-NEXT: stw 0, -96(1)
; CHECK-NEXT: blt 0, .LBB0_12
; CHECK-NEXT: # BB#11:                                # %entry
; CHECK-NEXT: lwz 3, -84(1)
; CHECK-NEXT: stw 3, -92(1)
; CHECK-NEXT: .LBB0_12:                               # %entry
; CHECK-NEXT: lwz 3, -92(1)
; CHECK-NEXT: lwz 4, -80(1)
; CHECK-NEXT: lwz 0, -96(1)
; CHECK-NEXT: mtcrf 128, 0
; CHECK-NEXT: stw 3, -100(1)
; CHECK-NEXT: stw 4, -104(1)
; CHECK-NEXT: blt 0, .LBB0_14
; CHECK-NEXT: # BB#13:                                # %entry
; CHECK-NEXT: lwz 3, -88(1)
; CHECK-NEXT: stw 3, -104(1)
; CHECK-NEXT: .LBB0_14:                               # %entry
; CHECK-NEXT: lwz 3, -104(1)
; CHECK-NEXT: lwz 4, -76(1)
; CHECK-NEXT: addi 5, 4, 1
; CHECK-NEXT: lwz 6, -4(1)
; CHECK-NEXT: stb 5, 0(6)
; CHECK-NEXT: lwz 5, -100(1)
; CHECK-NEXT: stw 5, 4(6)
; CHECK-NEXT: lwz 3, 0(3)
  store i32 %z, i32* @var3, align 4
; CHECK-NEXT: lis 5, var3@ha
; CHECK-NEXT: stw 3, var3@l(5)
  ret void
; CHECK-NEXT: stw 6, -108(1)
; CHECK-NEXT: blr 
}

