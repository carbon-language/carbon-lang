; RUN: llc < %s -mtriple=thumb-none-eabi -mcpu=cortex-m4 2>&1 | FileCheck %s --check-prefix=MCORE
; RUN: not llc < %s -mtriple=thumb-none-eabi -mcpu=cortex-m3 2>&1 | FileCheck %s --check-prefix=M3CORE
; RUN: not llc < %s -mtriple=arm-none-eabi -mcpu=cortex-a8 2>&1 | FileCheck %s --check-prefix=ACORE

; ACORE: LLVM ERROR: Invalid register name "control".
; M3CORE: LLVM ERROR: Invalid register name "xpsr_nzcvqg".

define i32 @read_mclass_registers() nounwind {
entry:
  ; MCORE-LABEL: read_mclass_registers:
  ; MCORE:   mrs r0, apsr
  ; MCORE:   mrs r1, iapsr
  ; MCORE:   mrs r1, eapsr
  ; MCORE:   mrs r1, xpsr
  ; MCORE:   mrs r1, ipsr
  ; MCORE:   mrs r1, epsr
  ; MCORE:   mrs r1, iepsr
  ; MCORE:   mrs r1, msp
  ; MCORE:   mrs r1, psp
  ; MCORE:   mrs r1, primask
  ; MCORE:   mrs r1, basepri
  ; MCORE:   mrs r1, basepri_max
  ; MCORE:   mrs r1, faultmask
  ; MCORE:   mrs r1, control

  %0 = call i32 @llvm.read_register.i32(metadata !0)
  %1 = call i32 @llvm.read_register.i32(metadata !4)
  %add1 = add i32 %1, %0
  %2 = call i32 @llvm.read_register.i32(metadata !8)
  %add2 = add i32 %add1, %2
  %3 = call i32 @llvm.read_register.i32(metadata !12)
  %add3 = add i32 %add2, %3
  %4 = call i32 @llvm.read_register.i32(metadata !16)
  %add4 = add i32 %add3, %4
  %5 = call i32 @llvm.read_register.i32(metadata !17)
  %add5 = add i32 %add4, %5
  %6 = call i32 @llvm.read_register.i32(metadata !18)
  %add6 = add i32 %add5, %6
  %7 = call i32 @llvm.read_register.i32(metadata !19)
  %add7 = add i32 %add6, %7
  %8 = call i32 @llvm.read_register.i32(metadata !20)
  %add8 = add i32 %add7, %8
  %9 = call i32 @llvm.read_register.i32(metadata !21)
  %add9 = add i32 %add8, %9
  %10 = call i32 @llvm.read_register.i32(metadata !22)
  %add10 = add i32 %add9, %10
  %11 = call i32 @llvm.read_register.i32(metadata !23)
  %add11 = add i32 %add10, %11
  %12 = call i32 @llvm.read_register.i32(metadata !24)
  %add12 = add i32 %add11, %12
  %13 = call i32 @llvm.read_register.i32(metadata !25)
  %add13 = add i32 %add12, %13
  ret i32 %add13
}

define void @write_mclass_registers(i32 %x) nounwind {
entry:
  ; MCORE-LABEL: write_mclass_registers:
  ; MCORE:   msr apsr_nzcvqg, r0
  ; MCORE:   msr apsr_nzcvq, r0
  ; MCORE:   msr apsr_g, r0
  ; MCORE:   msr apsr_nzcvqg, r0
  ; MCORE:   msr iapsr_nzcvqg, r0
  ; MCORE:   msr iapsr_nzcvq, r0
  ; MCORE:   msr iapsr_g, r0
  ; MCORE:   msr iapsr_nzcvqg, r0
  ; MCORE:   msr eapsr_nzcvqg, r0
  ; MCORE:   msr eapsr_nzcvq, r0
  ; MCORE:   msr eapsr_g, r0
  ; MCORE:   msr eapsr_nzcvqg, r0
  ; MCORE:   msr xpsr_nzcvqg, r0
  ; MCORE:   msr xpsr_nzcvq, r0
  ; MCORE:   msr xpsr_g, r0
  ; MCORE:   msr xpsr_nzcvqg, r0
  ; MCORE:   msr ipsr, r0
  ; MCORE:   msr epsr, r0
  ; MCORE:   msr iepsr, r0
  ; MCORE:   msr msp, r0
  ; MCORE:   msr psp, r0
  ; MCORE:   msr primask, r0
  ; MCORE:   msr basepri, r0
  ; MCORE:   msr basepri_max, r0
  ; MCORE:   msr faultmask, r0
  ; MCORE:   msr control, r0

  call void @llvm.write_register.i32(metadata !0, i32 %x)
  call void @llvm.write_register.i32(metadata !1, i32 %x)
  call void @llvm.write_register.i32(metadata !2, i32 %x)
  call void @llvm.write_register.i32(metadata !3, i32 %x)
  call void @llvm.write_register.i32(metadata !4, i32 %x)
  call void @llvm.write_register.i32(metadata !5, i32 %x)
  call void @llvm.write_register.i32(metadata !6, i32 %x)
  call void @llvm.write_register.i32(metadata !7, i32 %x)
  call void @llvm.write_register.i32(metadata !8, i32 %x)
  call void @llvm.write_register.i32(metadata !9, i32 %x)
  call void @llvm.write_register.i32(metadata !10, i32 %x)
  call void @llvm.write_register.i32(metadata !11, i32 %x)
  call void @llvm.write_register.i32(metadata !12, i32 %x)
  call void @llvm.write_register.i32(metadata !13, i32 %x)
  call void @llvm.write_register.i32(metadata !14, i32 %x)
  call void @llvm.write_register.i32(metadata !15, i32 %x)
  call void @llvm.write_register.i32(metadata !16, i32 %x)
  call void @llvm.write_register.i32(metadata !17, i32 %x)
  call void @llvm.write_register.i32(metadata !18, i32 %x)
  call void @llvm.write_register.i32(metadata !19, i32 %x)
  call void @llvm.write_register.i32(metadata !20, i32 %x)
  call void @llvm.write_register.i32(metadata !21, i32 %x)
  call void @llvm.write_register.i32(metadata !22, i32 %x)
  call void @llvm.write_register.i32(metadata !23, i32 %x)
  call void @llvm.write_register.i32(metadata !24, i32 %x)
  call void @llvm.write_register.i32(metadata !25, i32 %x)
  ret void
}

declare i32 @llvm.read_register.i32(metadata) nounwind
declare void @llvm.write_register.i32(metadata, i32) nounwind

!0 = !{!"apsr"}
!1 = !{!"apsr_nzcvq"}
!2 = !{!"apsr_g"}
!3 = !{!"apsr_nzcvqg"}
!4 = !{!"iapsr"}
!5 = !{!"iapsr_nzcvq"}
!6 = !{!"iapsr_g"}
!7 = !{!"iapsr_nzcvqg"}
!8 = !{!"eapsr"}
!9 = !{!"eapsr_nzcvq"}
!10 = !{!"eapsr_g"}
!11 = !{!"eapsr_nzcvqg"}
!12 = !{!"xpsr"}
!13 = !{!"xpsr_nzcvq"}
!14 = !{!"xpsr_g"}
!15 = !{!"xpsr_nzcvqg"}
!16 = !{!"ipsr"}
!17 = !{!"epsr"}
!18 = !{!"iepsr"}
!19 = !{!"msp"}
!20 = !{!"psp"}
!21 = !{!"primask"}
!22 = !{!"basepri"}
!23 = !{!"basepri_max"}
!24 = !{!"faultmask"}
!25 = !{!"control"}
