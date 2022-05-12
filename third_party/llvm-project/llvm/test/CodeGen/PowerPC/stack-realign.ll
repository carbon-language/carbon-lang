; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -frame-pointer=all < %s | FileCheck -check-prefix=CHECK-FP %s
; RUN: llc -mtriple=powerpc-unknown-linux-gnu -frame-pointer=all < %s | FileCheck -check-prefix=CHECK-32 %s
; RUN: llc -mtriple=powerpc-unknown-linux-gnu -frame-pointer=all -relocation-model=pic < %s | FileCheck -check-prefix=CHECK-32-PIC %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.s = type { i32, i32 }

declare void @bar(i32*)

@barbaz = external global i32

define void @goo(%struct.s* byval(%struct.s) nocapture readonly %a) {
entry:
  %x = alloca [2 x i32], align 32
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 32
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %2 = load i32, i32* @barbaz, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 1
  store i32 %2, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx)
  ret void
}

; CHECK-LABEL: @goo

; CHECK-DAG: mflr {{[0-9]+}}
; CHECK-DAG: clrldi [[REG:[0-9]+]], 1, 59
; CHECK-DAG: std 30, -16(1)
; CHECK-DAG: mr 30, 1
; CHECK-DAG: std 0, 16(1)
; CHECK-DAG: subfic 0, [[REG]], -160
; CHECK: stdux 1, 1, 0

; CHECK: .cfi_def_cfa_register r30
; CHECK: .cfi_offset r30, -16
; CHECK: .cfi_offset lr, 16

; CHECK: std 3, 48(30)

; CHECK: mr 1, 30
; CHECK-DAG: ld [[SR:[0-9]+]], 16(1)
; CHECK-DAG: ld 30, -16(1)
; CHECK-DAG: mtlr [[SR]]
; CHECK: blr

; CHECK-FP-LABEL: @goo

; CHECK-FP-DAG: mflr {{[0-9]+}}
; CHECK-FP-DAG: clrldi [[REG:[0-9]+]], 1, 59
; CHECK-FP-DAG: std 31, -8(1)
; CHECK-FP-DAG: std 30, -16(1)
; CHECK-FP-DAG: mr 30, 1
; CHECK-FP-DAG: std 0, 16(1)
; CHECK-FP-DAG: subfic 0, [[REG]], -160
; CHECK-FP: stdux 1, 1, 0

; CHECK-FP: .cfi_def_cfa_register r30
; CHECK-FP: .cfi_offset r31, -8
; CHECK-FP: .cfi_offset r30, -16
; CHECK-FP: .cfi_offset lr, 16

; CHECK-FP: mr 31, 1

; CHECK-FP: std 3, 48(30)

; CHECK-FP: mr 1, 30
; CHECK-FP-DAG: ld [[SR:[0-9]+]], 16(1)
; CHECK-FP-DAG: ld 31, -8(1)
; CHECK-FP-DAG: ld 30, -16(1)
; CHECK-FP-DAG: mtlr [[SR]]
; CHECK-FP: blr

; CHECK-32-LABEL: @goo
; CHECK-32-DAG: mflr [[LR:[0-9]+]]
; CHECK-32-DAG: clrlwi [[REG:[0-9]+]], 1, 27
; CHECK-32-DAG: stw [[LR]], 4(1)
; CHECK-32-DAG: subfic 0, [[REG]], -64
; CHECK-32: stwux 1, 1, 0
; CHECK-32: sub 0, 1, 0
; CHECK-32: addic 0, 0, -4
; CHECK-32: stwx 31, 0, 0
; CHECK-32: addic 0, 0, -4
; CHECK-32: stwx 30, 0, 0
; CHECK-32: addic 30, 0, 8

; CHECK-32-PIC-LABEL: @goo
; CHECK-32-PIC-DAG: mflr [[LR:[0-9]+]]
; CHECK-32-PIC-DAG: clrlwi [[REG:[0-9]+]], 1, 27
; CHECK-32-PIC-DAG: stw [[LR]], 4(1)
; CHECK-32-PIC-DAG: subfic 0, [[REG]], -64
; CHECK-32-PIC:     stwux 1, 1, 0
; CHECK-32-PIC:     sub 0, 1, 0
; CHECK-32-PIC:     addic 0, 0, -4
; CHECK-32-PIC:     stwx 31, 0, 0
; CHECK-32-PIC:     addic 0, 0, -4
; CHECK-32-PIC:     stwx 30, 0, 0
; CHECK-32-PIC:     addic 0, 0, -4
; CHECK-32-PIC:     stwx 29, 0, 0
; CHECK-32-PIC:     addic 29, 0, 12

; The large-frame-size case.
define void @hoo(%struct.s* byval(%struct.s) nocapture readonly %a) {
entry:
  %x = alloca [200000 x i32], align 32
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [200000 x i32], [200000 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 32
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [200000 x i32], [200000 x i32]* %x, i64 0, i64 1
  store i32 %1, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx)
  ret void
}

; CHECK-LABEL: @hoo

; CHECK-DAG: lis [[REG1:[0-9]+]], -13
; CHECK-DAG: clrldi [[REG3:[0-9]+]], 1, 59
; CHECK-DAG: mflr {{[0-9]+}}
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 51808
; CHECK-DAG: std 30, -16(1)
; CHECK-DAG: mr 30, 1
; CHECK-DAG: std 0, 16(1)
; CHECK-DAG: subc 0, [[REG2]], [[REG3]]
; CHECK: stdux 1, 1, 0

; CHECK: .cfi_def_cfa_register r30

; CHECK: blr

; CHECK-32-LABEL: @hoo

; CHECK-32-DAG: lis [[REG1:[0-9]+]], -13
; CHECK-32-DAG: clrlwi [[REG3:[0-9]+]], 1, 27
; CHECK-32-DAG: mflr [[LR:[0-9]+]]
; CHECK-32-DAG: ori [[REG2:[0-9]+]], [[REG1]], 51904
; CHECK-32-DAG: stw [[LR]], 4(1)
; CHECK-32-DAG: subc 0, [[REG2]], [[REG3]]
; CHECK-32:     stwux 1, 1, 0
; CHECK-32:     sub 0, 1, 0
; CHECK-32:     addic 0, 0, -4
; CHECK-32:     stwx 31, 0, 0
; CHECK-32:     addic 0, 0, -4
; CHECK-32:     stwx 30, 0, 0
; CHECK-32:     addic 30, 0, 8

; CHECK-32: blr

; CHECK-32-PIC-LABEL: @hoo

; CHECK-32-PIC-DAG: lis [[REG1:[0-9]+]], -13
; CHECK-32-PIC-DAG: clrlwi [[REG3:[0-9]+]], 1, 27
; CHECK-32-PIC-DAG: mflr {{[0-9]+}}
; CHECK-32-PIC-DAG: ori [[REG2:[0-9]+]], [[REG1]], 51904
; CHECK-32-PIC-DAG: stw 0, 4(1)
; CHECK-32-PIC-DAG: subc 0, [[REG2]], [[REG3]]
; CHECK-32-PIC:     stwux 1, 1, 0
; CHECK-32-PIC:     sub 0, 1, 0
; CHECK-32-PIC:     addic 0, 0, -4
; CHECK-32-PIC:     stwx 31, 0, 0
; CHECK-32-PIC:     addic 0, 0, -8
; CHECK-32-PIC:     stwx 29, 0, 0
; CHECK-32-PIC:     addic 29, 0, 12

; CHECK-32: blr

; Make sure that the FP save area is still allocated correctly relative to
; where r30 is saved.
define void @loo(%struct.s* byval(%struct.s) nocapture readonly %a) {
entry:
  %x = alloca [2 x i32], align 32
  %a1 = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 0
  %0 = load i32, i32* %a1, align 4
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 0
  store i32 %0, i32* %arrayidx, align 32
  %b = getelementptr inbounds %struct.s, %struct.s* %a, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [2 x i32], [2 x i32]* %x, i64 0, i64 1
  store i32 %1, i32* %arrayidx2, align 4
  call void @bar(i32* %arrayidx)
  call void asm sideeffect "", "~{f30}"() nounwind
  ret void
}

; CHECK-LABEL: @loo

; CHECK-DAG: mflr {{[0-9]+}}
; CHECK-DAG: clrldi [[REG:[0-9]+]], 1, 59
; CHECK-DAG: std 30, -32(1)
; CHECK-DAG: mr 30, 1
; CHECK-DAG: std 0, 16(1)
; CHECK-DAG: subfic 0, [[REG]], -192
; CHECK: stdux 1, 1, 0

; CHECK: .cfi_def_cfa_register r30

; CHECK: stfd 30, -16(30)

; CHECK: blr

; CHECK-FP-LABEL: @loo

; CHECK-FP-DAG: mflr {{[0-9]+}}
; CHECK-FP-DAG: clrldi [[REG:[0-9]+]], 1, 59
; CHECK-FP-DAG: std 31, -24(1)
; CHECK-FP-DAG: std 30, -32(1)
; CHECK-FP-DAG: mr 30, 1
; CHECK-FP-DAG: std 0, 16(1)
; CHECK-FP-DAG: subfic 0, [[REG]], -192
; CHECK-FP: stdux 1, 1, 0

; CHECK-FP: .cfi_def_cfa_register r30

; CHECK-FP: stfd 30, -16(30)

; CHECK-FP: blr
