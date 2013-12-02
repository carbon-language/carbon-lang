; RUN: llc < %s -march=xcore | FileCheck %s
; RUN: llc < %s -march=xcore -disable-fp-elim | FileCheck %s -check-prefix=CHECKFP


; CHECKFP-LABEL: f1
; CHECKFP: entsp 2
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 2
;
; CHECK-LABEL: f1
; CHECK: stw lr, sp[0]
; CHECK: ldw lr, sp[0]
; CHECK-NEXT: retsp 0
define void @f1() nounwind {
entry:
  tail call void asm sideeffect "", "~{lr}"() nounwind
  ret void
}


; CHECKFP-LABEL:f3
; CHECKFP: entsp 3
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: stw [[REG:r[4-9]+]], r10[2]
; CHECKFP-NEXT: mov [[REG]], r0
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: bl f2
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: mov r0, [[REG]]
; CHECKFP-NEXT: ldw [[REG]], r10[2]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 3
;
; CHECK-LABEL: f3
; CHECK: entsp 2
; CHECK: stw [[REG:r[4-9]+]], sp[1]
; CHECK: mov [[REG]], r0
; CHECK: bl f2
; CHECK: mov r0, [[REG]]
; CHECK: ldw [[REG]], sp[1]
; CHECK: retsp 2
declare void @f2()
define i32 @f3(i32 %i) nounwind {
entry:
  call void @f2()
  ret i32 %i
}


; CHECKFP-LABEL: f4
; CHECKFP: extsp 65535
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_offset 262140
; CHECKFP-NEXT: extsp 34467
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_offset 400008
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_offset 10, -400004
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_register 10
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: ldaw sp, sp[34467]
; CHECKFP-NEXT: retsp 0
;
; CHECK-LABEL: f4
; CHECK: extsp 65535
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_def_cfa_offset 262140
; CHECK-NEXT: extsp 34465
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_def_cfa_offset 400000
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: ldaw sp, sp[34465]
; CHECK-NEXT: retsp 0
define void @f4() {
entry:
  %0 = alloca [100000 x i32], align 4
  ret void
}


; CHECKFP-LABEL: f6
; CHECKFP: entsp 65535
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_offset 262140
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_offset 15, 0
; CHECKFP-NEXT: extsp 65535
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_offset 524280
; CHECKFP-NEXT: extsp 65535
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_offset 786420
; CHECKFP-NEXT: extsp 3396
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_offset 800004
; CHECKFP-NEXT: stw r10, sp[1]
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_offset 10, -800000
; CHECKFP-NEXT: ldaw r10, sp[0]
; CHECKFP-NEXT: .Ltmp{{[0-9]+}}
; CHECKFP-NEXT: .cfi_def_cfa_register 10
; CHECKFP-NEXT: extsp 1
; CHECKFP-NEXT: ldaw r0, r10[2]
; CHECKFP-NEXT: bl f5
; CHECKFP-NEXT: ldaw sp, sp[1]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: ldaw sp, sp[65535]
; CHECKFP-NEXT: retsp 3396
;
; CHECK-LABEL: f6
; CHECK: entsp 65535
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_def_cfa_offset 262140
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_offset 15, 0
; CHECK-NEXT: extsp 65535
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_def_cfa_offset 524280
; CHECK-NEXT: extsp 65535
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_def_cfa_offset 786420
; CHECK-NEXT: extsp 3395
; CHECK-NEXT: .Ltmp{{[0-9]+}}
; CHECK-NEXT: .cfi_def_cfa_offset 800000
; CHECK-NEXT: ldaw r0, sp[1]
; CHECK-NEXT: bl f5
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: ldaw sp, sp[65535]
; CHECK-NEXT: retsp 3395
declare void @f5(i32*)
define void @f6() {
entry:
  %0 = alloca [199999 x i32], align 4
  %1 = getelementptr inbounds [199999 x i32]* %0, i32 0, i32 0
  call void @f5(i32* %1)
  ret void
}
