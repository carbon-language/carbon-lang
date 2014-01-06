; RUN: llc < %s -march=xcore | FileCheck %s

declare i8* @llvm.frameaddress(i32) nounwind readnone
define i8* @FA0() nounwind {
entry:
; CHECK-LABEL: FA0
; CHECK: ldaw r0, sp[0]
; CHECK-NEXT: retsp 0
  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @FA1() nounwind {
entry:
; CHECK-LABEL: FA1
; CHECK: entsp 100
; CHECK-NEXT: ldaw r0, sp[0]
; CHECK-NEXT: retsp 100
  %0 = alloca [100 x i32]
  %1 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %1
}


declare i8* @llvm.returnaddress(i32) nounwind readnone
define i8* @RA0() nounwind {
entry:
; CHECK-LABEL: RA0
; CHECK: stw lr, sp[0]
; CHECK-NEXT: ldw r0, sp[0]
; CHECK-NEXT: ldw lr, sp[0]
; CHECK-NEXT: retsp 0
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @RA1() nounwind {
entry:
; CHECK-LABEL: RA1
; CHECK: entsp 100
; CHECK-NEXT: ldw r0, sp[100]
; CHECK-NEXT: retsp 100
  %0 = alloca [100 x i32]
  %1 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %1
}

; test FRAME_TO_ARGS_OFFSET lowering
declare i8* @llvm.eh.dwarf.cfa(i32) nounwind
define i8* @FTAO0() nounwind {
entry:
; CHECK-LABEL: FTAO0
; CHECK: ldc r0, 0
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: add r0, r1, r0
; CHECK-NEXT: retsp 0
  %0 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  ret i8* %0
}

define i8* @FTAO1() nounwind {
entry:
; CHECK-LABEL: FTAO1
; CHECK: entsp 100
; CHECK-NEXT: ldc r0, 400
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: add r0, r1, r0
; CHECK-NEXT: retsp 100
  %0 = alloca [100 x i32]
  %1 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  ret i8* %1
}
