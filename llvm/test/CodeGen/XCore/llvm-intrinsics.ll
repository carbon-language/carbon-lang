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

declare void @llvm.eh.return.i32(i32, i8*)
define i8* @EH0(i32 %offset, i8* %handler) {
entry:
; CHECK-LABEL: EH0
; CHECK: ldc r2, 0
; CHECK-NEXT: ldaw r3, sp[0]
; CHECK-NEXT: add r2, r3, r2
; CHECK-NEXT: add r2, r2, r0
; CHECK-NEXT: mov r3, r1
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void @llvm.eh.return.i32(i32 %offset, i8* %handler)
  unreachable
}

declare void @foo(...)
define i8* @EH1(i32 %offset, i8* %handler) {
entry:
; CHECK-LABEL: EH1
; CHECK: entsp 3
; CHECK: stw r4, sp[2]
; CHECK: stw r5, sp[1]
; CHECK: mov r4, r1
; CHECK-NEXT: mov r5, r0
; CHECK-NEXT: bl foo
; CHECK-NEXT: ldc r0, 12
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: add r0, r1, r0
; CHECK-NEXT: add r2, r0, r5
; CHECK-NEXT: mov r3, r4
; CHECK-NEXT: ldw r5, sp[1]
; CHECK-NEXT: ldw r4, sp[2]
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void (...)* @foo()
  call void @llvm.eh.return.i32(i32 %offset, i8* %handler)
  unreachable
}

@offset = external constant i32
@handler = external constant i8
define i8* @EH2(i32 %r0, i32 %r1, i32 %r2, i32 %r3) {
entry:
; CHECK-LABEL: EH2
; CHECK: entsp 1
; CHECK: bl foo
; CHECK-NEXT: ldw r0, cp[offset]
; CHECK-NEXT: ldc r1, 4
; CHECK-NEXT: ldaw r2, sp[0]
; CHECK-NEXT: add r1, r2, r1
; CHECK-NEXT: add r2, r1, r0
; CHECK-NEXT: ldaw r11, cp[handler]
; CHECK-NEXT: mov r3, r11
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void (...)* @foo()
  %0 = load i32* @offset
  call void @llvm.eh.return.i32(i32 %0, i8* @handler)
  unreachable
}
