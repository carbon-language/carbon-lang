; RUN: llc < %s -march=xcore | FileCheck %s
; RUN: llc < %s -march=xcore -disable-fp-elim | FileCheck %s -check-prefix=CHECKFP

declare i8* @llvm.frameaddress(i32) nounwind readnone
declare i8* @llvm.returnaddress(i32) nounwind
declare i8* @llvm.eh.dwarf.cfa(i32) nounwind
declare void @llvm.eh.return.i32(i32, i8*) nounwind
declare void @llvm.eh.unwind.init() nounwind

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
; CHECK-NEXT: ldw r0, dp[offset]
; CHECK-NEXT: ldc r1, 4
; CHECK-NEXT: ldaw r2, sp[0]
; CHECK-NEXT: add r1, r2, r1
; CHECK-NEXT: add r2, r1, r0
; CHECK-NEXT: ldaw r3, dp[handler]
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void (...)* @foo()
  %0 = load i32* @offset
  call void @llvm.eh.return.i32(i32 %0, i8* @handler)
  unreachable
}


; FP: spill FP+SR+R4:9 = entsp 2 + 6
; CHECKFP-LABEL: Unwind0:
; CHECKFP: entsp 8
; CHECKFP: stw r10, sp[1]
; CHECKFP: ldaw r10, sp[0]
; CHECKFP: stw r4, r10[7]
; CHECKFP: stw r5, r10[6]
; CHECKFP: stw r6, r10[5]
; CHECKFP: stw r7, r10[4]
; CHECKFP: stw r8, r10[3]
; CHECKFP: stw r9, r10[2]
; CHECKFP: ldw r9, r10[2]
; CHECKFP: ldw r8, r10[3]
; CHECKFP: ldw r7, r10[4]
; CHECKFP: ldw r6, r10[5]
; CHECKFP: ldw r5, r10[6]
; CHECKFP: ldw r4, r10[7]
; CHECKFP: set sp, r10
; CHECKFP: ldw r10, sp[1]
; CHECKFP: retsp 8
;
; !FP: spill R4:10 = entsp 7
; CHECK-LABEL: Unwind0:
; CHECK: entsp 7
; CHECK: stw r4, sp[6]
; CHECK: stw r5, sp[5]
; CHECK: stw r6, sp[4]
; CHECK: stw r7, sp[3]
; CHECK: stw r8, sp[2]
; CHECK: stw r9, sp[1]
; CHECK: stw r10, sp[0]
; CHECK: ldw r10, sp[0]
; CHECK: ldw r9, sp[1]
; CHECK: ldw r8, sp[2]
; CHECK: ldw r7, sp[3]
; CHECK: ldw r6, sp[4]
; CHECK: ldw r5, sp[5]
; CHECK: ldw r4, sp[6]
; CHECK: retsp 7
define void @Unwind0() {
  call void @llvm.eh.unwind.init()
  ret void
}


; FP: spill FP+SR+R4:9+LR = entsp 2 + 6 + extsp 1
; CHECKFP-LABEL: Unwind1:
; CHECKFP: entsp 8
; CHECKFP: stw r10, sp[1]
; CHECKFP: ldaw r10, sp[0]
; CHECKFP: stw r4, r10[7]
; CHECKFP: stw r5, r10[6]
; CHECKFP: stw r6, r10[5]
; CHECKFP: stw r7, r10[4]
; CHECKFP: stw r8, r10[3]
; CHECKFP: stw r9, r10[2]
; CHECKFP: extsp 1
; CHECKFP: bl foo
; CHECKFP: ldaw sp, sp[1]
; CHECKFP: ldw r9, r10[2]
; CHECKFP: ldw r8, r10[3]
; CHECKFP: ldw r7, r10[4]
; CHECKFP: ldw r6, r10[5]
; CHECKFP: ldw r5, r10[6]
; CHECKFP: ldw r4, r10[7]
; CHECKFP: set sp, r10
; CHECKFP: ldw r10, sp[1]
; CHECKFP: retsp 8
;
; !FP: spill R4:10+LR = entsp 7 + 1
; CHECK-LABEL: Unwind1:
; CHECK: entsp 8
; CHECK: stw r4, sp[7]
; CHECK: stw r5, sp[6]
; CHECK: stw r6, sp[5]
; CHECK: stw r7, sp[4]
; CHECK: stw r8, sp[3]
; CHECK: stw r9, sp[2]
; CHECK: stw r10, sp[1]
; CHECK: bl foo
; CHECK: ldw r10, sp[1]
; CHECK: ldw r9, sp[2]
; CHECK: ldw r8, sp[3]
; CHECK: ldw r7, sp[4]
; CHECK: ldw r6, sp[5]
; CHECK: ldw r5, sp[6]
; CHECK: ldw r4, sp[7]
; CHECK: retsp 8
define void @Unwind1() {
  call void (...)* @foo()
  call void @llvm.eh.unwind.init()
  ret void
}
