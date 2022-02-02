; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s

	%struct.anon = type { i32 (i32, i32, i32)*, i32, i32, [3 x i32], i8*, i8*, i8* }
@r = external global [14 x i32]		; <[14 x i32]*> [#uses=4]
@isa = external global [13 x %struct.anon]		; <[13 x %struct.anon]*> [#uses=1]
@pgm = external global [2 x { i32, [3 x i32] }]		; <[2 x { i32, [3 x i32] }]*> [#uses=4]
@numi = external global i32		; <i32*> [#uses=1]
@counter = external global [2 x i32]		; <[2 x i32]*> [#uses=1]

; CHECK-LABEL: main_bb_2E_i_bb205_2E_i_2E_i_bb115_2E_i_2E_i:
; CHECK-NOT: bx lr

define void @main_bb_2E_i_bb205_2E_i_2E_i_bb115_2E_i_2E_i() {
newFuncRoot:
	br label %bb115.i.i

bb115.i.i.bb170.i.i_crit_edge.exitStub:		; preds = %bb115.i.i
	ret void

bb115.i.i.bb115.i.i_crit_edge:		; preds = %bb115.i.i
	br label %bb115.i.i

bb115.i.i:		; preds = %bb115.i.i.bb115.i.i_crit_edge, %newFuncRoot
	%i_addr.3210.0.i.i = phi i32 [ %tmp166.i.i, %bb115.i.i.bb115.i.i_crit_edge ], [ 0, %newFuncRoot ]		; <i32> [#uses=7]
	%tmp124.i.i = getelementptr [2 x { i32, [3 x i32] }], [2 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %i_addr.3210.0.i.i, i32 1, i32 0		; <i32*> [#uses=1]
	%tmp125.i.i = load i32, i32* %tmp124.i.i		; <i32> [#uses=1]
	%tmp126.i.i = getelementptr [14 x i32], [14 x i32]* @r, i32 0, i32 %tmp125.i.i		; <i32*> [#uses=1]
	%tmp127.i.i = load i32, i32* %tmp126.i.i		; <i32> [#uses=1]
	%tmp131.i.i = getelementptr [2 x { i32, [3 x i32] }], [2 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %i_addr.3210.0.i.i, i32 1, i32 1		; <i32*> [#uses=1]
	%tmp132.i.i = load i32, i32* %tmp131.i.i		; <i32> [#uses=1]
	%tmp133.i.i = getelementptr [14 x i32], [14 x i32]* @r, i32 0, i32 %tmp132.i.i		; <i32*> [#uses=1]
	%tmp134.i.i = load i32, i32* %tmp133.i.i		; <i32> [#uses=1]
	%tmp138.i.i = getelementptr [2 x { i32, [3 x i32] }], [2 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %i_addr.3210.0.i.i, i32 1, i32 2		; <i32*> [#uses=1]
	%tmp139.i.i = load i32, i32* %tmp138.i.i		; <i32> [#uses=1]
	%tmp140.i.i = getelementptr [14 x i32], [14 x i32]* @r, i32 0, i32 %tmp139.i.i		; <i32*> [#uses=1]
	%tmp141.i.i = load i32, i32* %tmp140.i.i		; <i32> [#uses=1]
	%tmp143.i.i = add i32 %i_addr.3210.0.i.i, 12		; <i32> [#uses=1]
	%tmp146.i.i = getelementptr [2 x { i32, [3 x i32] }], [2 x { i32, [3 x i32] }]* @pgm, i32 0, i32 %i_addr.3210.0.i.i, i32 0		; <i32*> [#uses=1]
	%tmp147.i.i = load i32, i32* %tmp146.i.i		; <i32> [#uses=1]
	%tmp149.i.i = getelementptr [13 x %struct.anon], [13 x %struct.anon]* @isa, i32 0, i32 %tmp147.i.i, i32 0		; <i32 (i32, i32, i32)**> [#uses=1]
	%tmp150.i.i = load i32 (i32, i32, i32)*, i32 (i32, i32, i32)** %tmp149.i.i		; <i32 (i32, i32, i32)*> [#uses=1]
	%tmp154.i.i = tail call i32 %tmp150.i.i( i32 %tmp127.i.i, i32 %tmp134.i.i, i32 %tmp141.i.i )		; <i32> [#uses=1]
	%tmp155.i.i = getelementptr [14 x i32], [14 x i32]* @r, i32 0, i32 %tmp143.i.i		; <i32*> [#uses=1]
	store i32 %tmp154.i.i, i32* %tmp155.i.i
	%tmp159.i.i = getelementptr [2 x i32], [2 x i32]* @counter, i32 0, i32 %i_addr.3210.0.i.i		; <i32*> [#uses=2]
	%tmp160.i.i = load i32, i32* %tmp159.i.i		; <i32> [#uses=1]
	%tmp161.i.i = add i32 %tmp160.i.i, 1		; <i32> [#uses=1]
	store i32 %tmp161.i.i, i32* %tmp159.i.i
	%tmp166.i.i = add i32 %i_addr.3210.0.i.i, 1		; <i32> [#uses=2]
	%tmp168.i.i = load i32, i32* @numi		; <i32> [#uses=1]
	icmp slt i32 %tmp166.i.i, %tmp168.i.i		; <i1>:0 [#uses=1]
	br i1 %0, label %bb115.i.i.bb115.i.i_crit_edge, label %bb115.i.i.bb170.i.i_crit_edge.exitStub
}

define void @PR15520(void ()* %fn) {
  call void %fn()
  ret void

; CHECK-LABEL: PR15520:
; CHECK: mov lr, pc
; CHECK: mov pc, r0
}
