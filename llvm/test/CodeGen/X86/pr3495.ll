; RUN: llc < %s -march=x86 -stats |& grep {Number of reloads omited} | grep 1
; RUN: llc < %s -march=x86 -stats |& grep {Number of available reloads turned into copies} | grep 1
; RUN: llc < %s -march=x86 -stats |& grep {Number of machine instrs printed} | grep 40
; PR3495
; The loop reversal kicks in once here, resulting in one fewer instruction.

target triple = "i386-pc-linux-gnu"
@x = external global [8 x i32], align 32		; <[8 x i32]*> [#uses=1]
@rows = external global [8 x i32], align 32		; <[8 x i32]*> [#uses=2]
@up = external global [15 x i32], align 32		; <[15 x i32]*> [#uses=2]
@down = external global [15 x i32], align 32		; <[15 x i32]*> [#uses=1]

define i32 @queens(i32 %c) nounwind {
entry:
	%tmp91 = add i32 %c, 1		; <i32> [#uses=3]
	%tmp135 = getelementptr [8 x i32]* @x, i32 0, i32 %tmp91		; <i32*> [#uses=1]
	br label %bb

bb:		; preds = %bb569, %entry
	%r25.0.reg2mem.0 = phi i32 [ 0, %entry ], [ %indvar.next715, %bb569 ]		; <i32> [#uses=4]
	%tmp27 = getelementptr [8 x i32]* @rows, i32 0, i32 %r25.0.reg2mem.0		; <i32*> [#uses=1]
	%tmp28 = load i32* %tmp27, align 4		; <i32> [#uses=1]
	%tmp29 = icmp eq i32 %tmp28, 0		; <i1> [#uses=1]
	br i1 %tmp29, label %bb569, label %bb31

bb31:		; preds = %bb
	%tmp35 = sub i32 %r25.0.reg2mem.0, 0		; <i32> [#uses=1]
	%tmp36 = getelementptr [15 x i32]* @up, i32 0, i32 %tmp35		; <i32*> [#uses=1]
	%tmp37 = load i32* %tmp36, align 4		; <i32> [#uses=1]
	%tmp38 = icmp eq i32 %tmp37, 0		; <i1> [#uses=1]
	br i1 %tmp38, label %bb569, label %bb41

bb41:		; preds = %bb31
	%tmp54 = sub i32 %r25.0.reg2mem.0, %c		; <i32> [#uses=1]
	%tmp55 = add i32 %tmp54, 7		; <i32> [#uses=1]
	%tmp62 = getelementptr [15 x i32]* @up, i32 0, i32 %tmp55		; <i32*> [#uses=2]
	store i32 0, i32* %tmp62, align 4
	br label %bb92

bb92:		; preds = %bb545, %bb41
	%r20.0.reg2mem.0 = phi i32 [ 0, %bb41 ], [ %indvar.next711, %bb545 ]		; <i32> [#uses=5]
	%tmp94 = getelementptr [8 x i32]* @rows, i32 0, i32 %r20.0.reg2mem.0		; <i32*> [#uses=1]
	%tmp95 = load i32* %tmp94, align 4		; <i32> [#uses=0]
	%tmp112 = add i32 %r20.0.reg2mem.0, %tmp91		; <i32> [#uses=1]
	%tmp113 = getelementptr [15 x i32]* @down, i32 0, i32 %tmp112		; <i32*> [#uses=2]
	%tmp114 = load i32* %tmp113, align 4		; <i32> [#uses=1]
	%tmp115 = icmp eq i32 %tmp114, 0		; <i1> [#uses=1]
	br i1 %tmp115, label %bb545, label %bb118

bb118:		; preds = %bb92
	%tmp122 = sub i32 %r20.0.reg2mem.0, %tmp91		; <i32> [#uses=0]
	store i32 0, i32* %tmp113, align 4
	store i32 %r20.0.reg2mem.0, i32* %tmp135, align 4
	br label %bb142

bb142:		; preds = %bb142, %bb118
	%k18.0.reg2mem.0 = phi i32 [ 0, %bb118 ], [ %indvar.next709, %bb142 ]		; <i32> [#uses=1]
	%indvar.next709 = add i32 %k18.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond710 = icmp eq i32 %indvar.next709, 8		; <i1> [#uses=1]
	br i1 %exitcond710, label %bb155, label %bb142

bb155:		; preds = %bb142
	%tmp156 = tail call i32 @putchar(i32 10) nounwind		; <i32> [#uses=0]
	br label %bb545

bb545:		; preds = %bb155, %bb92
	%indvar.next711 = add i32 %r20.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond712 = icmp eq i32 %indvar.next711, 8		; <i1> [#uses=1]
	br i1 %exitcond712, label %bb553, label %bb92

bb553:		; preds = %bb545
	store i32 1, i32* %tmp62, align 4
	br label %bb569

bb569:		; preds = %bb553, %bb31, %bb
	%indvar.next715 = add i32 %r25.0.reg2mem.0, 1		; <i32> [#uses=1]
	br label %bb
}

declare i32 @putchar(i32)
