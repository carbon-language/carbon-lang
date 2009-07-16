; RUN: llvm-as < %s | llc | not grep implicit-def

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-linux"
@netsAssign = external global i64*, align 8		; <i64**> [#uses=1]
@channelColumns = external global i64, align 8		; <i64*> [#uses=1]
@TOP = external global i64*, align 8		; <i64**> [#uses=1]
@BOT = external global i64*, align 8		; <i64**> [#uses=1]
@horzPlane = external global i8*, align 8		; <i8**> [#uses=1]
@vertPlane = external global i8*, align 8		; <i8**> [#uses=1]

declare signext i32 @ExtendOK(i64, i8* nocapture, i64, i64, i64, i64) nounwind

define i1 @Maze2Mech_for_2E_cond11(i64 %bentNet, i64 %i, i64 %b2, i64 %xStart, i64 %row.0, i64 %sum263, i64 %sum262, i64 %cond.i64, i64 %cond27.i67, i64 %cond.i124, i64 %cond27.i127, i64 %conv, i64 %add, i64 %cond.i7, i64 %cond27.i10, i64 %sum267, i64 %tmp236, i64 %tmp243, i64* %col.0.out) nounwind {
newFuncRoot:
	br label %for.cond11

for.inc99.exitStub:		; preds = %for.cond11
	store i64 %col.0, i64* %col.0.out
	ret i1 true

if.then.exitStub:		; preds = %land.lhs.true53
	store i64 %col.0, i64* %col.0.out
	ret i1 false

cond.end.i.for.cond.i_crit_edge:		; preds = %cond.end.i
	br label %codeRepl

for.cond11:		; preds = %for.inc, %newFuncRoot
	%indvar237 = phi i64 [ 0, %newFuncRoot ], [ %indvar.next238, %for.inc ]		; <i64> [#uses=2]
	%colFree.0 = phi i1 [ %retval.0.i.reg2mem.1, %for.inc ], [ false, %newFuncRoot ]		; <i1> [#uses=1]
	%tmp252 = mul i64 %indvar237, %conv		; <i64> [#uses=2]
	%tmp244 = sub i64 %tmp243, %tmp252		; <i64> [#uses=2]
	%col.0 = add i64 %tmp252, %xStart		; <i64> [#uses=11]
	%tmp245 = icmp ult i64 %tmp244, %tmp236		; <i1> [#uses=1]
	%umax246 = select i1 %tmp245, i64 %tmp236, i64 %tmp244		; <i64> [#uses=1]
	%sum = add i64 %umax246, 1		; <i64> [#uses=1]
	%cmp15 = icmp eq i64 %col.0, %add		; <i1> [#uses=1]
	%or.cond = or i1 %colFree.0, %cmp15		; <i1> [#uses=1]
	br i1 %or.cond, label %for.inc99.exitStub, label %cond.end.i

cond.end.i:		; preds = %for.cond11
	%tmp18 = load i8** @horzPlane		; <i8*> [#uses=3]
	%cmp.i = icmp eq i64 %col.0, %i		; <i1> [#uses=1]
	%tmp13.i = load i64* @channelColumns		; <i64> [#uses=8]
	%mul.i = mul i64 %tmp13.i, %row.0		; <i64> [#uses=2]
	br i1 %cmp.i, label %cond.end.i.for.cond.i_crit_edge, label %if.else.i

codeRepl:		; preds = %cond.end.i.for.cond.i_crit_edge
	%targetBlock = call i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i(i8* %tmp18, i64 %tmp13.i, i64 %mul.i, i64 %i, i64 %row.0)		; <i1> [#uses=1]
	br i1 %targetBlock, label %for.cond.i.if.then.i16_crit_edge, label %for.body.i.for.inc_crit_edge

if.else.i:		; preds = %cond.end.i
	%cmp42.i = icmp ugt i64 %col.0, %i		; <i1> [#uses=2]
	%cond45.i = select i1 %cmp42.i, i64 %i, i64 %col.0		; <i64> [#uses=1]
	%cond60.i = select i1 %cmp42.i, i64 %col.0, i64 %i		; <i64> [#uses=1]
	br label %codeRepl1

codeRepl1:		; preds = %if.else.i
	%targetBlock2 = call i1 @Maze2Mech_for_2E_cond11_for_2E_cond53_2E_i(i64 %sum, i8* %tmp18, i64 %mul.i, i64 %cond45.i, i64 %cond60.i)		; <i1> [#uses=1]
	br i1 %targetBlock2, label %for.cond53.i.if.then.i16_crit_edge, label %for.body62.i.for.inc_crit_edge

if.then.i16:		; preds = %for.cond53.i.if.then.i16_crit_edge, %for.cond.i.if.then.i16_crit_edge
	%tmp26 = load i8** @vertPlane		; <i8*> [#uses=3]
	%mul.i9 = mul i64 %tmp13.i, %cond.i7		; <i64> [#uses=1]
	br label %codeRepl3

codeRepl3:		; preds = %if.then.i16
	%targetBlock4 = call i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i23(i64 %tmp13.i, i8* %tmp26, i64 %mul.i9, i64 %i, i64 %cond27.i10, i64 %sum267)		; <i1> [#uses=1]
	br i1 %targetBlock4, label %if.then.i73, label %for.body.i27.for.inc_crit_edge

if.then.i73:		; preds = %codeRepl3
	%mul.i66 = mul i64 %tmp13.i, %cond.i64		; <i64> [#uses=1]
	br label %codeRepl5

codeRepl5:		; preds = %if.then.i73
	%targetBlock6 = call i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i80(i64 %tmp13.i, i8* %tmp26, i64 %mul.i66, i64 %i, i64 %sum263, i64 %cond27.i67)		; <i1> [#uses=1]
	br i1 %targetBlock6, label %if.then.i133, label %for.body.i84.for.inc_crit_edge

if.then.i133:		; preds = %codeRepl5
	%mul.i126 = mul i64 %tmp13.i, %cond.i124		; <i64> [#uses=1]
	br label %codeRepl7

codeRepl7:		; preds = %if.then.i133
	%targetBlock8 = call i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i140(i64 %col.0, i64 %tmp13.i, i8* %tmp26, i64 %mul.i126, i64 %sum262, i64 %cond27.i127)		; <i1> [#uses=1]
	br i1 %targetBlock8, label %land.lhs.true49, label %for.body.i144.for.inc_crit_edge

land.lhs.true49:		; preds = %codeRepl7
	%tmp1.i178 = load i64** @TOP		; <i64*> [#uses=1]
	%arrayidx.i179 = getelementptr i64* %tmp1.i178, i64 %col.0		; <i64*> [#uses=1]
	%tmp2.i = load i64* %arrayidx.i179		; <i64> [#uses=3]
	%cmp.i180 = icmp eq i64 %tmp2.i, 0		; <i1> [#uses=1]
	br i1 %cmp.i180, label %land.lhs.true49.land.lhs.true53_crit_edge, label %land.lhs.true.i

land.lhs.true.i:		; preds = %land.lhs.true49
	%tmp4.i = load i64** @BOT		; <i64*> [#uses=1]
	%arrayidx5.i = getelementptr i64* %tmp4.i, i64 %col.0		; <i64*> [#uses=1]
	%tmp6.i = load i64* %arrayidx5.i		; <i64> [#uses=3]
	%cmp7.i = icmp eq i64 %tmp6.i, 0		; <i1> [#uses=1]
	%cmp17.i = icmp eq i64 %tmp2.i, %tmp6.i		; <i1> [#uses=1]
	%or.cond.i181 = or i1 %cmp7.i, %cmp17.i		; <i1> [#uses=1]
	br i1 %or.cond.i181, label %land.lhs.true.i.land.lhs.true53_crit_edge, label %HasVCV.exit

HasVCV.exit:		; preds = %land.lhs.true.i
	%tmp22.i = load i64** @netsAssign		; <i64*> [#uses=2]
	%arrayidx23.i = getelementptr i64* %tmp22.i, i64 %tmp2.i		; <i64*> [#uses=1]
	%tmp24.i = load i64* %arrayidx23.i		; <i64> [#uses=1]
	%arrayidx30.i = getelementptr i64* %tmp22.i, i64 %tmp6.i		; <i64*> [#uses=1]
	%tmp31.i182 = load i64* %arrayidx30.i		; <i64> [#uses=1]
	%phitmp189 = icmp ugt i64 %tmp24.i, %tmp31.i182		; <i1> [#uses=1]
	br i1 %phitmp189, label %HasVCV.exit.for.inc_crit_edge, label %HasVCV.exit.land.lhs.true53_crit_edge

land.lhs.true53:		; preds = %HasVCV.exit.land.lhs.true53_crit_edge, %land.lhs.true.i.land.lhs.true53_crit_edge, %land.lhs.true49.land.lhs.true53_crit_edge
	%call60 = tail call signext i32 @ExtendOK(i64 %bentNet, i8* %tmp18, i64 %col.0, i64 %b2, i64 %i, i64 %b2)		; <i32> [#uses=1]
	%tobool61 = icmp eq i32 %call60, 0		; <i1> [#uses=1]
	br i1 %tobool61, label %land.lhs.true53.for.inc_crit_edge, label %if.then.exitStub

for.inc:		; preds = %land.lhs.true53.for.inc_crit_edge, %HasVCV.exit.for.inc_crit_edge, %for.body.i144.for.inc_crit_edge, %for.body.i84.for.inc_crit_edge, %for.body.i27.for.inc_crit_edge, %for.body.i.for.inc_crit_edge, %for.body62.i.for.inc_crit_edge
	%retval.0.i.reg2mem.1 = phi i1 [ false, %HasVCV.exit.for.inc_crit_edge ], [ false, %land.lhs.true53.for.inc_crit_edge ], [ true, %for.body62.i.for.inc_crit_edge ], [ true, %for.body.i.for.inc_crit_edge ], [ false, %for.body.i27.for.inc_crit_edge ], [ false, %for.body.i84.for.inc_crit_edge ], [ false, %for.body.i144.for.inc_crit_edge ]		; <i1> [#uses=1]
	%indvar.next238 = add i64 %indvar237, 1		; <i64> [#uses=1]
	br label %for.cond11

for.body62.i.for.inc_crit_edge:		; preds = %codeRepl1
	br label %for.inc

for.cond.i.if.then.i16_crit_edge:		; preds = %codeRepl
	br label %if.then.i16

for.body.i.for.inc_crit_edge:		; preds = %codeRepl
	br label %for.inc

for.cond53.i.if.then.i16_crit_edge:		; preds = %codeRepl1
	br label %if.then.i16

for.body.i27.for.inc_crit_edge:		; preds = %codeRepl3
	br label %for.inc

for.body.i84.for.inc_crit_edge:		; preds = %codeRepl5
	br label %for.inc

for.body.i144.for.inc_crit_edge:		; preds = %codeRepl7
	br label %for.inc

HasVCV.exit.for.inc_crit_edge:		; preds = %HasVCV.exit
	br label %for.inc

land.lhs.true49.land.lhs.true53_crit_edge:		; preds = %land.lhs.true49
	br label %land.lhs.true53

land.lhs.true.i.land.lhs.true53_crit_edge:		; preds = %land.lhs.true.i
	br label %land.lhs.true53

HasVCV.exit.land.lhs.true53_crit_edge:		; preds = %HasVCV.exit
	br label %land.lhs.true53

land.lhs.true53.for.inc_crit_edge:		; preds = %land.lhs.true53
	br label %for.inc
}

declare i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i(i8*, i64, i64, i64, i64) nounwind

declare i1 @Maze2Mech_for_2E_cond11_for_2E_cond53_2E_i(i64, i8*, i64, i64, i64) nounwind

declare i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i23(i64, i8*, i64, i64, i64, i64) nounwind

declare i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i80(i64, i8*, i64, i64, i64, i64) nounwind

declare i1 @Maze2Mech_for_2E_cond11_for_2E_cond_2E_i140(i64, i64, i8*, i64, i64, i64) nounwind
