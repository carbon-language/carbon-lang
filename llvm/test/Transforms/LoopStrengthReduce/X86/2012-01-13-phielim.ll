; RUN: llc < %s -O3 -march=x86-64 -mcpu=core2 | FileCheck %s

declare i1 @check() nounwind
declare i1 @foo(i8*, i8*, i8*) nounwind

; Check that redundant phi elimination ran
; CHECK: @test
; CHECK: %while.body.i
; CHECK: movs
; CHECK-NOT: movs
; CHECK: %for.end.i
define i32 @test(i8* %base) nounwind uwtable ssp {
entry:
  br label %while.body.lr.ph.i

while.body.lr.ph.i:                               ; preds = %cond.true.i
  br label %while.body.i

while.body.i:                                     ; preds = %cond.true29.i, %while.body.lr.ph.i
  %indvars.iv7.i = phi i64 [ 16, %while.body.lr.ph.i ], [ %indvars.iv.next8.i, %cond.true29.i ]
  %i.05.i = phi i64 [ 0, %while.body.lr.ph.i ], [ %indvars.iv7.i, %cond.true29.i ]
  %sext.i = shl i64 %i.05.i, 32
  %idx.ext.i = ashr exact i64 %sext.i, 32
  %add.ptr.sum.i = add i64 %idx.ext.i, 16
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %while.body.i
  %indvars.iv.i = phi i64 [ 0, %while.body.i ], [ %indvars.iv.next.i, %for.body.i ]
  %add.ptr.sum = add i64 %add.ptr.sum.i, %indvars.iv.i
  %arrayidx22.i = getelementptr inbounds i8* %base, i64 %add.ptr.sum
  %0 = load i8* %arrayidx22.i, align 1
  %indvars.iv.next.i = add i64 %indvars.iv.i, 1
  %cmp = call i1 @check() nounwind
  br i1 %cmp, label %for.end.i, label %for.body.i

for.end.i:                                        ; preds = %for.body.i
  %add.ptr.i144 = getelementptr inbounds i8* %base, i64 %add.ptr.sum.i
  %cmp2 = tail call i1 @foo(i8* %add.ptr.i144, i8* %add.ptr.i144, i8* undef) nounwind
  br i1 %cmp2, label %cond.true29.i, label %cond.false35.i

cond.true29.i:                                    ; preds = %for.end.i
  %indvars.iv.next8.i = add i64 %indvars.iv7.i, 16
  br i1 false, label %exit, label %while.body.i

cond.false35.i:                                   ; preds = %for.end.i
  unreachable

exit:                                 ; preds = %cond.true29.i, %cond.true.i
  ret i32 0
}

%struct.anon.7.91.199.307.415.475.559.643.751.835.943.1003.1111.1219.1351.1375.1399.1435.1471.1483.1519.1531.1651.1771 = type { i32, i32, i32 }

@tags = external global [5000 x %struct.anon.7.91.199.307.415.475.559.643.751.835.943.1003.1111.1219.1351.1375.1399.1435.1471.1483.1519.1531.1651.1771], align 16

; PR11782: SCEVExpander assert
;
; Test phi reuse after LSR that requires SCEVExpander to hoist an
; interesting GEP.
;
; CHECK: @test2
; CHECK: %entry
; CHECK-NOT: mov
; CHECK: je
define void @test2(i32 %n) nounwind uwtable {
entry:
  br i1 undef, label %while.end, label %for.cond468

for.cond468:                                      ; preds = %if.then477, %entry
  %indvars.iv1163 = phi i64 [ %indvars.iv.next1164, %if.then477 ], [ 1, %entry ]
  %k.0.in = phi i32* [ %last, %if.then477 ], [ getelementptr inbounds ([5000 x %struct.anon.7.91.199.307.415.475.559.643.751.835.943.1003.1111.1219.1351.1375.1399.1435.1471.1483.1519.1531.1651.1771]* @tags, i64 0, i64 0, i32 2), %entry ]
  %k.0 = load i32* %k.0.in, align 4
  %0 = trunc i64 %indvars.iv1163 to i32
  %cmp469 = icmp slt i32 %0, %n
  br i1 %cmp469, label %for.body471, label %for.inc498

for.body471:                                      ; preds = %for.cond468
  %first = getelementptr inbounds [5000 x %struct.anon.7.91.199.307.415.475.559.643.751.835.943.1003.1111.1219.1351.1375.1399.1435.1471.1483.1519.1531.1651.1771]* @tags, i64 0, i64 %indvars.iv1163, i32 1
  %1 = load i32* %first, align 4
  br i1 undef, label %if.then477, label %for.inc498

if.then477:                                       ; preds = %for.body471
  %last = getelementptr inbounds [5000 x %struct.anon.7.91.199.307.415.475.559.643.751.835.943.1003.1111.1219.1351.1375.1399.1435.1471.1483.1519.1531.1651.1771]* @tags, i64 0, i64 %indvars.iv1163, i32 2
  %indvars.iv.next1164 = add i64 %indvars.iv1163, 1
  br label %for.cond468

for.inc498:                                       ; preds = %for.inc498, %for.body471, %for.cond468
  br label %for.inc498

while.end:                                        ; preds = %entry
  ret void
}
