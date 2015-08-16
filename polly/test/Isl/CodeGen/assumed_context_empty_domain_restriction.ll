; RUN: opt %loadPolly -S -polly-opt-isl -polly-codegen < %s
;
; TODO: This test will crash the scalar code generation. The problem step by step:
;         1) The assumed context is empty because of the out-of-bounds array access.
;         2) The dependence analysis will use the assumed context and determine
;            that there are not iterations executed, hence no dependences.
;         3) The scheduler will transform the program somehow according to
;            the orginal domains and empty dependences but not the assumed context.
;            The new ast is shown below.
;         4) The code generation will look for the new value of %0 when the
;            for_body_328_lr_ph statement is copied, however it has not yet seen %0
;            as the for_end_310 statement has been moved after for_body_328_lr_ph.
;         5) Crash as no new value of %0 can be found.
;
; AST:
;
;   if (0)
;       {
;         for (int c0 = 0; c0 <= 32; c0 += 1)
;           Stmt_for_body_328(c0);
;         Stmt_for_body_328_lr_ph();
;         Stmt_for_end_310();
;       }
;   else
;       {  /* original code */ }
;
; CHECK: polly.start
;
; TODO: This test should not crash Polly.
;
; XFAIL: *
;
@endposition = external global i32, align 4
@Bit = external global [0 x i32], align 4
@Init = external global [0 x i32], align 4

define void @maskgen() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.end.310, label %for.body

for.end.310:                                      ; preds = %for.body
  store i32 undef, i32* @endposition, align 4
  %sub325 = sub i32 33, 0
  %0 = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @Init, i64 0, i64 0), align 4
  br i1 false, label %for.cond.347.preheader, label %for.body.328.lr.ph

for.body.328.lr.ph:                               ; preds = %for.end.310
  %1 = sub i32 34, 0
  br label %for.body.328

for.body.328:                                     ; preds = %for.body.328, %for.body.328.lr.ph
  %indvars.iv546 = phi i64 [ %indvars.iv.next547, %for.body.328 ], [ 1, %for.body.328.lr.ph ]
  %2 = phi i32 [ %or331, %for.body.328 ], [ %0, %for.body.328.lr.ph ]
  %arrayidx330 = getelementptr inbounds [0 x i32], [0 x i32]* @Bit, i64 0, i64 %indvars.iv546
  %3 = load i32, i32* %arrayidx330, align 4
  %or331 = or i32 %3, %2
  %indvars.iv.next547 = add nuw nsw i64 %indvars.iv546, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next547 to i32
  %exitcond14 = icmp eq i32 %lftr.wideiv, %1
  br i1 %exitcond14, label %for.cond.347.preheader, label %for.body.328

for.cond.347.preheader:                           ; preds = %for.cond.347.preheader, %for.body.328, %for.end.310
  br i1 undef, label %if.end.471, label %for.cond.347.preheader

if.end.471:                                       ; preds = %for.cond.347.preheader
  ret void
}
