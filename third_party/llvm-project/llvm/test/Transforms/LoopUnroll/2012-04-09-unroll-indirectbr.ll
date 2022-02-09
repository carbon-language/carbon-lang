; RUN: opt < %s -S -loop-unroll -simplifycfg -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s
; PR12513: Loop unrolling breaks with indirect branches.
; If loop unrolling attempts to transform this loop, it replaces the
; indirectbr successors. SimplifyCFG then considers them to be unreachable.
declare void @subtract() nounwind uwtable

; CHECK-NOT: unreachable
define i32 @main(i32 %argc, i8** nocapture %argv) nounwind uwtable {
entry:
  %vals19 = alloca [5 x i32], align 16
  %x20 = alloca i32, align 4
  store i32 135, i32* %x20, align 4
  br label %for.body

for.body:                                         ; preds = ; %call2_termjoin, %call3_termjoin
  %indvars.iv = phi i64 [ 0, %entry ], [ %joinphi15.in.in, %call2_termjoin ]
  %a6 = call coldcc i8* @funca(i8* blockaddress(@main, %for.body_code), i8*
blockaddress(@main, %for.body_codeprime)) nounwind
  indirectbr i8* %a6, [label %for.body_code, label %for.body_codeprime]

for.body_code:                                    ; preds = %for.body
  call void @subtract()
  br label %call2_termjoin

call2_termjoin:                                   ; preds = %for.body_codeprime, %for.body_code
  %joinphi15.in.in = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %joinphi15.in.in, 5
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %call2_termjoin
  ret i32 0

for.body_codeprime:                               ; preds = %for.body
  call void @subtract_v2(i64 %indvars.iv)
  br label %call2_termjoin
}

declare coldcc i8* @funca(i8*, i8*) readonly

declare void @subtract_v2(i64) nounwind uwtable
