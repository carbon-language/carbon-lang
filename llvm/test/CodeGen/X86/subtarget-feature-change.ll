; RUN: llc < %s -march=x86-64 | FileCheck %s

; This should not generate SSE instructions:
;
; CHECK: without.sse:
; CHECK: flds
; CHECK: fmuls
; CHECK: fstps
define void @without.sse(float* nocapture %a, float* nocapture %b, float* nocapture %c, i32 %n) #0 {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %b, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float* %c, i64 %indvars.iv
  %1 = load float* %arrayidx2, align 4
  %mul = fmul float %0, %1
  %arrayidx4 = getelementptr inbounds float* %a, i64 %indvars.iv
  store float %mul, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; This should generate SSE instructions:
;
; CHECK: with.sse
; CHECK: movss
; CHECK: mulss
; CHECK: movss
define void @with.sse(float* nocapture %a, float* nocapture %b, float* nocapture %c, i32 %n) #1 {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %b, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float* %c, i64 %indvars.iv
  %1 = load float* %arrayidx2, align 4
  %mul = fmul float %0, %1
  %arrayidx4 = getelementptr inbounds float* %a, i64 %indvars.iv
  store float %mul, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind optsize ssp uwtable "target-cpu"="core2" "target-features"="-sse4a,-avx2,-xop,-fma4,-bmi2,-3dnow,-3dnowa,-pclmul,-sse,-avx,-sse41,-ssse3,+mmx,-rtm,-sse42,-lzcnt,-f16c,-popcnt,-bmi,-aes,-fma,-rdrand,-sse2,-sse3" }
attributes #1 = { nounwind optsize ssp uwtable "target-cpu"="core2" "target-features"="-sse4a,-avx2,-xop,-fma4,-bmi2,-3dnow,-3dnowa,-pclmul,+sse,-avx,-sse41,+ssse3,+mmx,-rtm,-sse42,-lzcnt,-f16c,-popcnt,-bmi,-aes,-fma,-rdrand,+sse2,+sse3" }
