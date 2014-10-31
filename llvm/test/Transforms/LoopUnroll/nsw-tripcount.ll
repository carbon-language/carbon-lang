; RUN: opt -loop-unroll -S %s | FileCheck %s

; extern void f(int);
; void test1(int v) {
;   for (int i=v; i<=v+1; ++i)
;     f(i);
; }
;
; We can use the nsw information to see that the tripcount will be 2, so the
; loop should be unrolled as this is always beneficial

declare void @f(i32)

; CHECK-LABEL: @test1
define void @test1(i32 %v) {
entry:
  %add = add nsw i32 %v, 1
  br label %for.body

for.body:
  %i.04 = phi i32 [ %v, %entry ], [ %inc, %for.body ]
  tail call void @f(i32 %i.04)
  %inc = add nsw i32 %i.04, 1
  %cmp = icmp slt i32 %i.04, %add
  br i1 %cmp, label %for.body, label %for.end

; CHECK: call void @f
; CHECK-NOT: br i1
; CHECK: call void @f
for.end:
  ret void
}
