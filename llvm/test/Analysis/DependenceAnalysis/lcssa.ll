; RUN: opt < %s -disable-output "-passes=print<da>"                            \
; RUN: "-aa-pipeline=basic-aa,tbaa" 2>&1 | FileCheck %s

; CHECK:      Src:  %v = load i32, i32* %arrayidx1, align 4 --> Dst:  store i32 %add, i32* %a.lcssa, align 4
; CHECK-NEXT: da analyze - confused!

define void @f(i32 *%a, i32 %n, i64 %n2) {
entry:
  br label %while.body

while.body:
  %n.addr = phi i32 [ %mul, %while.body ], [ %n, %entry ]
  %inc.phi = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %mul = mul i32 %n.addr, 3
  %div = udiv i32 %mul, 2
  %inc = add i32 %inc.phi, 1
  %incdec.ptr = getelementptr inbounds i32, i32* %a, i32 %inc
  %cmp.not = icmp eq i32 %div, 1
  br i1 %cmp.not, label %while.end, label %while.body

while.end:
  %a.lcssa = phi i32* [ %incdec.ptr, %while.body ]
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 1, %while.end ], [ %indvars.iv.next, %for.body ]
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %v = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %v, 1
  store i32 %add, i32* %a.lcssa, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n2
  br i1 %exitcond.not, label %ret, label %for.body

ret:
  ret void
}
