; RUN: opt < %s -analyze -basicaa -lda | FileCheck %s

;; x[5] = x[6] // with x being a pointer passed as argument

define void @f1(i32* nocapture %xptr) nounwind {
entry:
  %x.ld.addr = getelementptr i32* %xptr, i64 6
  %x.st.addr = getelementptr i32* %xptr, i64 5
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x = load i32* %x.ld.addr
  store i32 %x, i32* %x.st.addr
; CHECK: 0,1: dep
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; x[5] = x[6] // with x being an array on the stack

define void @foo(...) nounwind {
entry:
  %xptr = alloca [256 x i32], align 4
  %x.ld.addr = getelementptr [256 x i32]* %xptr, i64 0, i64 6
  %x.st.addr = getelementptr [256 x i32]* %xptr, i64 0, i64 5
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x = load i32* %x.ld.addr
  store i32 %x, i32* %x.st.addr
; CHECK: 0,1: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
