; RUN: opt < %s -disable-output -analyze -lda | FileCheck %s

@x = common global [256 x i32] zeroinitializer, align 4

;; x[5] = x[6]

define void @f1(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x = load i32* getelementptr ([256 x i32]* @x, i32 0, i64 6)
  store i32 %x, i32* getelementptr ([256 x i32]* @x, i32 0, i64 5)
; CHECK: 0,1: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; x[c] = x[c+1] // with c being a loop-invariant constant

define void @f2(i64 %c0) nounwind {
entry:
  %c1 = add i64 %c0, 1
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %c0
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %c1
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

;; x[6] = x[6]

define void @f3(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x = load i32* getelementptr ([256 x i32]* @x, i32 0, i64 6)
  store i32 %x, i32* getelementptr ([256 x i32]* @x, i32 0, i64 6)
; CHECK: 0,1: dep
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
