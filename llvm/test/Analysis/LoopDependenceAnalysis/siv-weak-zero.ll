; RUN: opt < %s -analyze -basicaa -lda | FileCheck %s

@x = common global [256 x i32] zeroinitializer, align 4
@y = common global [256 x i32] zeroinitializer, align 4

;; for (i = 0; i < 256; i++)
;;   x[i] = x[42] + y[i]

define void @f1(...) nounwind {
entry:
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 42
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %y.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x = load i32* %x.ld.addr   ; 0
  %y = load i32* %y.addr      ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.addr  ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; for (i = 0; i < 250; i++)
;;   x[i] = x[255] + y[i]

define void @f2(...) nounwind {
entry:
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 255
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %x.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %y.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x = load i32* %x.ld.addr   ; 0
  %y = load i32* %y.addr      ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.addr  ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 250
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
