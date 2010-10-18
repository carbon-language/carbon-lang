; RUN: opt < %s -analyze -basicaa -lda | FileCheck %s

@x = common global [256 x i32] zeroinitializer, align 4
@y = common global [256 x i32] zeroinitializer, align 4

;; for (i = 0; i < 256; i++)
;;   x[i] = x[i] + y[i]

define void @f1(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %y.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %x = load i32* %x.addr      ; 0
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

;; for (i = 0; i < 256; i++)
;;   x[i+1] = x[i] + y[i]

define void @f2(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %i.next = add i64 %i, 1
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.next
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; for (i = 0; i < 10; i++)
;;   x[i+20] = x[i] + y[i]

define void @f3(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %i.20 = add i64 %i, 20
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.20
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; for (i = 0; i < 10; i++)
;;   x[10*i+1] = x[10*i] + y[i]

define void @f4(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %i.10 = mul i64 %i, 10
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i.10
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.10
  %i.10.1 = add i64 %i.10, 1
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.10.1
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
