; RUN: opt < %s -disable-output -analyze -lda | FileCheck %s

@x = common global [256 x i32] zeroinitializer, align 4
@y = common global [256 x i32] zeroinitializer, align 4

;; for (i = 0; i < 256; i++)
;;   x[i] = x[255 - i] + y[i]

define void @f1(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %i.255 = sub i64 255, %i
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.255
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; for (i = 0; i < 100; i++)
;;   x[i] = x[255 - i] + y[i]

define void @f2(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %i.255 = sub i64 255, %i
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.255
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; // the first iteration (i=0) leads to an out-of-bounds access of x. as the
;; // result of this access is undefined, _any_ dependence result is safe.
;; for (i = 0; i < 256; i++)
;;   x[i] = x[256 - i] + y[i]

define void @f3(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %i.256 = sub i64 0, %i
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 1, i64 %i.256
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2:
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

;; // slightly contrived but valid IR for the following loop, where all
;; // accesses in all iterations are within bounds. while this example's first
;; // (ZIV-)subscript is (0, 1), accesses are dependent.
;; for (i = 1; i < 256; i++)
;;   x[i] = x[256 - i] + y[i]

define void @f4(...) nounwind {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %i.1 = add i64 1, %i
  %i.256 = sub i64 -1, %i
  %y.ld.addr = getelementptr [256 x i32]* @y, i64 0, i64 %i.1
  %x.ld.addr = getelementptr [256 x i32]* @x, i64 1, i64 %i.256
  %x.st.addr = getelementptr [256 x i32]* @x, i64 0, i64 %i.1
  %x = load i32* %x.ld.addr     ; 0
  %y = load i32* %y.ld.addr     ; 1
  %r = add i32 %y, %x
  store i32 %r, i32* %x.st.addr ; 2
; CHECK: 0,2: dep
; CHECK: 1,2: ind
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
