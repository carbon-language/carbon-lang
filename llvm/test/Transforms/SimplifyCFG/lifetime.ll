; RUN: opt < %s -simplifycfg -S | FileCheck %s

; Test that a lifetime intrinsic isn't removed because that would change semantics
; This case is that predecessor(s) of the target empty block (bb0) has multiple
; successors (bb0 and bb1) and its successor has multiple predecessors (entry and
; bb0).

; CHECK: foo
; CHECK: entry:
; CHECK: bb0:
; CHECK: bb1:
; CHECK: ret
define void @foo(i1 %x) {
entry:
  %a = alloca i8
  call void @llvm.lifetime.start(i64 -1, i8* %a) nounwind
  br i1 %x, label %bb0, label %bb1

bb0:
  call void @llvm.lifetime.end(i64 -1, i8* %a) nounwind
  br label %bb1

bb1:
  call void @f()
  ret void
}

declare void @f()

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

; Test that empty block including lifetime intrinsic and not related bitcast
; instruction cannot be removed. It is because the block is not empty.
 
; CHECK-LABEL: coo
; CHECK-LABEL: entry:
; CHECK-LABEL: if.then:
; CHECK-LABEL: if.else:
; CHECK-LABEL: if.end:
; CHECK-LABEL: bb:
; CHECK: ret

define void @coo(i1 %x, i1 %y) {
entry:
  %a = alloca i8, align 4
  %b = alloca i32, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  br i1 %y, label %while.body, label %bb

while.body:                                       ; preds = %while.cond
  call void @llvm.lifetime.start(i64 4, i8* %a)
  %c = load i8, i8* %a, align 4
  br i1 %x, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %d = add i8 %c, 1
  br label %if.end

if.else:                                          ; preds = %while.body
  %e = sub i8 %c, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %f = bitcast i32* %b to i8*
  call void @llvm.lifetime.end(i64 4, i8* %a)
  br label %while.cond

bb:                                               ; preds = %while.cond
  ret void
}

; Test that empty block including lifetime intrinsic can be removed.
; Lifetime.end intrinsic is moved to predecessors because successor has 
; multiple predecessors.

; CHECK-LABEL: soo
; CHECK-LABEL: entry:
; CHECK-LABEL: if.then:
; CHECK-NEXT: %e
; CHECK-NEXT: call void @llvm.lifetime.end
; CHECK-LABEL: if.else:
; CHECK-NEXT: %g
; CHECK-NEXT: call void @llvm.lifetime.end
; CHECK-NEXT: br label %while.cond
; CHECK-NOT: if.end:
; CHECK: ret

define void @soo(i1 %x, i1 %y) {
entry:
  %a = alloca i8, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  br i1 %y, label %while.body, label %bb

while.body:                                       ; preds = %while.cond
  call void @llvm.lifetime.start(i64 4, i8* %a)
  %d = load i8, i8* %a, align 4
  br i1 %x, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %e = add i8 %d, 1
  br label %if.end

if.else:                                          ; preds = %while.body
  %g = sub i8 %d, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.lifetime.end(i64 4, i8* %a)
  br label %while.cond

bb:                                               ; preds = %while.cond
  ret void
}

; Test that empty block including lifetime intrinsic and related bitcast
; instruction can be removed. Lifetime.end intrinsic and related bitcast 
; instruction are moved to predecessors because successor has multiple
; predecessors.

; CHECK-LABEL: boo
; CHECK-LABEL: entry:
; CHECK-LABEL: if.then:
; CHECK-NEXT: %e
; CHECK-NEXT: %[[T:[^ ]+]] = bitcast
; CHECK-NEXT: call void @llvm.lifetime.end(i64 4, i8* %[[T]])
; CHECK-LABEL: if.else:
; CHECK-NEXT: %g
; CHECK-NEXT: %[[B:[^ ]+]] = bitcast
; CHECK-NEXT: call void @llvm.lifetime.end(i64 4, i8* %[[B]])
; CHECK-NEXT: br label %while.cond
; CHECK-NOT: if.end:
; CHECK: ret

define void @boo(i1 %x, i1 %y) {
entry:
  %a = alloca i32, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  br i1 %y, label %while.body, label %bb

while.body:                                       ; preds = %while.cond
  %b = bitcast i32* %a to i8*
  call void @llvm.lifetime.start(i64 4, i8* %b)
  %d = load i32, i32* %a, align 4
  br i1 %x, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %e = add i32 %d, 1
  br label %if.end

if.else:                                          ; preds = %while.body
  %g = sub i32 %d, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %c = bitcast i32* %a to i8*
  call void @llvm.lifetime.end(i64 4, i8* %c)
  br label %while.cond

bb:                                               ; preds = %while.cond
  ret void
}

; Test that empty block including lifetime intrinsic can be removed.
; Lifetime.start intrinsic is moved to predecessors because successor has 
; multiple predecessors.

; CHECK-LABEL: koo
; CHECK-LABEL: entry:
; CHECK-LABEL: if.then:
; CHECK-NEXT: call void @f
; CHECK-NEXT: call void @llvm.lifetime.start
; CHECK-LABEL: if.else:
; CHECK-NEXT: call void @g
; CHECK-NEXT: call void @llvm.lifetime.start
; CHECK-NEXT: br label %bb
; CHECK-NOT: if.end:
; CHECK: ret

define void @koo(i1 %x, i1 %y, i1 %z) {
entry:
  %a = alloca i8, align 4
  br i1 %z, label %bb, label %bb0

bb0:                                              ; preds = %entry
  br i1 %x, label %if.then, label %if.else

if.then:                                          ; preds = %bb0
  call void @f()
  br label %if.end

if.else:                                          ; preds = %bb0
  call void @g()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.lifetime.start(i64 4, i8* %a)
  br label %bb

bb:                                               ; preds = %if.end, %entry
  %d = load i8, i8* %a, align 4
  call void @llvm.lifetime.end(i64 4, i8* %a)
  ret void
}

declare void @g()

; Test that empty block including lifetime intrinsic and related bitcast
; instruction can be removed. Lifetime.start intrinsic and related bitcast 
; instruction are moved to predecessors because successor has multiple
; predecessors.


; CHECK-LABEL: goo
; CHECK-LABEL: entry:
; CHECK-LABEL: if.then:
; CHECK-NEXT: call void @f
; CHECK-NEXT: %[[T:[^ ]+]] = bitcast
; CHECK-NEXT: call void @llvm.lifetime.start(i64 4, i8* %[[T]])
; CHECK-LABEL: if.else:
; CHECK-NEXT: call void @g
; CHECK-NEXT: %[[B:[^ ]+]] = bitcast
; CHECK-NEXT: call void @llvm.lifetime.start(i64 4, i8* %[[B]])
; CHECK-NEXT: br label %bb
; CHECK-NOT: if.end:
; CHECK: ret

define void @goo(i1 %x, i1 %y, i1 %z) {
entry:
  %a = alloca i32, align 4
  br i1 %z, label %bb, label %bb0

bb0:                                              ; preds = %entry
  br i1 %x, label %if.then, label %if.else

if.then:                                          ; preds = %bb0
  call void @f()
  br label %if.end

if.else:                                          ; preds = %bb0
  call void @g()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %b = bitcast i32* %a to i8*
  call void @llvm.lifetime.start(i64 4, i8* %b)
  br label %bb

bb:                                               ; preds = %if.end, %entry
  %d = load i32, i32* %a, align 4
  %c = bitcast i32* %a to i8*
  call void @llvm.lifetime.end(i64 4, i8* %c)
  ret void
}
