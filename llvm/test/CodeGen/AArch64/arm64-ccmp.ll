; RUN: llc < %s -mcpu=cyclone -verify-machineinstrs -aarch64-ccmp -aarch64-stress-ccmp | FileCheck %s
target triple = "arm64-apple-ios"

; CHECK: single_same
; CHECK: cmp w0, #5
; CHECK-NEXT: ccmp w1, #17, #4, ne
; CHECK-NEXT: b.ne
; CHECK: %if.then
; CHECK: bl _foo
; CHECK: %if.end
define i32 @single_same(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 5
  %cmp1 = icmp eq i32 %b, 17
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Different condition codes for the two compares.
; CHECK: single_different
; CHECK: cmp w0, #6
; CHECK-NEXT: ccmp w1, #17, #0, ge
; CHECK-NEXT: b.eq
; CHECK: %if.then
; CHECK: bl _foo
; CHECK: %if.end
define i32 @single_different(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp sle i32 %a, 5
  %cmp1 = icmp ne i32 %b, 17
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Second block clobbers the flags, can't convert (easily).
; CHECK: single_flagclobber
; CHECK: cmp
; CHECK: b.eq
; CHECK: cmp
; CHECK: b.gt
define i32 @single_flagclobber(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 5
  br i1 %cmp, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp1 = icmp slt i32 %b, 7
  %mul = shl nsw i32 %b, 1
  %add = add nsw i32 %b, 1
  %cond = select i1 %cmp1, i32 %mul, i32 %add
  %cmp2 = icmp slt i32 %cond, 17
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %entry
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false
  ret i32 7
}

; Second block clobbers the flags and ends with a tbz terminator.
; CHECK: single_flagclobber_tbz
; CHECK: cmp
; CHECK: b.eq
; CHECK: cmp
; CHECK: tbz
define i32 @single_flagclobber_tbz(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 5
  br i1 %cmp, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp1 = icmp slt i32 %b, 7
  %mul = shl nsw i32 %b, 1
  %add = add nsw i32 %b, 1
  %cond = select i1 %cmp1, i32 %mul, i32 %add
  %and = and i32 %cond, 8
  %cmp2 = icmp ne i32 %and, 0
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %entry
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false
  ret i32 7
}

; Speculatively execute division by zero.
; The sdiv/udiv instructions do not trap when the divisor is zero, so they are
; safe to speculate.
; CHECK: speculate_division
; CHECK-NOT: cmp
; CHECK: sdiv
; CHECK: cmp
; CHECK-NEXT: ccmp
define i32 @speculate_division(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:
  %div = sdiv i32 %b, %a
  %cmp1 = icmp slt i32 %div, 17
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Floating point compare.
; CHECK: single_fcmp
; CHECK: cmp
; CHECK-NOT: b.
; CHECK: fccmp {{.*}}, #8, ge
; CHECK: b.lt
define i32 @single_fcmp(i32 %a, float %b) nounwind ssp {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:
  %conv = sitofp i32 %a to float
  %div = fdiv float %b, %conv
  %cmp1 = fcmp oge float %div, 1.700000e+01
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Chain multiple compares.
; CHECK: multi_different
; CHECK: cmp
; CHECK: ccmp
; CHECK: ccmp
; CHECK: b.
define void @multi_different(i32 %a, i32 %b, i32 %c) nounwind ssp {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:
  %div = sdiv i32 %b, %a
  %cmp1 = icmp eq i32 %div, 5
  %cmp4 = icmp sgt i32 %div, %c
  %or.cond = and i1 %cmp1, %cmp4
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret void
}

; Convert a cbz in the head block.
; CHECK: cbz_head
; CHECK: cmp w0, #0
; CHECK: ccmp
define i32 @cbz_head(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 0
  %cmp1 = icmp ne i32 %b, 17
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Check that the immediate operand is in range. The ccmp instruction encodes a
; smaller range of immediates than subs/adds.
; The ccmp immediates must be in the range 0-31.
; CHECK: immediate_range
; CHECK-NOT: ccmp
define i32 @immediate_range(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 5
  %cmp1 = icmp eq i32 %b, 32
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Convert a cbz in the second block.
; CHECK: cbz_second
; CHECK: cmp w0, #0
; CHECK: ccmp w1, #0, #0, ne
; CHECK: b.eq
define i32 @cbz_second(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 0
  %cmp1 = icmp ne i32 %b, 0
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}

; Convert a cbnz in the second block.
; CHECK: cbnz_second
; CHECK: cmp w0, #0
; CHECK: ccmp w1, #0, #4, ne
; CHECK: b.ne
define i32 @cbnz_second(i32 %a, i32 %b) nounwind ssp {
entry:
  %cmp = icmp eq i32 %a, 0
  %cmp1 = icmp eq i32 %b, 0
  %or.cond = or i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end

if.then:
  %call = tail call i32 @foo() nounwind
  br label %if.end

if.end:
  ret i32 7
}
declare i32 @foo()

%str1 = type { %str2 }
%str2 = type { [24 x i8], i8*, i32, %str1*, i32, [4 x i8], %str1*, %str1*, %str1*, %str1*, %str1*, %str1*, %str1*, %str1*, %str1*, i8*, i8, i8*, %str1*, i8* }

; Test case distilled from 126.gcc.
; The phi in sw.bb.i.i gets multiple operands for the %entry predecessor.
; CHECK: build_modify_expr
define void @build_modify_expr() nounwind ssp {
entry:
  switch i32 undef, label %sw.bb.i.i [
    i32 69, label %if.end85
    i32 70, label %if.end85
    i32 71, label %if.end85
    i32 72, label %if.end85
    i32 73, label %if.end85
    i32 105, label %if.end85
    i32 106, label %if.end85
  ]

if.end85:
  ret void

sw.bb.i.i:
  %ref.tr.i.i = phi %str1* [ %0, %sw.bb.i.i ], [ undef, %entry ]
  %operands.i.i = getelementptr inbounds %str1, %str1* %ref.tr.i.i, i64 0, i32 0, i32 2
  %arrayidx.i.i = bitcast i32* %operands.i.i to %str1**
  %0 = load %str1** %arrayidx.i.i, align 8
  %code1.i.i.phi.trans.insert = getelementptr inbounds %str1, %str1* %0, i64 0, i32 0, i32 0, i64 16
  br label %sw.bb.i.i
}
