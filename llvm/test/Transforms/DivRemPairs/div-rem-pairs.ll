; RUN: opt < %s -div-rem-pairs -S -mtriple=x86_64-unknown-unknown    | FileCheck %s --check-prefix=ALL --check-prefix=X86
; RUN: opt < %s -div-rem-pairs -S -mtriple=powerpc64-unknown-unknown | FileCheck %s --check-prefix=ALL --check-prefix=PPC

declare void @foo(i32, i32)

define void @decompose_illegal_srem_same_block(i32 %a, i32 %b) {
; X86-LABEL: @decompose_illegal_srem_same_block(
; X86-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; X86-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; X86-NEXT:    call void @foo(i32 [[REM]], i32 [[DIV]])
; X86-NEXT:    ret void
;
; PPC-LABEL: @decompose_illegal_srem_same_block(
; PPC-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; PPC-NEXT:    [[TMP1:%.*]] = mul i32 [[DIV]], %b
; PPC-NEXT:    [[TMP2:%.*]] = sub i32 %a, [[TMP1]]
; PPC-NEXT:    call void @foo(i32 [[TMP2]], i32 [[DIV]])
; PPC-NEXT:    ret void
;
  %rem = srem i32 %a, %b
  %div = sdiv i32 %a, %b
  call void @foo(i32 %rem, i32 %div)
  ret void
}

define void @decompose_illegal_urem_same_block(i32 %a, i32 %b) {
; X86-LABEL: @decompose_illegal_urem_same_block(
; X86-NEXT:    [[DIV:%.*]] = udiv i32 %a, %b
; X86-NEXT:    [[REM:%.*]] = urem i32 %a, %b
; X86-NEXT:    call void @foo(i32 [[REM]], i32 [[DIV]])
; X86-NEXT:    ret void
;
; PPC-LABEL: @decompose_illegal_urem_same_block(
; PPC-NEXT:    [[DIV:%.*]] = udiv i32 %a, %b
; PPC-NEXT:    [[TMP1:%.*]] = mul i32 [[DIV]], %b
; PPC-NEXT:    [[TMP2:%.*]] = sub i32 %a, [[TMP1]]
; PPC-NEXT:    call void @foo(i32 [[TMP2]], i32 [[DIV]])
; PPC-NEXT:    ret void
;
  %div = udiv i32 %a, %b
  %rem = urem i32 %a, %b
  call void @foo(i32 %rem, i32 %div)
  ret void
}

; Hoist and optionally decompose the sdiv because it's safe and free.
; PR31028 - https://bugs.llvm.org/show_bug.cgi?id=31028

define i32 @hoist_sdiv(i32 %a, i32 %b) {
; X86-LABEL: @hoist_sdiv(
; X86-NEXT:  entry:
; X86-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; X86-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; X86-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; X86-NEXT:    br i1 [[CMP]], label %if, label %end
; X86:       if:
; X86-NEXT:    br label %end
; X86:       end:
; X86-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ 3, %entry ]
; X86-NEXT:    ret i32 [[RET]]
;
; PPC-LABEL: @hoist_sdiv(
; PPC-NEXT:  entry:
; PPC-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; PPC-NEXT:    [[TMP0:%.*]] = mul i32 [[DIV]], %b
; PPC-NEXT:    [[TMP1:%.*]] = sub i32 %a, [[TMP0]]
; PPC-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP1]], 42
; PPC-NEXT:    br i1 [[CMP]], label %if, label %end
; PPC:       if:
; PPC-NEXT:    br label %end
; PPC:       end:
; PPC-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ 3, %entry ]
; PPC-NEXT:    ret i32 [[RET]]
;
entry:
  %rem = srem i32 %a, %b
  %cmp = icmp eq i32 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %div = sdiv i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ 3, %entry ]
  ret i32 %ret
}

; Hoist and optionally decompose the udiv because it's safe and free.

define i64 @hoist_udiv(i64 %a, i64 %b) {
; X86-LABEL: @hoist_udiv(
; X86-NEXT:  entry:
; X86-NEXT:    [[REM:%.*]] = urem i64 %a, %b
; X86-NEXT:    [[DIV:%.*]] = udiv i64 %a, %b
; X86-NEXT:    [[CMP:%.*]] = icmp eq i64 [[REM]], 42
; X86-NEXT:    br i1 [[CMP]], label %if, label %end
; X86:       if:
; X86-NEXT:    br label %end
; X86:       end:
; X86-NEXT:    [[RET:%.*]] = phi i64 [ [[DIV]], %if ], [ 3, %entry ]
; X86-NEXT:    ret i64 [[RET]]
;
; PPC-LABEL: @hoist_udiv(
; PPC-NEXT:  entry:
; PPC-NEXT:    [[DIV:%.*]] = udiv i64 %a, %b
; PPC-NEXT:    [[TMP0:%.*]] = mul i64 [[DIV]], %b
; PPC-NEXT:    [[TMP1:%.*]] = sub i64 %a, [[TMP0]]
; PPC-NEXT:    [[CMP:%.*]] = icmp eq i64 [[TMP1]], 42
; PPC-NEXT:    br i1 [[CMP]], label %if, label %end
; PPC:       if:
; PPC-NEXT:    br label %end
; PPC:       end:
; PPC-NEXT:    [[RET:%.*]] = phi i64 [ [[DIV]], %if ], [ 3, %entry ]
; PPC-NEXT:    ret i64 [[RET]]
;
entry:
  %rem = urem i64 %a, %b
  %cmp = icmp eq i64 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %div = udiv i64 %a, %b
  br label %end

end:
  %ret = phi i64 [ %div, %if ], [ 3, %entry ]
  ret i64 %ret
}

; Hoist the srem if it's safe and free, otherwise decompose it.

define i16 @hoist_srem(i16 %a, i16 %b) {
; X86-LABEL: @hoist_srem(
; X86-NEXT:  entry:
; X86-NEXT:    [[DIV:%.*]] = sdiv i16 %a, %b
; X86-NEXT:    [[REM:%.*]] = srem i16 %a, %b
; X86-NEXT:    [[CMP:%.*]] = icmp eq i16 [[DIV]], 42
; X86-NEXT:    br i1 [[CMP]], label %if, label %end
; X86:       if:
; X86-NEXT:    br label %end
; X86:       end:
; X86-NEXT:    [[RET:%.*]] = phi i16 [ [[REM]], %if ], [ 3, %entry ]
; X86-NEXT:    ret i16 [[RET]]
;
; PPC-LABEL: @hoist_srem(
; PPC-NEXT:  entry:
; PPC-NEXT:    [[DIV:%.*]] = sdiv i16 %a, %b
; PPC-NEXT:    [[CMP:%.*]] = icmp eq i16 [[DIV]], 42
; PPC-NEXT:    br i1 [[CMP]], label %if, label %end
; PPC:       if:
; PPC-NEXT:    [[TMP0:%.*]] = mul i16 [[DIV]], %b
; PPC-NEXT:    [[TMP1:%.*]] = sub i16 %a, [[TMP0]]
; PPC-NEXT:    br label %end
; PPC:       end:
; PPC-NEXT:    [[RET:%.*]] = phi i16 [ [[TMP1]], %if ], [ 3, %entry ]
; PPC-NEXT:    ret i16 [[RET]]
;
entry:
  %div = sdiv i16 %a, %b
  %cmp = icmp eq i16 %div, 42
  br i1 %cmp, label %if, label %end

if:
  %rem = srem i16 %a, %b
  br label %end

end:
  %ret = phi i16 [ %rem, %if ], [ 3, %entry ]
  ret i16 %ret
}

; Hoist the urem if it's safe and free, otherwise decompose it.

define i8 @hoist_urem(i8 %a, i8 %b) {
; X86-LABEL: @hoist_urem(
; X86-NEXT:  entry:
; X86-NEXT:    [[DIV:%.*]] = udiv i8 %a, %b
; X86-NEXT:    [[REM:%.*]] = urem i8 %a, %b
; X86-NEXT:    [[CMP:%.*]] = icmp eq i8 [[DIV]], 42
; X86-NEXT:    br i1 [[CMP]], label %if, label %end
; X86:       if:
; X86-NEXT:    br label %end
; X86:       end:
; X86-NEXT:    [[RET:%.*]] = phi i8 [ [[REM]], %if ], [ 3, %entry ]
; X86-NEXT:    ret i8 [[RET]]
;
; PPC-LABEL: @hoist_urem(
; PPC-NEXT:  entry:
; PPC-NEXT:    [[DIV:%.*]] = udiv i8 %a, %b
; PPC-NEXT:    [[CMP:%.*]] = icmp eq i8 [[DIV]], 42
; PPC-NEXT:    br i1 [[CMP]], label %if, label %end
; PPC:       if:
; PPC-NEXT:    [[TMP0:%.*]] = mul i8 [[DIV]], %b
; PPC-NEXT:    [[TMP1:%.*]] = sub i8 %a, [[TMP0]]
; PPC-NEXT:    br label %end
; PPC:       end:
; PPC-NEXT:    [[RET:%.*]] = phi i8 [ [[TMP1]], %if ], [ 3, %entry ]
; PPC-NEXT:    ret i8 [[RET]]
;
entry:
  %div = udiv i8 %a, %b
  %cmp = icmp eq i8 %div, 42
  br i1 %cmp, label %if, label %end

if:
  %rem = urem i8 %a, %b
  br label %end

end:
  %ret = phi i8 [ %rem, %if ], [ 3, %entry ]
  ret i8 %ret
}

; If the ops don't match, don't do anything: signedness.

define i32 @dont_hoist_udiv(i32 %a, i32 %b) {
; ALL-LABEL: @dont_hoist_udiv(
; ALL-NEXT:  entry:
; ALL-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; ALL-NEXT:    br i1 [[CMP]], label %if, label %end
; ALL:       if:
; ALL-NEXT:    [[DIV:%.*]] = udiv i32 %a, %b
; ALL-NEXT:    br label %end
; ALL:       end:
; ALL-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ 3, %entry ]
; ALL-NEXT:    ret i32 [[RET]]
;
entry:
  %rem = srem i32 %a, %b
  %cmp = icmp eq i32 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %div = udiv i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ 3, %entry ]
  ret i32 %ret
}

; If the ops don't match, don't do anything: operation.

define i32 @dont_hoist_srem(i32 %a, i32 %b) {
; ALL-LABEL: @dont_hoist_srem(
; ALL-NEXT:  entry:
; ALL-NEXT:    [[REM:%.*]] = urem i32 %a, %b
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; ALL-NEXT:    br i1 [[CMP]], label %if, label %end
; ALL:       if:
; ALL-NEXT:    [[REM2:%.*]] = srem i32 %a, %b
; ALL-NEXT:    br label %end
; ALL:       end:
; ALL-NEXT:    [[RET:%.*]] = phi i32 [ [[REM2]], %if ], [ 3, %entry ]
; ALL-NEXT:    ret i32 [[RET]]
;
entry:
  %rem = urem i32 %a, %b
  %cmp = icmp eq i32 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %rem2 = srem i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %rem2, %if ], [ 3, %entry ]
  ret i32 %ret
}

; If the ops don't match, don't do anything: operands.

define i32 @dont_hoist_sdiv(i32 %a, i32 %b, i32 %c) {
; ALL-LABEL: @dont_hoist_sdiv(
; ALL-NEXT:  entry:
; ALL-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; ALL-NEXT:    br i1 [[CMP]], label %if, label %end
; ALL:       if:
; ALL-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %c
; ALL-NEXT:    br label %end
; ALL:       end:
; ALL-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ 3, %entry ]
; ALL-NEXT:    ret i32 [[RET]]
;
entry:
  %rem = srem i32 %a, %b
  %cmp = icmp eq i32 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %div = sdiv i32 %a, %c
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ 3, %entry ]
  ret i32 %ret
}

; If the target doesn't have a unified div/rem op for the type, decompose rem in-place to mul+sub.

define i128 @dont_hoist_urem(i128 %a, i128 %b) {
; ALL-LABEL: @dont_hoist_urem(
; ALL-NEXT:  entry:
; ALL-NEXT:    [[DIV:%.*]] = udiv i128 %a, %b
; ALL-NEXT:    [[CMP:%.*]] = icmp eq i128 [[DIV]], 42
; ALL-NEXT:    br i1 [[CMP]], label %if, label %end
; ALL:       if:
; ALL-NEXT:    [[TMP0:%.*]] = mul i128 [[DIV]], %b
; ALL-NEXT:    [[TMP1:%.*]] = sub i128 %a, [[TMP0]]
; ALL-NEXT:    br label %end
; ALL:       end:
; ALL-NEXT:    [[RET:%.*]] = phi i128 [ [[TMP1]], %if ], [ 3, %entry ]
; ALL-NEXT:    ret i128 [[RET]]
;
entry:
  %div = udiv i128 %a, %b
  %cmp = icmp eq i128 %div, 42
  br i1 %cmp, label %if, label %end

if:
  %rem = urem i128 %a, %b
  br label %end

end:
  %ret = phi i128 [ %rem, %if ], [ 3, %entry ]
  ret i128 %ret
}

; We don't hoist if one op does not dominate the other,
; but we could hoist both ops to the common predecessor block?

define i32 @no_domination(i1 %cmp, i32 %a, i32 %b) {
; ALL-LABEL: @no_domination(
; ALL-NEXT:  entry:
; ALL-NEXT:    br i1 %cmp, label %if, label %else
; ALL:       if:
; ALL-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; ALL-NEXT:    br label %end
; ALL:       else:
; ALL-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; ALL-NEXT:    br label %end
; ALL:       end:
; ALL-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ [[REM]], %else ]
; ALL-NEXT:    ret i32 [[RET]]
;
entry:
  br i1 %cmp, label %if, label %else

if:
  %div = sdiv i32 %a, %b
  br label %end

else:
  %rem = srem i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ %rem, %else ]
  ret i32 %ret
}

