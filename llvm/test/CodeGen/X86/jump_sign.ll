; RUN: llc < %s -march=x86 -mcpu=pentiumpro | FileCheck %s

define i32 @f(i32 %X) {
entry:
; CHECK: f:
; CHECK: jns
	%tmp1 = add i32 %X, 1		; <i32> [#uses=1]
	%tmp = icmp slt i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp2 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp3 = tail call i32 (...)* @baz( )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @bar(...)

declare i32 @baz(...)

; rdar://10633221
; rdar://11355268
define i32 @g(i32 %a, i32 %b) nounwind {
entry:
; CHECK: g:
; CHECK-NOT: test
; CHECK: cmovs
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, 0
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}

; rdar://10734411
define i32 @h(i32 %a, i32 %b) nounwind {
entry:
; CHECK: h:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp slt i32 %b, %a
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @i(i32 %a, i32 %b) nounwind {
entry:
; CHECK: i:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @j(i32 %a, i32 %b) nounwind {
entry:
; CHECK: j:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp ugt i32 %a, %b
  %sub = sub i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @k(i32 %a, i32 %b) nounwind {
entry:
; CHECK: k:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp ult i32 %b, %a
  %sub = sub i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
; rdar://11540023
define i32 @n(i32 %x, i32 %y) nounwind {
entry:
; CHECK: n:
; CHECK-NOT: sub
; CHECK: cmp
  %sub = sub nsw i32 %x, %y
  %cmp = icmp slt i32 %sub, 0
  %y.x = select i1 %cmp, i32 %y, i32 %x
  ret i32 %y.x
}
; PR://13046
define void @o() nounwind uwtable {
entry:
  %0 = load i16* undef, align 2
  br i1 undef, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %entry
  unreachable

if.end.i:                                         ; preds = %entry
  br i1 undef, label %sw.bb, label %sw.default

sw.bb:                                            ; preds = %if.end.i
  br i1 undef, label %if.then44, label %if.end29

if.end29:                                         ; preds = %sw.bb
; CHECK: o:
; CHECK: cmp
  %1 = urem i16 %0, 10
  %cmp25 = icmp eq i16 %1, 0
  %. = select i1 %cmp25, i16 2, i16 0
  br i1 %cmp25, label %if.then44, label %sw.default

sw.default:                                       ; preds = %if.end29, %if.end.i
  br i1 undef, label %if.then.i96, label %if.else.i97

if.then.i96:                                      ; preds = %sw.default
  unreachable

if.else.i97:                                      ; preds = %sw.default
  unreachable

if.then44:                                        ; preds = %if.end29, %sw.bb
  %aModeRefSel.1.ph = phi i16 [ %., %if.end29 ], [ 3, %sw.bb ]
  br i1 undef, label %if.then.i103, label %if.else.i104

if.then.i103:                                     ; preds = %if.then44
  unreachable

if.else.i104:                                     ; preds = %if.then44
  ret void
}
