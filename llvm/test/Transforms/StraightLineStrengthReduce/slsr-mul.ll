; RUN: opt < %s -slsr -gvn -dce -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

declare i32 @foo(i32 %a)

define i32 @slsr1(i32 %b, i32 %s) {
; CHECK-LABEL: @slsr1(
  ; v0 = foo(b * s);
  %mul0 = mul i32 %b, %s
; CHECK: mul i32
; CHECK-NOT: mul i32
  %v0 = call i32 @foo(i32 %mul0)

  ; v1 = foo((b + 1) * s);
  %b1 = add i32 %b, 1
  %mul1 = mul i32 %b1, %s
  %v1 = call i32 @foo(i32 %mul1)

  ; v2 = foo((b + 2) * s);
  %b2 = add i32 %b, 2
  %mul2 = mul i32 %b2, %s
  %v2 = call i32 @foo(i32 %mul2)

  ; return v0 + v1 + v2;
  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  ret i32 %2
}

; v0 = foo(a * b)
; v1 = foo((a + 1) * b)
; v2 = foo(a * (b + 1))
; v3 = foo((a + 1) * (b + 1))
define i32 @slsr2(i32 %a, i32 %b) {
; CHECK-LABEL: @slsr2(
  %a1 = add i32 %a, 1
  %b1 = add i32 %b, 1
  %mul0 = mul i32 %a, %b
; CHECK: mul i32
; CHECK-NOT: mul i32
  %mul1 = mul i32 %a1, %b
  %mul2 = mul i32 %a, %b1
  %mul3 = mul i32 %a1, %b1

  %v0 = call i32 @foo(i32 %mul0)
  %v1 = call i32 @foo(i32 %mul1)
  %v2 = call i32 @foo(i32 %mul2)
  %v3 = call i32 @foo(i32 %mul3)

  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  %3 = add i32 %2, %v3
  ret i32 %3
}

; The bump is a multiple of the stride.
;
; v0 = foo(b * s);
; v1 = foo((b + 2) * s);
; v2 = foo((b + 4) * s);
; return v0 + v1 + v2;
;
; ==>
;
; mul0 = b * s;
; v0 = foo(mul0);
; bump = s * 2;
; mul1 = mul0 + bump; // GVN ensures mul1 and mul2 use the same bump.
; v1 = foo(mul1);
; mul2 = mul1 + bump;
; v2 = foo(mul2);
; return v0 + v1 + v2;
define i32 @slsr3(i32 %b, i32 %s) {
; CHECK-LABEL: @slsr3(
  %mul0 = mul i32 %b, %s
; CHECK: mul i32
  %v0 = call i32 @foo(i32 %mul0)

  %b1 = add i32 %b, 2
  %mul1 = mul i32 %b1, %s
; CHECK: [[BUMP:%[a-zA-Z0-9]+]] = shl i32 %s, 1
; CHECK: %mul1 = add i32 %mul0, [[BUMP]]
  %v1 = call i32 @foo(i32 %mul1)

  %b2 = add i32 %b, 4
  %mul2 = mul i32 %b2, %s
; CHECK: %mul2 = add i32 %mul1, [[BUMP]]
  %v2 = call i32 @foo(i32 %mul2)

  %1 = add i32 %v0, %v1
  %2 = add i32 %1, %v2
  ret i32 %2
}

; Do not rewrite a candidate if its potential basis does not dominate it.
; v0 = 0;
; if (cond)
;   v0 = foo(a * b);
; v1 = foo((a + 1) * b);
; return v0 + v1;
define i32 @not_dominate(i1 %cond, i32 %a, i32 %b) {
; CHECK-LABEL: @not_dominate(
entry:
  %a1 = add i32 %a, 1
  br i1 %cond, label %then, label %merge

then:
  %mul0 = mul i32 %a, %b
; CHECK: %mul0 = mul i32 %a, %b
  %v0 = call i32 @foo(i32 %mul0)
  br label %merge

merge:
  %v0.phi = phi i32 [ 0, %entry ], [ %mul0, %then ]
  %mul1 = mul i32 %a1, %b
; CHECK: %mul1 = mul i32 %a1, %b
  %v1 = call i32 @foo(i32 %mul1)
  %sum = add i32 %v0.phi, %v1
  ret i32 %sum
}
