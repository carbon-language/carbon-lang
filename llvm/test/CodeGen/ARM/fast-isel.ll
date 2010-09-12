; RUN: llc < %s -O0 -arm-fast-isel -fast-isel-abort -mtriple=armv7-apple-darwin
; RUN: llc < %s -O0 -arm-fast-isel -fast-isel-abort -mtriple=thumbv7-apple-darwin

; Very basic fast-isel functionality.

define i32 @add(i32 %a, i32 %b) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %tmp = load i32* %a.addr
  %tmp1 = load i32* %b.addr
  %add = add nsw i32 %tmp, %tmp1
  ret i32 %add
}

define float @fp_ops(float %a, float %b) nounwind {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, float* %a.addr
  store float %b, float* %b.addr
  %tmp = load float* %a.addr
  %tmp1 = load float* %b.addr
  %mul = fmul float %tmp, %tmp1
  %tmp2 = load float* %b.addr
  %tmp3 = load float* %a.addr
  %mul2 = fmul float %tmp2, %tmp3
  %add = fadd float %mul, %mul2
  ret float %mul
}

define i32* @foo(i32* %p, i32* %q, i32** %z) nounwind {
entry:
  %r = load i32* %p
  %s = load i32* %q
  %y = load i32** %z
  br label %fast

fast:
  %t0 = add i32 %r, %s
  %t1 = mul i32 %t0, %s
  %t2 = sub i32 %t1, %s
  %t3 = and i32 %t2, %s
  %t4 = xor i32 %t3, 3
  %t5 = xor i32 %t4, %s
  %t6 = add i32 %t5, 2
  %t7 = getelementptr i32* %y, i32 1
  %t8 = getelementptr i32* %t7, i32 %t6
  br label %exit

exit:
  ret i32* %t8
}
