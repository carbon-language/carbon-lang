; RUN: llc < %s -fast-isel -fast-isel-abort -march=x86 -mattr=sse2
; RUN: llc < %s -fast-isel -fast-isel-abort -march=x86-64

; This tests very minimal fast-isel functionality.

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
  call void asm sideeffect "hello world", ""()
  br label %exit

exit:
  ret i32* %t8
}

define double @bar(double* %p, double* %q) nounwind {
entry:
  %r = load double* %p
  %s = load double* %q
  br label %fast

fast:
  %t0 = fadd double %r, %s
  %t1 = fmul double %t0, %s
  %t2 = fsub double %t1, %s
  %t3 = fadd double %t2, 707.0
  br label %exit

exit:
  ret double %t3
}

define i32 @cast() nounwind {
entry:
	%tmp2 = bitcast i32 0 to i32
	ret i32 %tmp2
}

define void @ptrtoint_i1(i8* %p, i1* %q) nounwind {
  %t = ptrtoint i8* %p to i1
  store i1 %t, i1* %q
  ret void
}
define i8* @inttoptr_i1(i1 %p) nounwind {
  %t = inttoptr i1 %p to i8*
  ret i8* %t
}
define i32 @ptrtoint_i32(i8* %p) nounwind {
  %t = ptrtoint i8* %p to i32
  ret i32 %t
}
define i8* @inttoptr_i32(i32 %p) nounwind {
  %t = inttoptr i32 %p to i8*
  ret i8* %t
}

define i8 @trunc_i32_i8(i32 %x) signext nounwind  {
	%tmp1 = trunc i32 %x to i8
	ret i8 %tmp1
}

define i8 @trunc_i16_i8(i16 signext %x) signext nounwind  {
	%tmp1 = trunc i16 %x to i8
	ret i8 %tmp1
}

define i8 @shl_i8(i8 %a, i8 %c) nounwind {
       %tmp = shl i8 %a, %c
       ret i8 %tmp
}

define i8 @mul_i8(i8 %a) nounwind {
       %tmp = mul i8 %a, 17
       ret i8 %tmp
}

define void @load_store_i1(i1* %p, i1* %q) nounwind {
  %t = load i1* %p
  store i1 %t, i1* %q
  ret void
}


@crash_test1x = external global <2 x i32>, align 8

define void @crash_test1() nounwind ssp {
  %tmp = load <2 x i32>* @crash_test1x, align 8
  %neg = xor <2 x i32> %tmp, <i32 -1, i32 -1>
  ret void
}

