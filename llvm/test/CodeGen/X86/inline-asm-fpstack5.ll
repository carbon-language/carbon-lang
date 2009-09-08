; RUN: llc < %s -march=x86
; PR4485

define void @test(x86_fp80* %a) {
entry:
	%0 = load x86_fp80* %a, align 16
	%1 = fmul x86_fp80 %0, 0xK4006B400000000000000
	%2 = fmul x86_fp80 %1, 0xK4012F424000000000000
	tail call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %2)
	%3 = load x86_fp80* %a, align 16
	%4 = fmul x86_fp80 %3, 0xK4006B400000000000000
	%5 = fmul x86_fp80 %4, 0xK4012F424000000000000
	tail call void asm sideeffect "fistpl $0", "{st},~{st}"(x86_fp80 %5)
	ret void
}
