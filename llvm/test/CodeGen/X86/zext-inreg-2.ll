; RUN: llvm-as < %s | llc -march=x86-64 > %t
; RUN: not grep and %t
; RUN: not grep movzbq %t
; RUN: not grep movzwq %t
; RUN: not grep movzlq %t

; These should use movzbl instead of 'and 255'.
; This related to not having a ZERO_EXTEND_REG opcode.

; This test was split out of zext-inreg-0.ll because these
; cases don't yet work on x86-32 due to the 8-bit subreg
; issue.

define i32 @a(i32 %d) nounwind  {
        %e = add i32 %d, 1
        %retval = and i32 %e, 255
        ret i32 %retval
}
define i32 @b(float %d) nounwind  {
        %tmp12 = fptoui float %d to i8
        %retval = zext i8 %tmp12 to i32
        ret i32 %retval
}
define i64 @d(i64 %d) nounwind  {
        %e = add i64 %d, 1
        %retval = and i64 %e, 255
        ret i64 %retval
}
