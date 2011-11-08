; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

define i32 @t0(i1 zeroext %a) nounwind {
  %1 = zext i1 %a to i32
  ret i32 %1
}

define i32 @t1(i8 signext %a) nounwind {
  %1 = sext i8 %a to i32
  ret i32 %1
}

define i32 @t2(i8 zeroext %a) nounwind {
  %1 = zext i8 %a to i32
  ret i32 %1
}

define i32 @t3(i16 signext %a) nounwind {
  %1 = sext i16 %a to i32
  ret i32 %1
}

define i32 @t4(i16 zeroext %a) nounwind {
  %1 = zext i16 %a to i32
  ret i32 %1
}

define void @foo(i8 %a, i16 %b) nounwind {
; ARM: foo
; THUMB: foo
;; Materialize i1 1
; ARM: movw r2, #1
;; zero-ext
; ARM: and r2, r2, #1
; THUMB: and r2, r2, #1
  %1 = call i32 @t0(i1 zeroext 1)
; ARM: sxtb	r2, r1
; ARM: mov r0, r2
; THUMB: sxtb	r2, r1
; THUMB: mov r0, r2
  %2 = call i32 @t1(i8 signext %a)
; ARM: uxtb	r2, r1
; ARM: mov r0, r2
; THUMB: uxtb	r2, r1
; THUMB: mov r0, r2
  %3 = call i32 @t2(i8 zeroext %a)
; ARM: sxth	r2, r1
; ARM: mov r0, r2
; THUMB: sxth	r2, r1
; THUMB: mov r0, r2
  %4 = call i32 @t3(i16 signext %b)
; ARM: uxth	r2, r1
; ARM: mov r0, r2
; THUMB: uxth	r2, r1
; THUMB: mov r0, r2
  %5 = call i32 @t4(i16 zeroext %b)

;; A few test to check materialization
;; Note: i1 1 was materialized with t1 call
; ARM: movw r1, #255
%6 = call i32 @t2(i8 zeroext 255)
; ARM: movw r1, #65535
; THUMB: movw r1, #65535
%7 = call i32 @t4(i16 zeroext 65535)
  ret void
}

define void @foo2() nounwind {
  %1 = call signext i16 @t5()
  %2 = call zeroext i16 @t6()
  %3 = call signext i8 @t7()
  %4 = call zeroext i8 @t8()
  %5 = call zeroext i1 @t9()
  ret void
}

declare signext i16 @t5();
declare zeroext i16 @t6();
declare signext i8 @t7();
declare zeroext i8 @t8();
declare zeroext i1 @t9();
