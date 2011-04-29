; RUN: llc < %s -O0 -fast-isel-abort -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=THUMB

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

; Check truncate to bool
define void @test1(i32 %tmp) nounwind {
entry:
%tobool = trunc i32 %tmp to i1
br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
call void @test1(i32 0)
br label %if.end

if.end:                                           ; preds = %if.then, %entry
ret void
; ARM: test1:
; ARM: tst r0, #1
; THUMB: test1:
; THUMB: tst.w r0, #1
}

; Check some simple operations with immediates
define void @test2(i32 %tmp, i32* %ptr) nounwind {
; THUMB: test2:
; ARM: test2:

b1:
  %b = add i32 %tmp, 4096
  store i32 %b, i32* %ptr
  br label %b2

; THUMB: add.w {{.*}} #4096
; ARM: add {{.*}} #1, #20

b2:
  %c = or i32 %tmp, 4
  store i32 %c, i32* %ptr
  ret void

; THUMB: orr {{.*}} #4
; ARM: orr {{.*}} #4
}
