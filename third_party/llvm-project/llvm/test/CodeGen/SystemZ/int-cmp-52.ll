; This used to crash the backend due to a failed assertion.
; No particular output expected, but must compile.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu

define void @test(i16 *%input, i32 *%result) {
entry:
  %0 = load i16, i16* %input, align 2
  %1 = zext i16 %0 to i32
  %2 = icmp slt i32 %1, 0
  br i1 %2, label %if.then, label %if.else

if.then:
  store i32 1, i32* %result, align 4
  br label %return

if.else:
  store i32 0, i32* %result, align 4
  br label %return

return:
  ret void
}

