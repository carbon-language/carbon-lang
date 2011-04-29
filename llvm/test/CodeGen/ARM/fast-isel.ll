; RUN: llc < %s -O0 -fast-isel-abort -mtriple=armv7-apple-darwin
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=thumbv7-apple-darwin

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

define void @test1(i32 %tmp) nounwind {
entry:
%tobool = trunc i32 %tmp to i1
br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
call void @test1(i32 0)
br label %if.end

if.end:                                           ; preds = %if.then, %entry
ret void
; CHECK: test1:
; CHECK: tst	r0, #1
}
