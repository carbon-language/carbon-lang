; RUN: opt < %s -mem2reg -S
; PR5023

declare i32 @bar()

define i32 @foo() {
entry:
  %whichFlag = alloca i32
  %A = invoke i32 @bar()
          to label %invcont2 unwind label %lpad86

invcont2:
  store i32 %A, i32* %whichFlag
  br label %bb15

bb15:
  %B = load i32* %whichFlag
  ret i32 %B

lpad86:
  br label %bb15
  
}

