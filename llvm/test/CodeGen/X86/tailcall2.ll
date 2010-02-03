; RUN: llc < %s -march=x86    -asm-verbose=false | FileCheck %s -check-prefix=32
; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s -check-prefix=64
; XFAIL: *

define void @t1(i32 %x) nounwind ssp {
entry:
; 32: t1:
; 32: jmp {{_?}}foo

; 64: t1:
; 64: jmp {{_?}}foo
  tail call void @foo() nounwind
  ret void
}

declare void @foo()

define void @t2() nounwind ssp {
entry:
; 32: t2:
; 32: jmp {{_?}}foo2

; 64: t2:
; 64: jmp {{_?}}foo2
  %0 = tail call i32 @foo2() nounwind
  ret void
}

declare i32 @foo2()

define void @t3() nounwind ssp {
entry:
; 32: t3:
; 32: jmp {{_?}}foo3

; 64: t3:
; 64: jmp {{_?}}foo3
  %0 = tail call i32 @foo3() nounwind
  ret void
}

declare i32 @foo3()

define void @t4(void (i32)* nocapture %x) nounwind ssp {
entry:
; 32: t4:
; 32: call *
; FIXME: gcc can generate a tailcall for this. But it's tricky.

; 64: t4:
; 64-NOT: call
; 64: jmpq *
  tail call void %x(i32 0) nounwind
  ret void
}

define void @t5(void ()* nocapture %x) nounwind ssp {
entry:
; 32: t5:
; 32-NOT: call
; 32: jmpl *

; 64: t5:
; 64-NOT: call
; 64: jmpq *
  tail call void %x() nounwind
  ret void
}

define i32 @t6(i32 %x) nounwind ssp {
entry:
; 32: t6:
; 32: call {{_?}}t6
; 32: call {{_?}}bar

; 64: t6:
; 64: jmp {{_?}}t6
; 64: jmp {{_?}}bar
  %0 = icmp slt i32 %x, 10
  br i1 %0, label %bb, label %bb1

bb:
  %1 = add nsw i32 %x, -1
  %2 = tail call i32 @t6(i32 %1) nounwind ssp
  ret i32 %2

bb1:
  %3 = tail call i32 @bar(i32 %x) nounwind
  ret i32 %3
}

declare i32 @bar(i32)

define i32 @t7(i32 %a, i32 %b, i32 %c) nounwind ssp {
entry:
; 32: t7:
; 32: jmp {{_?}}bar2

; 64: t7:
; 64: jmp {{_?}}bar2
  %0 = tail call i32 @bar2(i32 %a, i32 %b, i32 %c) nounwind
  ret i32 %0
}

declare i32 @bar2(i32, i32, i32)

define signext i16 @t8() nounwind ssp {
entry:
; 32: t8:
; 32: jmp {{_?}}bar3

; 64: t8:
; 64: jmp {{_?}}bar3
  %0 = tail call signext i16 @bar3() nounwind      ; <i16> [#uses=1]
  ret i16 %0
}

declare signext i16 @bar3()

define signext i16 @t9(i32 (i32)* nocapture %x) nounwind ssp {
entry:
; 32: t9:
; 32: call *

; 64: t9:
; 64: callq *
  %0 = bitcast i32 (i32)* %x to i16 (i32)*
  %1 = tail call signext i16 %0(i32 0) nounwind
  ret i16 %1
}

define void @t10() nounwind ssp {
entry:
; 32: t10:
; 32: call

; 64: t10:
; 64: callq
  %0 = tail call i32 @foo4() noreturn nounwind
  unreachable
}

declare i32 @foo4()
