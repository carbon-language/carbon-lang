; RUN: opt < %s -deadargelim | llvm-dis
; PR3807

define internal { i32, i32 } @foo() {
  ret {i32,i32} {i32 42, i32 4}
}

define i32 @bar() {
  %x = invoke {i32,i32} @foo() to label %T unwind label %T2
T:
  %y = extractvalue {i32,i32} %x, 1
  ret i32 %y
T2:
  unreachable
}

define i32 @bar2() {
entry:
  %x = invoke {i32,i32} @foo() to label %T unwind label %T2
T:
  %PN = phi i32 [0, %entry]
  %y = extractvalue {i32,i32} %x, 1
  ret i32 %y
T2:
  unreachable
}
