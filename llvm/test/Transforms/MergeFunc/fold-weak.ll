; RUN: opt < %s -mergefunc -S > %t
; RUN: grep {define weak} %t | count 2
; RUN: grep {call} %t | count 2

define weak i32 @sum(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  ret i32 %sum
}

define weak i32 @add(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  ret i32 %sum
}
