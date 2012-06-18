; RUN: not llvm-as < %s -o /dev/null |& FileCheck %s

declare i32 @size() readonly
declare i32 @sizeR()
declare i32 @size1(i32) readnone
declare i32 @size1i8(i8) readnone
declare i32* @sizeptr() readnone

define void @f1(i8** %x, i32* %y) {
entry:
  %0 = load i8** %x, !alloc !0
  %1 = load i8** %x, !alloc !1
  %2 = load i8** %x, !alloc !2
  %3 = load i8** %x, !alloc !3
  %4 = load i8** %x, !alloc !4
  %5 = load i8** %x, !alloc !5
  %6 = load i8** %x, !alloc !6
  %7 = load i8** %x, !alloc !7
  %8 = load i8** %x, !alloc !8
  %9 = load i32* %y, !alloc !9
  %10 = load i8** %x, !alloc !10
  %11 = load i8** %x, !alloc !11
  ret void
}
; CHECK: alloc takes at least one operand
!0 = metadata !{}
; CHECK: first parameter of alloc must be a function
!1 = metadata !{i32 0}
; CHECK: second parameter of alloc must be either a function or null
!2 = metadata !{i32 ()* @size, i32 0}
; CHECK: size function number of parameters mismatch
!3 = metadata !{i32 ()* @size, null, i32 0}
; CHECK: offset function number of parameters mismatch
!4 = metadata !{i32 (i32)* @size1, i32 ()* @size, i32 1}
; CHECK: size function must be readonly/readnone
!5 = metadata !{i32 ()* @sizeR, i32 ()* @size}
; CHECK: offset function must be readonly/readnone
!6 = metadata !{i32 ()* @size, i32 ()* @sizeR}
; CHECK: size function parameter type mismatch
!7 = metadata !{i32 (i32)* @size1, i32 (i8)* @size1i8, i8 5}
; CHECK: offset function parameter type mismatch
!8 = metadata !{i32 (i8)* @size1i8, i32 (i32)* @size1, i8 5}
; CHECK: alloc requires a pointer result
!9 = metadata !{i32 ()* @size, null}
; CHECK: size function must return an integer
!10 = metadata !{i32* ()* @sizeptr, null}
; CHECK: offset function must return an integer
!11 = metadata !{i32 ()* @size, i32* ()* @sizeptr}
