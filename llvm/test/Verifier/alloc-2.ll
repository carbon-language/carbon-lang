; RUN: llvm-as < %s -o /dev/null

declare i32 @size() readonly
declare i32 @size1(i32) readnone
declare i32 @size1i8(i8) readnone

define void @ok(i8** %x, i32 %y) {
entry:
  %0 = load i8** %x, !alloc !0
  %1 = load i8** %x, !alloc !1
  %2 = load i8** %x, !alloc !2
  %3 = load i8** %x, !alloc !3
  %4 = load i8** %x, !alloc !{i32 (i32)* @size1, i32 (i32)* @size1, i32 %y}
  ret void
}
!0 = metadata !{i32 ()* @size, i32 ()* @size}
!1 = metadata !{i32 ()* @size, null}
!2 = metadata !{i32 (i32)* @size1, i32 (i32)* @size1, i32 0}
!3 = metadata !{i32 (i8)* @size1i8, i32 (i8)* @size1i8, i8 0}
