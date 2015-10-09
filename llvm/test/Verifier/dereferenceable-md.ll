; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i8* @foo()

define void @f1() {
entry:
  call i8* @foo(), !dereferenceable !{i64 2}
  ret void
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to load instructions, use attributes for calls or invokes
; CHECK-NEXT: call i8* @foo()

define void @f2() {
entry:
  call i8* @foo(), !dereferenceable_or_null !{i64 2}
  ret void
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to load instructions, use attributes for calls or invokes
; CHECK-NEXT: call i8* @foo()

define i8 @f3(i8* %x) {
entry:
  %y = load i8, i8* %x, !dereferenceable !{i64 2}
  ret i8 %y
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to pointer types
; CHECK-NEXT: load i8, i8* %x

define i8 @f4(i8* %x) {
entry:
  %y = load i8, i8* %x, !dereferenceable_or_null !{i64 2}
  ret i8 %y
}
; CHECK: dereferenceable, dereferenceable_or_null apply only to pointer types
; CHECK-NEXT: load i8, i8* %x

define i8* @f5(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable !{}
  ret i8* %y
}
; CHECK: dereferenceable, dereferenceable_or_null take one operand
; CHECK-NEXT: load i8*, i8** %x


define i8* @f6(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_or_null !{}
  ret i8* %y
}
; CHECK: dereferenceable, dereferenceable_or_null take one operand
; CHECK-NEXT: load i8*, i8** %x

define i8* @f7(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable !{!"str"}
  ret i8* %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x


define i8* @f8(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_or_null !{!"str"}
  ret i8* %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x

define i8* @f9(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable !{i32 2}
  ret i8* %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x


define i8* @f10(i8** %x) {
entry:
  %y = load i8*, i8** %x, !dereferenceable_or_null !{i32 2}
  ret i8* %y
}
; CHECK: dereferenceable, dereferenceable_or_null metadata value must be an i64!
; CHECK-NEXT: load i8*, i8** %x