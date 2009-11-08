; RUN: lli -force-interpreter
; Extending a value due to zeroext/signext will leave it the wrong size
; causing problems later, such as a crash if you try to extend it again.

define void @zero(i8 zeroext %foo) {
  zext i8 %foo to i32
  ret void
}

define void @sign(i8 signext %foo) {
  sext i8 %foo to i32
  ret void
}

define i32 @main() {
  call void @zero(i8 0)
  call void @sign(i8 0)
  ret i32 0
}
