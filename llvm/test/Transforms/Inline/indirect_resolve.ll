; RUN: llvm-as < %s | opt -inline | llvm-dis
; PR4834

define i32 @main() {
  %funcall1_ = call fastcc i32 ()* ()* @f1()
  %executecommandptr1_ = call i32 %funcall1_()
  ret i32 %executecommandptr1_
}

define internal fastcc i32 ()* @f1() nounwind readnone {
  ret i32 ()* @f2
}

define internal i32 @f2() nounwind readnone {
  ret i32 1
}
