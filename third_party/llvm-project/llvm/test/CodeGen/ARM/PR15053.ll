; RUN: llc -mtriple=armv7 < %s
; PR15053

declare i32 @llvm.arm.strexd(i32, i32, i8*) nounwind
declare { i32, i32 } @llvm.arm.ldrexd(i8*) nounwind readonly

define void @foo() {
entry:
  %0 = tail call { i32, i32 } @llvm.arm.ldrexd(i8* undef) nounwind
  %1 = extractvalue { i32, i32 } %0, 0
  %2 = tail call i32 @llvm.arm.strexd(i32 %1, i32 undef, i8* undef) nounwind
  ret void
}
