; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; Test that align(N) is accepted as an alternative syntax to align N

; CHECK: define void @param_align4(i8* align 4 %ptr) {
define void @param_align4(i8* align(4) %ptr) {
  ret void
}

; CHECK: define void @param_align128(i8* align 128 %0) {
define void @param_align128(i8* align(128)) {
  ret void
}
