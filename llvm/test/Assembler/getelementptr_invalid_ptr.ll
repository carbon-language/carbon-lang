; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test the case of an invalid pointer type on a GEP

; CHECK: pointer type is not valid

define i32* @foo(i32 %a) {
  %gep = getelementptr i32, i32 %a, i32 1
  return i32* %gep
}

