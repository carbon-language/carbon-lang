; Test function attributes
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: define void @fn1() alwaysinline
define void @fn1() alwaysinline {
  ret void
}

; CHECK: define void @fn2() noinline
define void @fn2() noinline {
  ret void
}

; CHECK: define void @fn3()
; CHECK-NOT: define void @fn3(){{.*}}inline
define void @fn3() {
  ret void
}
