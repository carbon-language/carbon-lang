; RUN: opt < %s -constprop -S | FileCheck %s

%i8i1 = type {i8, i1}

;;-----------------------------
;; uadd
;;-----------------------------

define {i8, i1} @uadd_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 42, i8 100)
  ret {i8, i1} %t

; CHECK: @uadd_1
; CHECK: ret %i8i1 { i8 -114, i1 false }
}

define {i8, i1} @uadd_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 142, i8 120)
  ret {i8, i1} %t

; CHECK: @uadd_2
; CHECK: ret %i8i1 { i8 6, i1 true }
}


;;-----------------------------
;; usub
;;-----------------------------

define {i8, i1} @usub_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.usub.with.overflow.i8(i8 4, i8 2)
  ret {i8, i1} %t

; CHECK: @usub_1
; CHECK: ret %i8i1 { i8 2, i1 false }
}

define {i8, i1} @usub_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.usub.with.overflow.i8(i8 4, i8 6)
  ret {i8, i1} %t

; CHECK: @usub_2
; CHECK: ret %i8i1 { i8 -2, i1 true }
}



declare {i8, i1} @llvm.uadd.with.overflow.i8(i8, i8)
declare {i8, i1} @llvm.usub.with.overflow.i8(i8, i8)
