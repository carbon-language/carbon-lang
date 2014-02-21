; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

; CHECK: D a1
; CHECK-NEXT: d a2
; CHECK-NEXT: T f1
; CHECK-NEXT: t f2
; CHECK-NEXT: W f3
; CHECK-NEXT: U f4
; CHECK-NEXT: D g1
; CHECK-NEXT: d g2
; CHECK-NEXT: C g3
; CHECK-NOT: g4

@g1 = global i32 42
@g2 = internal global i32 42
@g3 = common global i32 0
@g4 = private global i32 42

@a1 = alias i32* @g1
@a2 = alias internal i32* @g1

define void @f1() {
  ret void
}

define internal void @f2() {
  ret void
}

define linkonce_odr void @f3() {
  ret void
}

declare void @f4()
