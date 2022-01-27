; RUN: opt -S -mtriple=arm-unknown-linux-gnu -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml -lowertypetests-write-summary=%t < %s | FileCheck %s

target datalayout = "e-p:64:64"

define void @f1() "target-features"="+thumb-mode" !type !0 {
  ret void
}

define void @g1() "target-features"="-thumb-mode" !type !0 {
  ret void
}

define void @f2() "target-features"="+thumb-mode" !type !1 {
  ret void
}

define void @g2() "target-features"="-thumb-mode" !type !1 {
  ret void
}

define void @h2() "target-features"="-thumb-mode" !type !1 {
  ret void
}

declare void @takeaddr(void()*, void()*, void()*, void()*, void()*)
define void @addrtaken() {
  call void @takeaddr(void()* @f1, void()* @g1, void()* @f2, void()* @g2, void()* @h2)
  ret void
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

; CHECK: define private void {{.*}} #[[AT:.*]] align 4 {
; CHECK-NEXT: entry:
; CHECK-NEXT:  call void asm sideeffect "b.w $0\0Ab.w $1\0A", "s,s"(void ()* @f1.cfi, void ()* @g1.cfi)
; CHECK-NEXT:  unreachable
; CHECK-NEXT: }

; CHECK: define private void {{.*}} #[[AA:.*]] align 4 {
; CHECK-NEXT: entry:
; CHECK-NEXT:  call void asm sideeffect "b $0\0Ab $1\0Ab $2\0A", "s,s,s"(void ()* @f2.cfi, void ()* @g2.cfi, void ()* @h2.cfi)
; CHECK-NEXT:  unreachable
; CHECK-NEXT: }

; CHECK-DAG: attributes #[[AA]] = { naked nounwind "target-features"="-thumb-mode" }
; CHECK-DAG: attributes #[[AT]] = { naked nounwind "target-cpu"="cortex-a8" "target-features"="+thumb-mode" }
