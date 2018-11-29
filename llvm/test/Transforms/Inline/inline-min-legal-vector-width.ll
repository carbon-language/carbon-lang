; RUN: opt %s -inline -S | FileCheck %s

define internal void @innerSmall() "min-legal-vector-width"="128" {
  ret void
}

define internal void @innerLarge() "min-legal-vector-width"="512" {
  ret void
}

define void @outerNoAttribute() {
  call void @innerLarge()
  ret void
}

define void @outerConflictingAttributeSmall() "min-legal-vector-width"="128" {
  call void @innerLarge()
  ret void
}

define void @outerConflictingAttributeLarge() "min-legal-vector-width"="512" {
  call void @innerSmall()
  ret void
}

; CHECK: define void @outerNoAttribute() #0
; CHECK: define void @outerConflictingAttributeSmall() #0
; CHECK: define void @outerConflictingAttributeLarge() #0
; CHECK: attributes #0 = { "min-legal-vector-width"="512" }
