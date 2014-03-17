; RUN: opt < %s -S -globalopt | FileCheck %s
;; check that global opt annotates loads from global variales that have
;; constant values stored to them.

@G = internal global i32 5
@H = internal global i32 7
@I = internal global i32 17
@J = internal global i32 29
@K = internal global i32 31

define void @set() {
  store i32 6, i32* @G
  store i32 13, i32* @H
  store i32 16, i32* @I
  store i32 29, i32* @J
  store i32 -37, i32* @K
  ret void
}

define i32 @getG() {
; CHECK: %t = load i32* @G, !range [[G:![0-9]+]]
  %t = load i32* @G
  ret i32 %t
}
define i32 @getH() {
; CHECK: %t = load i32* @H, !range [[H:![0-9]+]]
  %t = load i32* @H
  ret i32 %t
}

define i32 @getI() {
; CHECK: %t = load i32* @I, !range [[I:![0-9]+]]
  %t = load i32* @I
  ret i32 %t
}

define i32 @getJ() {
; CHECK: ret i32 29
  %t = load i32* @J
  ret i32 %t
}

define i32 @getK() {
; CHECK: %t = load i32* @K, !range [[K:![0-9]+]]
  %t = load i32* @K
  ret i32 %t
}

; CHECK: [[G]] = metadata !{i32 5, i32 7}
; CHECK: [[H]] = metadata !{i32 7, i32 8, i32 13, i32 14}
; CHECK: [[I]] = metadata !{i32 16, i32 18}
; CHECK: [[K]] = metadata !{i32 -37, i32 -36, i32 31, i32 32}
