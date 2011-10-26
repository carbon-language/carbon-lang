; RUN: llc < %s -mtriple=armv7-unknown-linux-eabi | FileCheck %s

; Check that when optimizing for size, a literal pool load is used
; instead of the (potentially faster) movw/movt pair when loading
; a large constant.

@x = global i32* inttoptr (i32 305419888 to i32*), align 4

define i32 @f() optsize {
  ; CHECK: f:
  ; CHECK: ldr  r{{.}}, {{.?}}LCPI{{.}}_{{.}}
  ; CHECK: ldr  r{{.}}, [{{(pc, )?}}r{{.}}]
  ; CHECK: ldr  r{{.}}, [r{{.}}]
  %1 = load i32** @x, align 4
  %2 = load i32* %1
  ret i32 %2
}

define i32 @g() {
  ; CHECK: g:
  ; CHECK: movw
  ; CHECK: movt
  %1 = load i32** @x, align 4
  %2 = load i32* %1
  ret i32 %2
}
