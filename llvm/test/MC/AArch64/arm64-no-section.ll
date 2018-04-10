; RUN: llc -mtriple=aarch64-darwin-- -filetype=obj %s -o -
; CHECK: <inline asm>:1:2: error: unsupported relocation of local symbol
; CHECK-SAME: 'L_foo_end'. Must have non-local symbol earlier in section.

; Make sure that we emit an error when we try to reference something that
; doesn't belong to a section.
define void @foo() local_unnamed_addr {
  tail call void asm sideeffect "b L_foo_end\0A", ""()
  ret void
}
