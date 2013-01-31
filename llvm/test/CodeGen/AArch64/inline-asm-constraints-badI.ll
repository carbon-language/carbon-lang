; RUN: not llc -march=aarch64 < %s

define void @foo() {
  ; Out of range immediate for I.
  call void asm sideeffect "add x0, x0, $0", "I"(i32 4096)
  ret void
}