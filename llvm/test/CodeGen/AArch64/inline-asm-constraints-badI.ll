; RUN: not llc -mtriple=aarch64-none-linux-gnu < %s

define void @foo() {
  ; Out of range immediate for I.
  call void asm sideeffect "add x0, x0, $0", "I"(i32 4096)
  ret void
}