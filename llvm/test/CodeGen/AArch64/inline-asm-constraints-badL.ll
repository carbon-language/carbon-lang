; RUN: not llc -mtriple=aarch64-none-linux-gnu < %s

define void @foo() {
  ; 32-bit bitpattern ending in 1101 can't be produced.
  call void asm sideeffect "and x0, x0, $0", "L"(i32 13)
  ret void
}