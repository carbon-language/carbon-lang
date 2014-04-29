; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s

; Function Attrs: nounwind
define void @foo() {
entry:
  ret void
; CHECK: jr	$ra
}
