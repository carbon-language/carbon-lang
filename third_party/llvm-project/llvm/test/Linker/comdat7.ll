; RUN: not llvm-link %s %s -S -o - 2>&1 | FileCheck %s

$c1 = comdat largest

define void @c1() comdat($c1) {
  ret void
}
; CHECK: GlobalVariable required for data dependent selection!
