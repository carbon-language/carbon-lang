; RUN: llc -march=hexagon -filetype=obj < %s | llvm-readobj -file-headers | FileCheck %s

; CHECK: OS/ABI: SystemV (0x0)
define void @foo() {
  ret void
}

