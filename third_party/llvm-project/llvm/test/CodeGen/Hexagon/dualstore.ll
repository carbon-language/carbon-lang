; RUN: llc -march=hexagon -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s
; Check that we generate dual stores in one packet in V4

; CHECK: 00 40 9f 52 529f4000
; CHECK: 10 10 00 f0 f0001010

define void @foo(i32* %a, i32* %b) {
  store i32 0, i32* %a
  store i32 0, i32* %b
  ret void
}
