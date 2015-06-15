; RUN: llc -march=hexagon -filetype=obj -o - < %s | llvm-objdump -d -r - | FileCheck %s

declare void @bar(i32);

define void @foo(i32 %a) {
  %b = mul i32 %a, 3
  call void @bar(i32 %b)
  ret void
}
; CHECK:     0x8 R_HEX_B22_PCREL - 0x4
