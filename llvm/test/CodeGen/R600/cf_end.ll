; RUN: llc < %s -march=r600 -mcpu=redwood --show-mc-encoding | FileCheck %s

; CHECK: CF_END ; encoding: [0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x80]
define void @eop() {
  ret void
}
