; RUN: llc -mtriple=thumbv7-linux-gnueabihf -o - -show-mc-encoding -t2-reduce-limit2=0 %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-linux-gnueabihf -o - -show-mc-encoding %s | FileCheck %s --check-prefix=CHECK-OPT

define i32 @and(i32 %a, i32 %b) nounwind readnone {
; CHECK-LABEL: and:
; CHECK: and.w r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}} @ encoding: [{{0x..,0x..,0x..,0x..}}]
; CHECK-OPT: ands r{{[0-7]}}, r{{[0-7]}} @ encoding: [{{0x..,0x..}}]
entry:
  %and = and i32 %b, %a
  ret i32 %and
}

