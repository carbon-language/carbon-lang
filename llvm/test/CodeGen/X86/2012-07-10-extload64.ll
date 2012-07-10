; RUN: llc < %s -march=x86 -mcpu=corei7 -mtriple=i686-pc-win32 | FileCheck %s

; CHECK: load_store
define void @load_store(<4 x i16>* %in) {
entry:
  %A27 = load <4 x i16>* %in, align 4
  %A28 = add <4 x i16> %A27, %A27
  store <4 x i16> %A28, <4 x i16>* %in, align 4
  ret void
; CHECK: movd
; CHECK: pinsrd
; CHECK: ret
}
