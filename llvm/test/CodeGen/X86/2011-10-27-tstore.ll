; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

;CHECK: ltstore
;CHECK: pshufd
;CHECK: pshufd
;CHECK: ret
define void @ltstore() {
entry:
  %in = load <4 x i32>* undef
  %j = shufflevector <4 x i32> %in, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  store <2 x i32> %j, <2 x i32>* undef
  ret void
}

