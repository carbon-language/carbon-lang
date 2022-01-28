; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -verify-machineinstrs | FileCheck %s

; Test that phis are lowered.

target triple = "wasm32-unknown-unknown"

; Basic phi triangle.

; CHECK-LABEL: test0:
; CHECK: div_s $[[NUM0:[0-9]+]]=, $0, $pop[[NUM1:[0-9]+]]{{$}}
; CHECK: return $[[NUM0]]{{$}}
define i32 @test0(i32 %p) {
entry:
  %t = icmp slt i32 %p, 0
  br i1 %t, label %true, label %done
true:
  %a = sdiv i32 %p, 3
  br label %done
done:
  %s = phi i32 [ %a, %true ], [ %p, %entry ]
  ret i32 %s
}

; Swap phis.

; CHECK-LABEL: test1:
; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: local.copy $[[NUM0:[0-9]+]]=, $[[NUM1:[0-9]+]]{{$}}
; CHECK: local.copy $[[NUM1]]=, $[[NUM2:[0-9]+]]{{$}}
; CHECK: local.copy $[[NUM2]]=, $[[NUM0]]{{$}}
define i32 @test1(i32 %n) {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %b, %loop ]
  %b = phi i32 [ 1, %entry ], [ %a, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  %i.next = add i32 %i, 1
  %t = icmp slt i32 %i.next, %n
  br i1 %t, label %loop, label %exit

exit:
  ret i32 %a
}
