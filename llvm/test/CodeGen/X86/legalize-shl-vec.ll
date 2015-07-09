; RUN: llc < %s -march=x86-64 | FileCheck %s

define <2 x i256> @test_shl(<2 x i256> %In) {
  %Amt = insertelement <2 x i256> undef, i256 -1, i32 0
  %Out = shl <2 x i256> %In, %Amt
  ret <2 x i256> %Out

  ; CHECK-LABEL: test_shl
  ; CHECK:       movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK:       retq
}

define <2 x i256> @test_srl(<2 x i256> %In) {
  %Amt = insertelement <2 x i256> undef, i256 -1, i32 0
  %Out = lshr <2 x i256> %In, %Amt
  ret <2 x i256> %Out

  ; CHECK-LABEL: test_srl
  ; CHECK:       movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK-NEXT:  movq $0
  ; CHECK:       retq
}

define <2 x i256> @test_sra(<2 x i256> %In) {
  %Amt = insertelement <2 x i256> undef, i256 -1, i32 0
  %Out = ashr <2 x i256> %In, %Amt
  ret <2 x i256> %Out

  ; CHECK-LABEL: test_sra
  ; CHECK:       sarq $63
}
