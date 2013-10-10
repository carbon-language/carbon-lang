; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s

; This test just checks that the compiler doesn't crash.
; CHECK-LABEL: @v32i8_to_v8i32
; CHECK: S_ENDPGM

define void @v32i8_to_v8i32(<32 x i8> addrspace(2)* inreg) #0 {
entry:
  %1 = load <32 x i8> addrspace(2)* %0
  %2 = bitcast <32 x i8> %1 to <8 x i32>
  %3 = extractelement <8 x i32> %2, i32 1
  %4 = icmp ne i32 %3, 0
  %5 = select i1 %4, float 0.0, float 1.0
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %5, float %5, float %5, float %5)
  ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }

