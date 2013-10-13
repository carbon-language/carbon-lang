; RUN: llc < %s -march=r600 -mcpu=SI --verify-machineinstrs | FileCheck %s

;CHECK-LABEL: @main
;CHECK: S_WAITCNT lgkmcnt(0)
;CHECK: S_WAITCNT vmcnt(0)
;CHECK: S_WAITCNT expcnt(0) lgkmcnt(0)

define void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, i32 inreg, i32, i32, i32, i32) #0 {
main_body:
  %10 = getelementptr <16 x i8> addrspace(2)* %3, i32 0
  %11 = load <16 x i8> addrspace(2)* %10, !tbaa !0
  %12 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %11, i32 0, i32 %6)
  %13 = extractelement <4 x float> %12, i32 0
  %14 = extractelement <4 x float> %12, i32 1
  %15 = extractelement <4 x float> %12, i32 2
  %16 = extractelement <4 x float> %12, i32 3
  %17 = getelementptr <16 x i8> addrspace(2)* %3, i32 1
  %18 = load <16 x i8> addrspace(2)* %17, !tbaa !0
  %19 = call <4 x float> @llvm.SI.vs.load.input(<16 x i8> %18, i32 0, i32 %6)
  %20 = extractelement <4 x float> %19, i32 0
  %21 = extractelement <4 x float> %19, i32 1
  %22 = extractelement <4 x float> %19, i32 2
  %23 = extractelement <4 x float> %19, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %20, float %21, float %22, float %23)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %13, float %14, float %15, float %16)
  ret void
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.SI.vs.load.input(<16 x i8>, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { nounwind readnone }

!0 = metadata !{metadata !"const", null, i32 1}
