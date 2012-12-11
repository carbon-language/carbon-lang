;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

; CHECK: S_ENDPGM

define void @main() {
main_body:
  call void @llvm.AMDGPU.shader.type(i32 1)
  %0 = load <4 x i32> addrspace(2)* addrspace(8)* inttoptr (i32 6 to <4 x i32> addrspace(2)* addrspace(8)*)
  %1 = getelementptr <4 x i32> addrspace(2)* %0, i32 0
  %2 = load <4 x i32> addrspace(2)* %1
  %3 = call i32 @llvm.SI.vs.load.buffer.index()
  %4 = call <4 x float> @llvm.SI.vs.load.input(<4 x i32> %2, i32 0, i32 %3)
  %5 = extractelement <4 x float> %4, i32 0
  %6 = extractelement <4 x float> %4, i32 1
  %7 = extractelement <4 x float> %4, i32 2
  %8 = extractelement <4 x float> %4, i32 3
  %9 = load <4 x i32> addrspace(2)* addrspace(8)* inttoptr (i32 6 to <4 x i32> addrspace(2)* addrspace(8)*)
  %10 = getelementptr <4 x i32> addrspace(2)* %9, i32 1
  %11 = load <4 x i32> addrspace(2)* %10
  %12 = call i32 @llvm.SI.vs.load.buffer.index()
  %13 = call <4 x float> @llvm.SI.vs.load.input(<4 x i32> %11, i32 0, i32 %12)
  %14 = extractelement <4 x float> %13, i32 0
  %15 = extractelement <4 x float> %13, i32 1
  %16 = extractelement <4 x float> %13, i32 2
  %17 = extractelement <4 x float> %13, i32 3
  call void @llvm.SI.export(i32 15, i32 0, i32 0, i32 32, i32 0, float %14, float %15, float %16, float %17)
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %5, float %6, float %7, float %8)
  ret void
}

declare void @llvm.AMDGPU.shader.type(i32)

declare i32 @llvm.SI.vs.load.buffer.index() readnone

declare <4 x float> @llvm.SI.vs.load.input(<4 x i32>, i32, i32)

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
