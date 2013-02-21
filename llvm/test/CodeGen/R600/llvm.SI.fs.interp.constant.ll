;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

;CHECK: S_MOV_B32
;CHECK-NEXT: V_INTERP_MOV_F32

define void @main() {
main_body:
  call void @llvm.AMDGPU.shader.type(i32 0)
  %0 = load i32 addrspace(8)* inttoptr (i32 6 to i32 addrspace(8)*)
  %1 = call float @llvm.SI.fs.interp.constant(i32 0, i32 0, i32 %0)
  %2 = call i32 @llvm.SI.packf16(float %1, float %1)
  %3 = bitcast i32 %2 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %3, float %3, float %3, float %3)
  ret void
}

declare void @llvm.AMDGPU.shader.type(i32)

declare float @llvm.SI.fs.interp.constant(i32, i32, i32) readonly

declare i32 @llvm.SI.packf16(float, float) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
