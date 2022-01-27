; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; The first image store and the second image load use the same descriptor and
; the same coordinate. Check that they do not get swapped by the machine
; instruction scheduler.

; GCN-LABEL: {{^}}_amdgpu_cs_main:
; GCN: image_load
; GCN: image_store
; GCN: image_load
; GCN: image_store

define dllexport amdgpu_cs void @_amdgpu_cs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <3 x i32> inreg %arg3, i32 inreg %arg4, <3 x i32> %arg5) local_unnamed_addr #0 {
.entry:
  %tmp = call i64 @llvm.amdgcn.s.getpc() #1
  %tmp6 = bitcast i64 %tmp to <2 x i32>
  %.0.vec.insert = insertelement <2 x i32> undef, i32 %arg2, i32 0
  %.4.vec.insert = shufflevector <2 x i32> %.0.vec.insert, <2 x i32> %tmp6, <2 x i32> <i32 0, i32 3>
  %tmp7 = bitcast <2 x i32> %.4.vec.insert to i64
  %tmp8 = inttoptr i64 %tmp7 to [4294967295 x i8] addrspace(4)*
  %tmp9 = add <3 x i32> %arg3, %arg5
  %tmp10 = getelementptr [4294967295 x i8], [4294967295 x i8] addrspace(4)* %tmp8, i64 0, i64 32
  %tmp11 = bitcast i8 addrspace(4)* %tmp10 to <8 x i32> addrspace(4)*, !amdgpu.uniform !0
  %tmp12 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp11, align 16
  %tmp13.0 = extractelement <3 x i32> %tmp9, i32 0
  %tmp13.1 = extractelement <3 x i32> %tmp9, i32 1
  %tmp14 = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 %tmp13.0, i32 %tmp13.1, <8 x i32> %tmp12, i32 0, i32 0) #0
  %tmp15 = inttoptr i64 %tmp7 to <8 x i32> addrspace(4)*
  %tmp16 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp15, align 16
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %tmp14, i32 15, i32 %tmp13.0, i32 %tmp13.1, <8 x i32> %tmp16, i32 0, i32 0) #0
  %tmp17 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp15, align 16
  %tmp18 = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 165, i32 %tmp13.0, i32 %tmp13.1, <8 x i32> %tmp17, i32 0, i32 0) #0
  %tmp19 = getelementptr [4294967295 x i8], [4294967295 x i8] addrspace(4)* %tmp8, i64 0, i64 64
  %tmp20 = bitcast i8 addrspace(4)* %tmp19 to <8 x i32> addrspace(4)*, !amdgpu.uniform !0
  %tmp21 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp20, align 16
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %tmp18, i32 15, i32 %tmp13.0, i32 %tmp13.1, <8 x i32> %tmp21, i32 0, i32 0) #0
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.amdgcn.s.getpc() #1

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #2

; Function Attrs: nounwind writeonly
declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32, i32, i32, <8 x i32>, i32, i32) #3

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind writeonly }

!0 = !{}
