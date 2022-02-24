; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test1:
; CHECK-NOT: s_waitcnt
; CHECK: image_store
; CHECK-NEXT: s_waitcnt vmcnt(0) expcnt(0){{$}}
; CHECK-NEXT: image_store
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test1(<8 x i32> inreg %rsrc, <4 x float> %d0, <4 x float> %d1, i32 %c0, i32 %c1) {
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %d0, i32 15, i32 %c0, <8 x i32> %rsrc, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 3840) ; 0xf00
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %d1, i32 15, i32 %c1, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; Test that the intrinsic is merged with automatically generated waits and
; emitted as late as possible.
;
; CHECK-LABEL: {{^}}test2:
; CHECK-NOT: s_waitcnt
; CHECK: image_load
; CHECK-NEXT: v_lshlrev_b32
; CHECK-NEXT: s_waitcnt vmcnt(0) expcnt(0){{$}}
; CHECK-NEXT: image_store
define amdgpu_ps void @test2(<8 x i32> inreg %rsrc, i32 %c) {
  %t = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %c, <8 x i32> %rsrc, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 3840) ; 0xf00
  %c.1 = mul i32 %c, 2
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %t, i32 15, i32 %c.1, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; CHECK-LABEL: {{^}}test3:
; CHECK: image_load
; CHECK: s_waitcnt vmcnt(0) lgkmcnt(0)
; CHECK: image_store
define amdgpu_ps void @test3(<8 x i32> inreg %rsrc, i32 %c) {
  %t = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %c, <8 x i32> %rsrc, i32 0, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 49279) ; not isInt<16>, but isUInt<16>
  %c.1 = mul i32 %c, 2
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %t, i32 15, i32 %c.1, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.s.waitcnt(i32) #0

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
