; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 < %s | FileCheck -check-prefix=HSA %s

@lds.align16.0 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 16
@lds.align16.1 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 16

@lds.align8.0 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 8
@lds.align32.0 = internal unnamed_addr addrspace(3) global [38 x i8] undef, align 32

@lds.missing.align.0 = internal unnamed_addr addrspace(3) global [39 x i32] undef
@lds.missing.align.1 = internal unnamed_addr addrspace(3) global [7 x i64] undef

declare void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* nocapture, i8 addrspace(1)* nocapture readonly, i32, i1) #0
declare void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* nocapture, i8 addrspace(3)* nocapture readonly, i32, i1) #0


; HSA-LABEL: {{^}}test_no_round_size_1:
; HSA: workgroup_group_segment_byte_size = 38
define amdgpu_kernel void @test_no_round_size_1(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.align16.0.bc, i8 addrspace(1)* align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.align16.0.bc, i32 38, i1 false)
  ret void
}

; There are two objects, so one requires padding to be correctly
; aligned after the other.

; (38 -> 48) + 38 = 92

; I don't think it is necessary to add padding after since if there
; were to be a dynamically sized LDS kernel arg, the runtime should
; add the alignment padding if necessary alignment padding if needed.

; HSA-LABEL: {{^}}test_round_size_2:
; HSA: workgroup_group_segment_byte_size = 86
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_2(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.align16.0.bc, i8 addrspace(1)* align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.align16.0.bc, i32 38, i1 false)

  %lds.align16.1.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.1 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.align16.1.bc, i8 addrspace(1)* align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.align16.1.bc, i32 38, i1 false)

  ret void
}

; 38 + (10 pad) + 38  (= 86)
; HSA-LABEL: {{^}}test_round_size_2_align_8:
; HSA: workgroup_group_segment_byte_size = 86
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_2_align_8(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  ret void
}

; HSA-LABEL: {{^}}test_round_local_lds_and_arg:
; HSA: workgroup_group_segment_byte_size = 38
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_local_lds_and_arg(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i8 addrspace(3)* %lds.arg) #1 {
  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.align16.0.bc, i8 addrspace(1)* align 4 %in, i32 38, i1 false)

  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.align16.0.bc, i32 38, i1 false)
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.arg, i8 addrspace(1)* align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.arg, i32 38, i1 false)
  ret void
}

; HSA-LABEL: {{^}}test_round_lds_arg:
; HSA: workgroup_group_segment_byte_size = 0
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_lds_arg(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i8 addrspace(3)* %lds.arg) #1 {
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.arg, i8 addrspace(1)* align 4 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.arg, i32 38, i1 false)
  ret void
}

; FIXME: Parameter alignment not considered
; HSA-LABEL: {{^}}test_high_align_lds_arg:
; HSA: workgroup_group_segment_byte_size = 0
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_high_align_lds_arg(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i8 addrspace(3)* align 64 %lds.arg) #1 {
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 64 %lds.arg, i8 addrspace(1)* align 64 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 64 %out, i8 addrspace(3)* align 64 %lds.arg, i32 38, i1 false)
  ret void
}

; (39 * 4) + (4 pad) + (7 * 8) = 216
; HSA-LABEL: {{^}}test_missing_alignment_size_2_order0:
; HSA: workgroup_group_segment_byte_size = 216
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_missing_alignment_size_2_order0(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.missing.align.0.bc = bitcast [39 x i32] addrspace(3)* @lds.missing.align.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.missing.align.0.bc, i8 addrspace(1)* align 4 %in, i32 160, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.missing.align.0.bc, i32 160, i1 false)

  %lds.missing.align.1.bc = bitcast [7 x i64] addrspace(3)* @lds.missing.align.1 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.missing.align.1.bc, i8 addrspace(1)* align 8 %in, i32 56, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.missing.align.1.bc, i32 56, i1 false)

  ret void
}

; (39 * 4) + (4 pad) + (7 * 8) = 216
; HSA-LABEL: {{^}}test_missing_alignment_size_2_order1:
; HSA: workgroup_group_segment_byte_size = 216
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_missing_alignment_size_2_order1(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.missing.align.1.bc = bitcast [7 x i64] addrspace(3)* @lds.missing.align.1 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.missing.align.1.bc, i8 addrspace(1)* align 8 %in, i32 56, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.missing.align.1.bc, i32 56, i1 false)

  %lds.missing.align.0.bc = bitcast [39 x i32] addrspace(3)* @lds.missing.align.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 4 %lds.missing.align.0.bc, i8 addrspace(1)* align 4 %in, i32 160, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 4 %out, i8 addrspace(3)* align 4 %lds.missing.align.0.bc, i32 160, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  ( = 134)
; HSA-LABEL: {{^}}test_round_size_3_order0:
; HSA: workgroup_group_segment_byte_size = 134
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_3_order0(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align32.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align32.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align32.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align32.0.bc, i32 38, i1 false)

  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 (+ 10 pad) + 38 + (10 pad) + 38 ( = 134)
; HSA-LABEL: {{^}}test_round_size_3_order1:
; HSA: workgroup_group_segment_byte_size = 134
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_3_order1(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align32.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align32.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align32.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align32.0.bc, i32 38, i1 false)

  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  ( = 126)
; HSA-LABEL: {{^}}test_round_size_3_order2:
; HSA: workgroup_group_segment_byte_size = 134
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_3_order2(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  %lds.align32.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align32.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align32.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align32.0.bc, i32 38, i1 false)

  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38 ( = 134)
; HSA-LABEL: {{^}}test_round_size_3_order3:
; HSA: workgroup_group_segment_byte_size = 134
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_3_order3(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  %lds.align32.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align32.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align32.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align32.0.bc, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  (= 134)
; HSA-LABEL: {{^}}test_round_size_3_order4:
; HSA: workgroup_group_segment_byte_size = 134
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_3_order4(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  %lds.align32.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align32.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align32.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align32.0.bc, i32 38, i1 false)

  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  ret void
}

; align 32, 16, 16
; 38 + (10 pad) + 38 + (10 pad) + 38  (= 134)
; HSA-LABEL: {{^}}test_round_size_3_order5:
; HSA: workgroup_group_segment_byte_size = 134
; HSA: group_segment_alignment = 4
define amdgpu_kernel void @test_round_size_3_order5(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
  %lds.align8.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align8.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align8.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align8.0.bc, i32 38, i1 false)

  %lds.align16.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align16.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align16.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align16.0.bc, i32 38, i1 false)

  %lds.align32.0.bc = bitcast [38 x i8] addrspace(3)* @lds.align32.0 to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p1i8.i32(i8 addrspace(3)* align 8 %lds.align32.0.bc, i8 addrspace(1)* align 8 %in, i32 38, i1 false)
  call void @llvm.memcpy.p1i8.p3i8.i32(i8 addrspace(1)* align 8 %out, i8 addrspace(3)* align 8 %lds.align32.0.bc, i32 38, i1 false)

  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind }
attributes #2 = { convergent nounwind }
