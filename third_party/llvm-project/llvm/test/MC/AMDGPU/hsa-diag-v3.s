// RUN: not llvm-mc --amdhsa-code-object-version=3 -triple amdgcn-amd-amdhsa -mcpu=gfx810 -mattr=+xnack -show-encoding %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=GCN,GFX8,NONGFX10,AMDHSA
// RUN: not llvm-mc --amdhsa-code-object-version=3 -triple amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+xnack -show-encoding %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=GCN,GFX10,AMDHSA
// RUN: not llvm-mc --amdhsa-code-object-version=3 -triple amdgcn-amd- -mcpu=gfx810 -mattr=+xnack -show-encoding %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=GCN,NONAMDHSA
// RUN: not llvm-mc --amdhsa-code-object-version=3 -triple amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+xnack -show-encoding %s 2>&1 >/dev/null | FileCheck %s --check-prefixes=GFX90A,NONGFX10,AMDHSA,ALL

.text

// GCN-LABEL: warning: test_target
// GFX8-NOT: error:
// GFX10: error: .amdgcn_target directive's target id amdgcn-amd-amdhsa--gfx810+xnack does not match the specified target id amdgcn-amd-amdhsa--gfx1010+xnack
// NONAMDHSA: error: .amdgcn_target directive's target id amdgcn-amd-amdhsa--gfx810+xnack does not match the specified target id amdgcn-amd-unknown--gfx810
.warning "test_target"
.amdgcn_target "amdgcn-amd-amdhsa--gfx810+xnack"

// GCN-LABEL: warning: test_amdhsa_kernel_no_name
// GCN: error: unknown directive
.warning "test_amdhsa_kernel_no_name"
.amdhsa_kernel
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_kernel_empty
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_kernel_empty"
.amdhsa_kernel test_amdhsa_kernel_empty
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_kernel_unknown_directive
// AMDHSA: error: expected .amdhsa_ directive or .end_amdhsa_kernel
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_kernel_unknown_directive"
.amdhsa_kernel test_amdhsa_kernel_unknown_directive
  1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_group_segment_fixed_size_invalid_size
// AMDHSA: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_group_segment_fixed_size_invalid_size"
.amdhsa_kernel test_amdhsa_group_segment_fixed_size_invalid_size
  .amdhsa_group_segment_fixed_size -1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_group_segment_fixed_size_invalid_expression
// AMDHSA: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_group_segment_fixed_size_invalid_expression"
.amdhsa_kernel test_amdhsa_group_segment_fixed_size_invalid_expression
  .amdhsa_group_segment_fixed_size 10000000000 + 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_group_segment_fixed_size_repeated
// AMDHSA: error: .amdhsa_ directives cannot be repeated
// NONAMDHSA-: error: unknown directive
.warning "test_amdhsa_group_segment_fixed_size_repeated"
.amdhsa_kernel test_amdhsa_group_segment_fixed_size_repeated
  .amdhsa_group_segment_fixed_size 1
  .amdhsa_group_segment_fixed_size 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_next_free_vgpr_missing
// AMDHSA: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_next_free_vgpr_missing"
.amdhsa_kernel test_amdhsa_next_free_vgpr_missing
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_next_free_sgpr_missing
// AMDHSA: error: .amdhsa_next_free_sgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_next_free_sgpr_missing"
.amdhsa_kernel test_amdhsa_next_free_sgpr_missing
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_accum_offset
// NONGFX9A: error: directive requires gfx90a+
// GFX90A: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_accum_offset"
.amdhsa_kernel test_amdhsa_accum_offset
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_accum_offset_missing
// NONGFX9A: error: directive requires gfx90a+
// GFX90A: error: .amdhsa_accum_offset directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_accum_offset_missing"
.amdhsa_kernel test_amdhsa_accum_offset_missing
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_accum_offset_invalid0
// NONGFX9A: error: directive requires gfx90a+
// GFX90A: error: accum_offset should be in range [4..256] in increments of 4
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_accum_offset_invalid0"
.amdhsa_kernel test_amdhsa_accum_offset_invalid0
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_accum_offset 0
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_accum_offset_invalid5
// NONGFX9A: error: directive requires gfx90a+
// GFX90A: error: accum_offset should be in range [4..256] in increments of 4
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_accum_offset_invalid5"
.amdhsa_kernel test_amdhsa_accum_offset_invalid5
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_accum_offset 5
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_accum_offset_invalid257
// NONGFX9A: error: directive requires gfx90a+
// GFX90A: error: accum_offset should be in range [4..256] in increments of 4
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_accum_offset_invalid257"
.amdhsa_kernel test_amdhsa_accum_offset_invalid257
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_accum_offset 257
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_accum_offset_invalid8
// NONGFX9A: error: directive requires gfx90a+
// GFX90A: error: accum_offset exceeds total VGPR allocation
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_accum_offset_invalid8"
.amdhsa_kernel test_amdhsa_accum_offset_invalid8
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
  .amdhsa_accum_offset 8
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_tg_split
// NONGFX90A: error: directive requires gfx90a+
// GFX90A: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_tg_split"
.amdhsa_kernel test_amdhsa_tg_split
  .amdhsa_tg_split 1
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_tg_split_invalid
// NONGFX90A: error: directive requires gfx90a+
// GFX90A: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_tg_split_invalid"
.amdhsa_kernel test_amdhsa_tg_split_invalid
  .amdhsa_tg_split 5
.end_amdhsa_kernel

// ALL-LABEL: warning: test_amdhsa_wavefront_size32
// NONGFX10: error: directive requires gfx10+
// GFX10: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_wavefront_size32"
.amdhsa_kernel test_amdhsa_wavefront_size32
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_wavefront_size32_invalid
// NONGFX10: error: directive requires gfx10+
// GFX10: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_wavefront_size32_invalid"
.amdhsa_kernel test_amdhsa_wavefront_size32_invalid
  .amdhsa_wavefront_size32 5
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_workgroup_processor_mode
// NONGFX10: error: directive requires gfx10+
// GFX10: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_workgroup_processor_mode"
.amdhsa_kernel test_amdhsa_workgroup_processor_mode
  .amdhsa_workgroup_processor_mode 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_workgroup_processor_mode_invalid
// NONGFX10: error: directive requires gfx10+
// GFX10: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_workgroup_processor_mode_invalid"
.amdhsa_kernel test_amdhsa_workgroup_processor_mode_invalid
  .amdhsa_workgroup_processor_mode 5
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_memory_ordered
// NONGFX10: error: directive requires gfx10+
// GFX10: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_memory_ordered"
.amdhsa_kernel test_amdhsa_memory_ordered
  .amdhsa_memory_ordered 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_memory_ordered_invalid
// NONGFX10: error: directive requires gfx10+
// GFX10: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_memory_ordered_invalid"
.amdhsa_kernel test_amdhsa_memory_ordered_invalid
  .amdhsa_memory_ordered 5
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_forward_progress
// NONGFX10: error: directive requires gfx10+
// GFX10: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_forward_progress"
.amdhsa_kernel test_amdhsa_forward_progress
  .amdhsa_forward_progress 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_forward_progress_invalid
// NONGFX10: error: directive requires gfx10+
// GFX10: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_forward_progress_invalid"
.amdhsa_kernel test_amdhsa_forward_progress_invalid
  .amdhsa_forward_progress 5
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_shared_vgpr_count_invalid1
// NONGFX10: error: directive requires gfx10+
// GFX10: error: .amdhsa_next_free_vgpr directive is required
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_shared_vgpr_count_invalid1"
.amdhsa_kernel test_amdhsa_shared_vgpr_count_invalid1
  .amdhsa_shared_vgpr_count 8
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_shared_vgpr_count_invalid2
// NONGFX10: error: directive requires gfx10+
// GFX10: error: shared_vgpr_count directive not valid on wavefront size 32
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_shared_vgpr_count_invalid2"
.amdhsa_kernel test_amdhsa_shared_vgpr_count_invalid2
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 0
  .amdhsa_shared_vgpr_count 8
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_shared_vgpr_count_invalid3
// NONGFX10: error: directive requires gfx10+
// GFX10: error: value out of range
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_shared_vgpr_count_invalid3"
.amdhsa_kernel test_amdhsa_shared_vgpr_count_invalid3
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 0
  .amdhsa_shared_vgpr_count 16
.end_amdhsa_kernel

// GCN-LABEL: warning: test_amdhsa_shared_vgpr_count_invalid4
// NONGFX10: error: directive requires gfx10+
// GFX10: error: shared_vgpr_count*2 + compute_pgm_rsrc1.GRANULATED_WORKITEM_VGPR_COUNT cannot exceed 63
// NONAMDHSA: error: unknown directive
.warning "test_amdhsa_shared_vgpr_count_invalid4"
.amdhsa_kernel test_amdhsa_shared_vgpr_count_invalid4
  .amdhsa_next_free_vgpr 273
  .amdhsa_next_free_sgpr 0
  .amdhsa_shared_vgpr_count 15
.end_amdhsa_kernel

// GCN-LABEL: warning: test_next_free_vgpr_invalid
// AMDHSA: error: .amdgcn.next_free_{v,s}gpr symbols must be absolute expressions
// NONAMDHSA-NOT: error:
.warning "test_next_free_vgpr_invalid"
.set .amdgcn.next_free_vgpr, "foo"
v_mov_b32_e32 v0, s0

// GCN-LABEL: warning: test_end
.warning "test_end"
