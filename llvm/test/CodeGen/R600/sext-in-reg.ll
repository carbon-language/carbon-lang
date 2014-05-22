; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.imax(i32, i32) nounwind readnone


; FUNC-LABEL: @sext_in_reg_i1_i32
; SI: S_LOAD_DWORD [[ARG:s[0-9]+]],
; SI: S_BFE_I32 [[SEXTRACT:s[0-9]+]], [[ARG]], 0x10000
; SI: V_MOV_B32_e32 [[EXTRACT:v[0-9]+]], [[SEXTRACT]]
; SI: BUFFER_STORE_DWORD [[EXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]], {{.*}}, 0.0, 1
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i1_i32(i32 addrspace(1)* %out, i32 %in) {
  %shl = shl i32 %in, 31
  %sext = ashr i32 %shl, 31
  store i32 %sext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @sext_in_reg_i8_to_i32
; SI: S_ADD_I32 [[VAL:s[0-9]+]],
; SI: S_SEXT_I32_I8 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: V_MOV_B32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; SI: BUFFER_STORE_DWORD [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i8_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %shl = shl i32 %c, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @sext_in_reg_i16_to_i32
; SI: S_ADD_I32 [[VAL:s[0-9]+]],
; SI: S_SEXT_I32_I16 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: V_MOV_B32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; SI: BUFFER_STORE_DWORD [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i16_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %shl = shl i32 %c, 16
  %ashr = ashr i32 %shl, 16
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @sext_in_reg_i8_to_v1i32
; SI: S_ADD_I32 [[VAL:s[0-9]+]],
; SI: S_SEXT_I32_I8 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: V_MOV_B32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; SI: BUFFER_STORE_DWORD [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %c = add <1 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <1 x i32> %c, <i32 24>
  %ashr = ashr <1 x i32> %shl, <i32 24>
  store <1 x i32> %ashr, <1 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @sext_in_reg_i1_to_i64
; SI: S_ADD_I32 [[VAL:s[0-9]+]],
; SI: S_BFE_I32 s{{[0-9]+}}, s{{[0-9]+}}, 0x10000
; SI: S_MOV_B32 {{s[0-9]+}}, -1
; SI: BUFFER_STORE_DWORDX2
define void @sext_in_reg_i1_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = add i64 %a, %b
  %shl = shl i64 %c, 63
  %ashr = ashr i64 %shl, 63
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_i8_to_i64
; SI: S_ADD_I32 [[VAL:s[0-9]+]],
; SI: S_SEXT_I32_I8 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: S_MOV_B32 {{s[0-9]+}}, -1
; SI: BUFFER_STORE_DWORDX2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES_LO:T[0-9]+\.[XYZW]]], [[ADDR_LO:T[0-9]+.[XYZW]]]
; EG: MEM_{{.*}} STORE_{{.*}} [[RES_HI:T[0-9]+\.[XYZW]]], [[ADDR_HI:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT {{\*?}} [[RES_LO]], {{.*}}, 0.0, literal
; EG: ASHR [[RES_HI]]
; EG-NOT: BFE_INT
; EG: LSHR
; EG: LSHR
;; TODO Check address computation, using | with variables in {{}} does not work,
;; also the _LO/_HI order might be different
define void @sext_in_reg_i8_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = add i64 %a, %b
  %shl = shl i64 %c, 56
  %ashr = ashr i64 %shl, 56
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_i16_to_i64
; SI: S_ADD_I32 [[VAL:s[0-9]+]],
; SI: S_SEXT_I32_I16 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: S_MOV_B32 {{s[0-9]+}}, -1
; SI: BUFFER_STORE_DWORDX2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES_LO:T[0-9]+\.[XYZW]]], [[ADDR_LO:T[0-9]+.[XYZW]]]
; EG: MEM_{{.*}} STORE_{{.*}} [[RES_HI:T[0-9]+\.[XYZW]]], [[ADDR_HI:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT {{\*?}} [[RES_LO]], {{.*}}, 0.0, literal
; EG: ASHR [[RES_HI]]
; EG-NOT: BFE_INT
; EG: LSHR
; EG: LSHR
;; TODO Check address computation, using | with variables in {{}} does not work,
;; also the _LO/_HI order might be different
define void @sext_in_reg_i16_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = add i64 %a, %b
  %shl = shl i64 %c, 48
  %ashr = ashr i64 %shl, 48
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_i32_to_i64
; SI: S_LOAD_DWORD
; SI: S_LOAD_DWORD
; SI: S_ADD_I32 [[ADD:s[0-9]+]],
; SI: S_ASHR_I32 s{{[0-9]+}}, [[ADD]], 31
; SI: BUFFER_STORE_DWORDX2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES_LO:T[0-9]+\.[XYZW]]], [[ADDR_LO:T[0-9]+.[XYZW]]]
; EG: MEM_{{.*}} STORE_{{.*}} [[RES_HI:T[0-9]+\.[XYZW]]], [[ADDR_HI:T[0-9]+.[XYZW]]]
; EG-NOT: BFE_INT
; EG: ADD_INT {{\*?}} [[RES_LO]]
; EG: ASHR [[RES_HI]]
; EG: ADD_INT
; EG: LSHR
; EG: LSHR
;; TODO Check address computation, using | with variables in {{}} does not work,
;; also the _LO/_HI order might be different
define void @sext_in_reg_i32_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = add i64 %a, %b
  %shl = shl i64 %c, 32
  %ashr = ashr i64 %shl, 32
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; This is broken on Evergreen for some reason related to the <1 x i64> kernel arguments.
; XFUNC-LABEL: @sext_in_reg_i8_to_v1i64
; XSI: S_BFE_I32 [[EXTRACT:s[0-9]+]], {{s[0-9]+}}, 524288
; XSI: S_ASHR_I32 {{v[0-9]+}}, [[EXTRACT]], 31
; XSI: BUFFER_STORE_DWORD
; XEG: BFE_INT
; XEG: ASHR
; define void @sext_in_reg_i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i64> %a, <1 x i64> %b) nounwind {
;   %c = add <1 x i64> %a, %b
;   %shl = shl <1 x i64> %c, <i64 56>
;   %ashr = ashr <1 x i64> %shl, <i64 56>
;   store <1 x i64> %ashr, <1 x i64> addrspace(1)* %out, align 8
;   ret void
; }

; FUNC-LABEL: @sext_in_reg_i1_in_i32_other_amount
; SI-NOT: BFE
; SI: S_LSHL_B32 [[REG:s[0-9]+]], {{s[0-9]+}}, 6
; SI: S_ASHR_I32 {{s[0-9]+}}, [[REG]], 7

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_i1_in_i32_other_amount(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b
  %x = shl i32 %c, 6
  %y = ashr i32 %x, 7
  store i32 %y, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @sext_in_reg_v2i1_in_v2i32_other_amount
; SI: S_LSHL_B32 [[REG0:s[0-9]+]], {{s[0-9]}}, 6
; SI: S_ASHR_I32 {{s[0-9]+}}, [[REG0]], 7
; SI: S_LSHL_B32 [[REG1:s[0-9]+]], {{s[0-9]}}, 6
; SI: S_ASHR_I32 {{s[0-9]+}}, [[REG1]], 7

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i1_in_v2i32_other_amount(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b
  %x = shl <2 x i32> %c, <i32 6, i32 6>
  %y = ashr <2 x i32> %x, <i32 7, i32 7>
  store <2 x i32> %y, <2 x i32> addrspace(1)* %out, align 2
  ret void
}


; FUNC-LABEL: @sext_in_reg_v2i1_to_v2i32
; SI: S_BFE_I32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: S_BFE_I32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: BUFFER_STORE_DWORDX2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i1_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 31, i32 31>
  %ashr = ashr <2 x i32> %shl, <i32 31, i32 31>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_v4i1_to_v4i32
; SI: S_BFE_I32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: S_BFE_I32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: S_BFE_I32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: S_BFE_I32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: BUFFER_STORE_DWORDX4

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW][XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v4i1_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %c = add <4 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 31, i32 31, i32 31, i32 31>
  %ashr = ashr <4 x i32> %shl, <i32 31, i32 31, i32 31, i32 31>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_v2i8_to_v2i32
; SI: S_SEXT_I32_I8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: S_SEXT_I32_I8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: BUFFER_STORE_DWORDX2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 24, i32 24>
  %ashr = ashr <2 x i32> %shl, <i32 24, i32 24>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_v4i8_to_v4i32
; SI: S_SEXT_I32_I8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: S_SEXT_I32_I8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: S_SEXT_I32_I8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: S_SEXT_I32_I8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: BUFFER_STORE_DWORDX4

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW][XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %c = add <4 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 24, i32 24, i32 24, i32 24>
  %ashr = ashr <4 x i32> %shl, <i32 24, i32 24, i32 24, i32 24>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @sext_in_reg_v2i16_to_v2i32
; SI: S_SEXT_I32_I16 {{s[0-9]+}}, {{s[0-9]+}}
; SI: S_SEXT_I32_I16 {{s[0-9]+}}, {{s[0-9]+}}
; SI: BUFFER_STORE_DWORDX2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 16, i32 16>
  %ashr = ashr <2 x i32> %shl, <i32 16, i32 16>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @testcase
define void @testcase(i8 addrspace(1)* %out, i8 %a) nounwind {
  %and_a_1 = and i8 %a, 1
  %cmp_eq = icmp eq i8 %and_a_1, 0
  %cmp_slt = icmp slt i8 %a, 0
  %sel0 = select i1 %cmp_slt, i8 0, i8 %a
  %sel1 = select i1 %cmp_eq, i8 0, i8 %a
  %xor = xor i8 %sel0, %sel1
  store i8 %xor, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @testcase_3
define void @testcase_3(i8 addrspace(1)* %out, i8 %a) nounwind {
  %and_a_1 = and i8 %a, 1
  %cmp_eq = icmp eq i8 %and_a_1, 0
  %cmp_slt = icmp slt i8 %a, 0
  %sel0 = select i1 %cmp_slt, i8 0, i8 %a
  %sel1 = select i1 %cmp_eq, i8 0, i8 %a
  %xor = xor i8 %sel0, %sel1
  store i8 %xor, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @vgpr_sext_in_reg_v4i8_to_v4i32
; SI: V_BFE_I32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; SI: V_BFE_I32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; SI: V_BFE_I32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; SI: V_BFE_I32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
define void @vgpr_sext_in_reg_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b) nounwind {
  %loada = load <4 x i32> addrspace(1)* %a, align 16
  %loadb = load <4 x i32> addrspace(1)* %b, align 16
  %c = add <4 x i32> %loada, %loadb ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 24, i32 24, i32 24, i32 24>
  %ashr = ashr <4 x i32> %shl, <i32 24, i32 24, i32 24, i32 24>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @vgpr_sext_in_reg_v4i16_to_v4i32
; SI: V_BFE_I32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 16
; SI: V_BFE_I32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 16
define void @vgpr_sext_in_reg_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b) nounwind {
  %loada = load <4 x i32> addrspace(1)* %a, align 16
  %loadb = load <4 x i32> addrspace(1)* %b, align 16
  %c = add <4 x i32> %loada, %loadb ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 16, i32 16, i32 16, i32 16>
  %ashr = ashr <4 x i32> %shl, <i32 16, i32 16, i32 16, i32 16>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FIXME: The BFE should really be eliminated. I think it should happen
; when computeKnownBitsForTargetNode is implemented for imax.

; FUNC-LABEL: @sext_in_reg_to_illegal_type
; SI: BUFFER_LOAD_SBYTE
; SI: V_MAX_I32
; SI: V_BFE_I32
; SI: BUFFER_STORE_SHORT
define void @sext_in_reg_to_illegal_type(i16 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %src) nounwind {
  %tmp5 = load i8 addrspace(1)* %src, align 1
  %tmp2 = sext i8 %tmp5 to i32
  %tmp3 = tail call i32 @llvm.AMDGPU.imax(i32 %tmp2, i32 0) nounwind readnone
  %tmp4 = trunc i32 %tmp3 to i8
  %tmp6 = sext i8 %tmp4 to i16
  store i16 %tmp6, i16 addrspace(1)* %out, align 2
  ret void
}

declare i32 @llvm.AMDGPU.bfe.i32(i32, i32, i32) nounwind readnone

; FUNC-LABEL: @bfe_0_width
; SI-NOT: BFE
; SI: S_ENDPGM
define void @bfe_0_width(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32 addrspace(1)* %ptr, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 8, i32 0) nounwind readnone
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_8_bfe_8
; SI: V_BFE_I32
; SI-NOT: BFE
; SI: S_ENDPGM
define void @bfe_8_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 8) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 8) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_8_bfe_16
; SI: V_BFE_I32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; SI: S_ENDPGM
define void @bfe_8_bfe_16(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 8) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 16) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; This really should be folded into 1
; FUNC-LABEL: @bfe_16_bfe_8
; SI: V_BFE_I32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; SI: V_BFE_I32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; SI: S_ENDPGM
define void @bfe_16_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 16) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 8) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; Make sure there isn't a redundant BFE
; FUNC-LABEL: @sext_in_reg_i8_to_i32_bfe
; SI: S_BFE_I32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80000
; SI-NOT: BFE
; SI: S_ENDPGM
define void @sext_in_reg_i8_to_i32_bfe(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %c, i32 0, i32 8) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @sext_in_reg_i8_to_i32_bfe_wrong
define void @sext_in_reg_i8_to_i32_bfe_wrong(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %c, i32 8, i32 0) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @sextload_i8_to_i32_bfe
; SI: BUFFER_LOAD_SBYTE
; SI-NOT: BFE
; SI: S_ENDPGM
define void @sextload_i8_to_i32_bfe(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) nounwind {
  %load = load i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %sext, i32 0, i32 8) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @sextload_i8_to_i32_bfe_0:
; SI-NOT: BFE
; SI: S_ENDPGM
define void @sextload_i8_to_i32_bfe_0(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) nounwind {
  %load = load i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %sext, i32 8, i32 0) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}
