; RUN: llc -march=r600 -mcpu=bonaire < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare double @llvm.trunc.f64(double) nounwind readnone
declare <2 x double> @llvm.trunc.v2f64(<2 x double>) nounwind readnone
declare <3 x double> @llvm.trunc.v3f64(<3 x double>) nounwind readnone
declare <4 x double> @llvm.trunc.v4f64(<4 x double>) nounwind readnone
declare <8 x double> @llvm.trunc.v8f64(<8 x double>) nounwind readnone
declare <16 x double> @llvm.trunc.v16f64(<16 x double>) nounwind readnone

; FUNC-LABEL: {{^}}v_ftrunc_f64:
; CI: V_TRUNC_F64
; SI: V_BFE_U32 {{v[0-9]+}}, {{v[0-9]+}}, 20, 11
; SI: S_ENDPGM
define void @v_ftrunc_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
  %x = load double addrspace(1)* %in, align 8
  %y = call double @llvm.trunc.f64(double %x) nounwind readnone
  store double %y, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_f64:
; CI: V_TRUNC_F64_e32

; SI: S_BFE_U32 [[SEXP:s[0-9]+]], {{s[0-9]+}}, 0xb0014
; SI: S_ADD_I32 s{{[0-9]+}}, [[SEXP]], 0xfffffc01
; SI: S_LSHR_B64
; SI: S_NOT_B64
; SI: S_AND_B64
; SI: S_AND_B32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80000000
; SI: CMP_LT_I32
; SI: CNDMASK_B32
; SI: CNDMASK_B32
; SI: CMP_GT_I32
; SI: CNDMASK_B32
; SI: CNDMASK_B32
; SI: S_ENDPGM
define void @ftrunc_f64(double addrspace(1)* %out, double %x) {
  %y = call double @llvm.trunc.f64(double %x) nounwind readnone
  store double %y, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v2f64:
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
define void @ftrunc_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %x) {
  %y = call <2 x double> @llvm.trunc.v2f64(<2 x double> %x) nounwind readnone
  store <2 x double> %y, <2 x double> addrspace(1)* %out
  ret void
}

; FIXME-FUNC-LABEL: {{^}}ftrunc_v3f64:
; FIXME-CI: V_TRUNC_F64_e32
; FIXME-CI: V_TRUNC_F64_e32
; FIXME-CI: V_TRUNC_F64_e32
; define void @ftrunc_v3f64(<3 x double> addrspace(1)* %out, <3 x double> %x) {
;   %y = call <3 x double> @llvm.trunc.v3f64(<3 x double> %x) nounwind readnone
;   store <3 x double> %y, <3 x double> addrspace(1)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}ftrunc_v4f64:
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
define void @ftrunc_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %x) {
  %y = call <4 x double> @llvm.trunc.v4f64(<4 x double> %x) nounwind readnone
  store <4 x double> %y, <4 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v8f64:
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
define void @ftrunc_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %x) {
  %y = call <8 x double> @llvm.trunc.v8f64(<8 x double> %x) nounwind readnone
  store <8 x double> %y, <8 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v16f64:
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
; CI: V_TRUNC_F64_e32
define void @ftrunc_v16f64(<16 x double> addrspace(1)* %out, <16 x double> %x) {
  %y = call <16 x double> @llvm.trunc.v16f64(<16 x double> %x) nounwind readnone
  store <16 x double> %y, <16 x double> addrspace(1)* %out
  ret void
}
