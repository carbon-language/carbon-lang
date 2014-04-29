; REQUIRES: asserts
; XFAIL: *
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s


; SI-LABEL: @dynamic_insertelement_v2f64:
; SI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x double> %a, double 8.0, i32 %b
  store <2 x double> %vecins, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v2f64:
; SI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i64> %a, i64 5, i32 %b
  store <2 x i64> %vecins, <2 x i64> addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @dynamic_insertelement_v4f64:
; SI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x double> %a, double 8.0, i32 %b
  store <4 x double> %vecins, <4 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: @dynamic_insertelement_v8f64:
; SI: BUFFER_STORE_DWORDX4
define void @dynamic_insertelement_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %a, i32 %b) nounwind {
  %vecins = insertelement <8 x double> %a, double 8.0, i32 %b
  store <8 x double> %vecins, <8 x double> addrspace(1)* %out, align 16
  ret void
}
