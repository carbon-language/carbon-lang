; RUN: llc < %s -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=SI
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=VI

; GCN: {{^}}f64_kernel_arg:
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; SI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0xb
; VI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x24
; VI-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x2c
; GCN: buffer_store_dwordx2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}
