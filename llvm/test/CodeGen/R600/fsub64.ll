; RUN: llc -march=r600 -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}fsub_f64:
; SI: V_ADD_F64 {{v\[[0-9]+:[0-9]+\], v\[[0-9]+:[0-9]+\], -v\[[0-9]+:[0-9]+\]}}
define void @fsub_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fsub double %r0, %r1
   store double %r2, double addrspace(1)* %out
   ret void
}
