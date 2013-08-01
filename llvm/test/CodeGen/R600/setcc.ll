;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s

; CHECK: @setcc_v2i32
; EG-CHECK-DAG: SETE_INT * T{{[0-9]+\.[XYZW]}}, KC0[3].X, KC0[3].Z
; EG-CHECK-DAG: SETE_INT * T{{[0-9]+\.[XYZW]}}, KC0[2].W, KC0[3].Y

define void @setcc_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) {
  %result = icmp eq <2 x i32> %a, %b
  %sext = sext <2 x i1> %result to <2 x i32>
  store <2 x i32> %sext, <2 x i32> addrspace(1)* %out
  ret void
}

; CHECK: @setcc_v4i32
; EG-CHECK-DAG: SETE_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG-CHECK-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG-CHECK-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG-CHECK-DAG: SETE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @setcc_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = icmp eq <4 x i32> %a, %b
  %sext = sext <4 x i1> %result to <4 x i32>
  store <4 x i32> %sext, <4 x i32> addrspace(1)* %out
  ret void
}
