; RUN: llc -march=x86-64 -mtriple=x86_64-unknown-unknown < %s | FileCheck -check-prefix=DEFAULTCPU -check-prefix=ALL %s
; RUN: llc -march=x86-64 -mcpu=x86-64 -mtriple=x86_64-unknown-unknown < %s | FileCheck -check-prefix=X8664CPU -check-prefix=ALL %s


; ALL-LABEL: {{^}}merge_8_float_zero_stores:

; DEFAULTCPU-DAG: movq $0, ([[PTR:%[a-z]+]])
; DEFAULTCPU-DAG: movq $0, 8([[PTR]])
; DEFAULTCPU-DAG: movq $0, 16([[PTR]])
; DEFAULTCPU-DAG: movq $0, 24([[PTR]])

; X8664CPU: xorps [[ZEROREG:%xmm[0-9]+]], [[ZEROREG]]
; X8664CPU-DAG: movups [[ZEROREG]], ([[PTR:%[a-z]+]])
; X8664CPU-DAG: movups [[ZEROREG]], 16([[PTR:%[a-z]+]])

; ALL: retq
define void @merge_8_float_zero_stores(float* %ptr) {
  %idx0 = getelementptr float, float* %ptr, i64 0
  %idx1 = getelementptr float, float* %ptr, i64 1
  %idx2 = getelementptr float, float* %ptr, i64 2
  %idx3 = getelementptr float, float* %ptr, i64 3
  %idx4 = getelementptr float, float* %ptr, i64 4
  %idx5 = getelementptr float, float* %ptr, i64 5
  %idx6 = getelementptr float, float* %ptr, i64 6
  %idx7 = getelementptr float, float* %ptr, i64 7
  store float 0.0, float* %idx0, align 4
  store float 0.0, float* %idx1, align 4
  store float 0.0, float* %idx2, align 4
  store float 0.0, float* %idx3, align 4
  store float 0.0, float* %idx4, align 4
  store float 0.0, float* %idx5, align 4
  store float 0.0, float* %idx6, align 4
  store float 0.0, float* %idx7, align 4
  ret void
}
