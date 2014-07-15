; RUN: llc -march=r600 -mcpu=tahiti < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: @test_copy_v4i8
; SI: BUFFER_LOAD_DWORD [[REG:v[0-9]+]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: S_ENDPGM
define void @test_copy_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_x2
; SI: BUFFER_LOAD_DWORD [[REG:v[0-9]+]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: S_ENDPGM
define void @test_copy_v4i8_x2(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_x3
; SI: BUFFER_LOAD_DWORD [[REG:v[0-9]+]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: S_ENDPGM
define void @test_copy_v4i8_x3(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_x4
; SI: BUFFER_LOAD_DWORD [[REG:v[0-9]+]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: BUFFER_STORE_DWORD [[REG]]
; SI: S_ENDPGM
define void @test_copy_v4i8_x4(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %out3, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out3, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_extra_use
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI-DAG: V_ADD
; SI-DAG: V_ADD
; SI-DAG: V_ADD
; SI-DAG: V_ADD
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI_DAG: BUFFER_STORE_BYTE

; After scalarizing v4i8 loads is fixed.
; XSI: BUFFER_LOAD_DWORD
; XSI: V_BFE
; XSI: V_ADD
; XSI: V_ADD
; XSI: V_ADD
; XSI: BUFFER_STORE_DWORD
; XSI: BUFFER_STORE_DWORD

; SI: S_ENDPGM
define void @test_copy_v4i8_extra_use(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  %add = add <4 x i8> %val, <i8 9, i8 9, i8 9, i8 9>
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_x2_extra_use
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI-DAG: V_ADD
; SI-DAG: V_ADD
; SI-DAG: V_ADD
; SI-DAG: V_ADD
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI_DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI_DAG: BUFFER_STORE_BYTE

; XSI: BUFFER_LOAD_DWORD
; XSI: BFE
; XSI: BUFFER_STORE_DWORD
; XSI: V_ADD
; XSI: BUFFER_STORE_DWORD
; XSI-NEXT: BUFFER_STORE_DWORD

; SI: S_ENDPGM
define void @test_copy_v4i8_x2_extra_use(<4 x i8> addrspace(1)* %out0, <4 x i8> addrspace(1)* %out1, <4 x i8> addrspace(1)* %out2, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  %add = add <4 x i8> %val, <i8 9, i8 9, i8 9, i8 9>
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out0, align 4
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out1, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v3i8
; SI-NOT: BFE
; SI-NOT: BFI
; SI: S_ENDPGM
define void @test_copy_v3i8(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(1)* %in) nounwind {
  %val = load <3 x i8> addrspace(1)* %in, align 4
  store <3 x i8> %val, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_volatile_load
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: S_ENDPGM
define void @test_copy_v4i8_volatile_load(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load volatile <4 x i8> addrspace(1)* %in, align 4
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_copy_v4i8_volatile_store
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_LOAD_UBYTE
; SI: BUFFER_STORE_BYTE
; SI: BUFFER_STORE_BYTE
; SI: BUFFER_STORE_BYTE
; SI: BUFFER_STORE_BYTE
; SI: S_ENDPGM
define void @test_copy_v4i8_volatile_store(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %in) nounwind {
  %val = load <4 x i8> addrspace(1)* %in, align 4
  store volatile <4 x i8> %val, <4 x i8> addrspace(1)* %out, align 4
  ret void
}
