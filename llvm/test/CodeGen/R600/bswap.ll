; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare <2 x i32> @llvm.bswap.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>) nounwind readnone
declare <8 x i32> @llvm.bswap.v8i32(<8 x i32>) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>) nounwind readnone
declare <4 x i64> @llvm.bswap.v4i64(<4 x i64>) nounwind readnone

; FUNC-LABEL: @test_bswap_i32
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]]
; SI-DAG: V_ALIGNBIT_B32 [[TMP0:v[0-9]+]], [[VAL]], [[VAL]], 8
; SI-DAG: V_ALIGNBIT_B32 [[TMP1:v[0-9]+]], [[VAL]], [[VAL]], 24
; SI-DAG: S_MOV_B32 [[K:s[0-9]+]], 0xff00ff
; SI: V_BFI_B32 [[RESULT:v[0-9]+]], [[K]], [[TMP1]], [[TMP0]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
; SI: S_ENDPGM
define void @test_bswap_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %val = load i32 addrspace(1)* %in, align 4
  %bswap = call i32 @llvm.bswap.i32(i32 %val) nounwind readnone
  store i32 %bswap, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_bswap_v2i32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI: S_ENDPGM
define void @test_bswap_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) nounwind {
  %val = load <2 x i32> addrspace(1)* %in, align 8
  %bswap = call <2 x i32> @llvm.bswap.v2i32(<2 x i32> %val) nounwind readnone
  store <2 x i32> %bswap, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_bswap_v4i32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI: S_ENDPGM
define void @test_bswap_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) nounwind {
  %val = load <4 x i32> addrspace(1)* %in, align 16
  %bswap = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %val) nounwind readnone
  store <4 x i32> %bswap, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: @test_bswap_v8i32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_ALIGNBIT_B32
; SI-DAG: V_BFI_B32
; SI: S_ENDPGM
define void @test_bswap_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> addrspace(1)* %in) nounwind {
  %val = load <8 x i32> addrspace(1)* %in, align 32
  %bswap = call <8 x i32> @llvm.bswap.v8i32(<8 x i32> %val) nounwind readnone
  store <8 x i32> %bswap, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

define void @test_bswap_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %val = load i64 addrspace(1)* %in, align 8
  %bswap = call i64 @llvm.bswap.i64(i64 %val) nounwind readnone
  store i64 %bswap, i64 addrspace(1)* %out, align 8
  ret void
}

define void @test_bswap_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) nounwind {
  %val = load <2 x i64> addrspace(1)* %in, align 16
  %bswap = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %val) nounwind readnone
  store <2 x i64> %bswap, <2 x i64> addrspace(1)* %out, align 16
  ret void
}

define void @test_bswap_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) nounwind {
  %val = load <4 x i64> addrspace(1)* %in, align 32
  %bswap = call <4 x i64> @llvm.bswap.v4i64(<4 x i64> %val) nounwind readnone
  store <4 x i64> %bswap, <4 x i64> addrspace(1)* %out, align 32
  ret void
}
