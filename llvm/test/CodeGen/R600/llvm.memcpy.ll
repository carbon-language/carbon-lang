; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* nocapture, i8 addrspace(3)* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture, i64, i32, i1) nounwind


; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align1:
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8

; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8

; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_WRITE_B8
; SI: DS_READ_U8
; SI: DS_READ_U8


; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8

; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8
; SI: DS_READ_U8

; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8

; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8
; SI: DS_WRITE_B8

; SI: S_ENDPGM
define void @test_small_memcpy_i64_lds_to_lds_align1(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* %bcout, i8 addrspace(3)* %bcin, i32 32, i32 1, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align2:
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16

; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16
; SI: DS_READ_U16

; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16

; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16
; SI: DS_WRITE_B16

; SI: S_ENDPGM
define void @test_small_memcpy_i64_lds_to_lds_align2(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* %bcout, i8 addrspace(3)* %bcin, i32 32, i32 2, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align4:
; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI: S_ENDPGM
define void @test_small_memcpy_i64_lds_to_lds_align4(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* %bcout, i8 addrspace(3)* %bcin, i32 32, i32 4, i1 false) nounwind
  ret void
}

; FIXME: Use 64-bit ops
; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align8:

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: DS_READ_B32
; SI-DAG: DS_WRITE_B32

; SI-DAG: S_ENDPGM
define void @test_small_memcpy_i64_lds_to_lds_align8(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* %bcout, i8 addrspace(3)* %bcin, i32 32, i32 8, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align1:
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE

; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE

; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE

; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE
; SI-DAG: BUFFER_LOAD_UBYTE
; SI-DAG: BUFFER_STORE_BYTE

; SI: S_ENDPGM
define void @test_small_memcpy_i64_global_to_global_align1(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %bcout, i8 addrspace(1)* %bcin, i64 32, i32 1, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align2:
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT
; SI-DAG: BUFFER_LOAD_USHORT

; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT
; SI-DAG: BUFFER_STORE_SHORT

; SI: S_ENDPGM
define void @test_small_memcpy_i64_global_to_global_align2(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %bcout, i8 addrspace(1)* %bcin, i64 32, i32 2, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align4:
; SI: BUFFER_LOAD_DWORDX4
; SI: BUFFER_LOAD_DWORDX4
; SI: BUFFER_STORE_DWORDX4
; SI: BUFFER_STORE_DWORDX4
; SI: S_ENDPGM
define void @test_small_memcpy_i64_global_to_global_align4(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %bcout, i8 addrspace(1)* %bcin, i64 32, i32 4, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align8:
; SI: BUFFER_LOAD_DWORDX4
; SI: BUFFER_LOAD_DWORDX4
; SI: BUFFER_STORE_DWORDX4
; SI: BUFFER_STORE_DWORDX4
; SI: S_ENDPGM
define void @test_small_memcpy_i64_global_to_global_align8(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %bcout, i8 addrspace(1)* %bcin, i64 32, i32 8, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align16:
; SI: BUFFER_LOAD_DWORDX4
; SI: BUFFER_LOAD_DWORDX4
; SI: BUFFER_STORE_DWORDX4
; SI: BUFFER_STORE_DWORDX4
; SI: S_ENDPGM
define void @test_small_memcpy_i64_global_to_global_align16(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %bcout, i8 addrspace(1)* %bcin, i64 32, i32 16, i1 false) nounwind
  ret void
}
