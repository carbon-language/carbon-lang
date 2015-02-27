; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


@a = internal addrspace(2) constant [1 x i8] [ i8 7 ], align 1

; FUNC-LABEL: {{^}}test_i8:
; EG: CF_END
; SI: buffer_store_byte
; SI: s_endpgm
define void @test_i8( i32 %s, i8 addrspace(1)* %out) #3 {
  %arrayidx = getelementptr inbounds [1 x i8], [1 x i8] addrspace(2)* @a, i32 0, i32 %s
  %1 = load i8 addrspace(2)* %arrayidx, align 1
  store i8 %1, i8 addrspace(1)* %out
  ret void
}

@b = internal addrspace(2) constant [1 x i16] [ i16 7 ], align 2

; FUNC-LABEL: {{^}}test_i16:
; EG: CF_END
; SI: buffer_store_short
; SI: s_endpgm
define void @test_i16( i32 %s, i16 addrspace(1)* %out) #3 {
  %arrayidx = getelementptr inbounds [1 x i16], [1 x i16] addrspace(2)* @b, i32 0, i32 %s
  %1 = load i16 addrspace(2)* %arrayidx, align 2
  store i16 %1, i16 addrspace(1)* %out
  ret void
}

%struct.bar = type { float, [5 x i8] }

; The illegal i8s aren't handled
@struct_bar_gv = internal addrspace(2) constant [1 x %struct.bar] [ %struct.bar { float 16.0, [5 x i8] [i8 0, i8 1, i8 2, i8 3, i8 4] } ]

; FUNC-LABEL: {{^}}struct_bar_gv_load:
define void @struct_bar_gv_load(i8 addrspace(1)* %out, i32 %index) {
  %gep = getelementptr inbounds [1 x %struct.bar], [1 x %struct.bar] addrspace(2)* @struct_bar_gv, i32 0, i32 0, i32 1, i32 %index
  %load = load i8 addrspace(2)* %gep, align 1
  store i8 %load, i8 addrspace(1)* %out, align 1
  ret void
}


; The private load isn't scalarzied.
@array_vector_gv = internal addrspace(2) constant [4 x <4 x i32>] [ <4 x i32> <i32 1, i32 2, i32 3, i32 4>,
                                                                    <4 x i32> <i32 5, i32 6, i32 7, i32 8>,
                                                                    <4 x i32> <i32 9, i32 10, i32 11, i32 12>,
                                                                    <4 x i32> <i32 13, i32 14, i32 15, i32 16> ]

; FUNC-LABEL: {{^}}array_vector_gv_load:
define void @array_vector_gv_load(<4 x i32> addrspace(1)* %out, i32 %index) {
  %gep = getelementptr inbounds [4 x <4 x i32>], [4 x <4 x i32>] addrspace(2)* @array_vector_gv, i32 0, i32 %index
  %load = load <4 x i32> addrspace(2)* %gep, align 16
  store <4 x i32> %load, <4 x i32> addrspace(1)* %out, align 16
  ret void
}
