; XFAIL: *
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s


@a = internal addrspace(2) constant [1 x i8] [ i8 7 ], align 1

; FUNC-LABEL: @test_i8
; EG: CF_END
; SI: BUFFER_STORE_BYTE
; SI: S_ENDPGM
define void @test_i8( i32 %s, i8 addrspace(1)* %out) #3 {
  %arrayidx = getelementptr inbounds [1 x i8] addrspace(2)* @a, i32 0, i32 %s
  %1 = load i8 addrspace(2)* %arrayidx, align 1
  store i8 %1, i8 addrspace(1)* %out
  ret void
}

@b = internal addrspace(2) constant [1 x i16] [ i16 7 ], align 2

; FUNC-LABEL: @test_i16
; EG: CF_END
; SI: BUFFER_STORE_SHORT
; SI: S_ENDPGM
define void @test_i16( i32 %s, i16 addrspace(1)* %out) #3 {
  %arrayidx = getelementptr inbounds [1 x i16] addrspace(2)* @b, i32 0, i32 %s
  %1 = load i16 addrspace(2)* %arrayidx, align 2
  store i16 %1, i16 addrspace(1)* %out
  ret void
}
