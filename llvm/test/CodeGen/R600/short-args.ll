; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=EG-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; EG-CHECK: @i8_arg
; EG-CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI-CHECK: BUFFER_LOAD_UBYTE

define void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; EG-CHECK: @i8_zext_arg
; EG-CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI-CHECK: S_LOAD_DWORD SGPR{{[0-9]}}, SGPR0_SGPR1, 11

define void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; EG-CHECK: @i8_sext_arg
; EG-CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI-CHECK: S_LOAD_DWORD SGPR{{[0-9]}}, SGPR0_SGPR1, 11

define void @i8_sext_arg(i32 addrspace(1)* nocapture %out, i8 signext %in) nounwind {
entry:
  %0 = sext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; EG-CHECK: @i16_arg
; EG-CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI-CHECK: BUFFER_LOAD_USHORT

define void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; EG-CHECK: @i16_zext_arg
; EG-CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI-CHECK: S_LOAD_DWORD SGPR{{[0-9]}}, SGPR0_SGPR1, 11

define void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; EG-CHECK: @i16_sext_arg
; EG-CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
; SI-CHECK: S_LOAD_DWORD SGPR{{[0-9]}}, SGPR0_SGPR1, 11

define void @i16_sext_arg(i32 addrspace(1)* nocapture %out, i16 signext %in) nounwind {
entry:
  %0 = sext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}
