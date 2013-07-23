; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s

; CHECK: @i8_arg
; CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z

define void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: @i8_zext_arg
; CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z

define void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: @i8_sext_arg
; CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z
define void @i8_sext_arg(i32 addrspace(1)* nocapture %out, i8 signext %in) nounwind {
entry:
  %0 = sext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: @i16_arg
; CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z

define void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: @i16_zext_arg
; CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z

define void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: @i16_sext_arg
; CHECK: MOV {{[ *]*}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z

define void @i16_sext_arg(i32 addrspace(1)* nocapture %out, i16 signext %in) nounwind {
entry:
  %0 = sext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}
