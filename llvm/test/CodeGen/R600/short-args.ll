; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @i8_arg(i32 addrspace(1)* nocapture %out, i8 %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @i8_zext_arg(i32 addrspace(1)* nocapture %out, i8 zeroext %in) nounwind {
entry:
  %0 = zext i8 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @i16_arg(i32 addrspace(1)* nocapture %out, i16 %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @i16_zext_arg(i32 addrspace(1)* nocapture %out, i16 zeroext %in) nounwind {
entry:
  %0 = zext i16 %in to i32
  store i32 %0, i32 addrspace(1)* %out, align 4
  ret void
}
