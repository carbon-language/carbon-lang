; RUN: llc < %s -march=r600 -mcpu=SI -show-mc-encoding -verify-machineinstrs | FileCheck %s

;;;==========================================================================;;;
;;; MUBUF LOAD TESTS
;;;==========================================================================;;;

; MUBUF load with an immediate byte offset that fits into 12-bits
; CHECK-LABEL: @mubuf_load0
; CHECK: BUFFER_LOAD_DWORD v{{[0-9]}}, s[{{[0-9]:[0-9]}}] + v[{{[0-9]:[0-9]}}] + 4 ; encoding: [0x04,0x80
define void @mubuf_load0(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = getelementptr i32 addrspace(1)* %in, i64 1
  %1 = load i32 addrspace(1)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; MUBUF load with the largest possible immediate offset
; CHECK-LABEL: @mubuf_load1
; CHECK: BUFFER_LOAD_UBYTE v{{[0-9]}}, s[{{[0-9]:[0-9]}}] + v[{{[0-9]:[0-9]}}] + 4095 ; encoding: [0xff,0x8f
define void @mubuf_load1(i8 addrspace(1)* %out, i8 addrspace(1)* %in) {
entry:
  %0 = getelementptr i8 addrspace(1)* %in, i64 4095
  %1 = load i8 addrspace(1)* %0
  store i8 %1, i8 addrspace(1)* %out
  ret void
}

; MUBUF load with an immediate byte offset that doesn't fit into 12-bits
; CHECK-LABEL: @mubuf_load2
; CHECK: BUFFER_LOAD_DWORD v{{[0-9]}}, s[{{[0-9]:[0-9]}}] + v[{{[0-9]:[0-9]}}] + 0 ; encoding: [0x00,0x80
define void @mubuf_load2(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = getelementptr i32 addrspace(1)* %in, i64 1024
  %1 = load i32 addrspace(1)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; MUBUF load with a 12-bit immediate offset and a register offset
; CHECK-LABEL: @mubuf_load3
; CHECK-NOT: ADD
; CHECK: BUFFER_LOAD_DWORD v{{[0-9]}}, s[{{[0-9]:[0-9]}}] + v[{{[0-9]:[0-9]}}] + 4 ; encoding: [0x04,0x80
define void @mubuf_load3(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i64 %offset) {
entry:
  %0 = getelementptr i32 addrspace(1)* %in, i64 %offset
  %1 = getelementptr i32 addrspace(1)* %0, i64 1
  %2 = load i32 addrspace(1)* %1
  store i32 %2, i32 addrspace(1)* %out
  ret void
}
