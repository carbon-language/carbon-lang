; RUN: llc -march=r600 -mcpu=SI -show-mc-encoding -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.r600.read.tidig.x() readnone

;;;==========================================================================;;;
;;; MUBUF LOAD TESTS
;;;==========================================================================;;;

; MUBUF load with an immediate byte offset that fits into 12-bits
; CHECK-LABEL: {{^}}mubuf_load0:
; CHECK: BUFFER_LOAD_DWORD v{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0 offset:0x4 ; encoding: [0x04,0x00,0x30,0xe0
define void @mubuf_load0(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = getelementptr i32 addrspace(1)* %in, i64 1
  %1 = load i32 addrspace(1)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; MUBUF load with the largest possible immediate offset
; CHECK-LABEL: {{^}}mubuf_load1:
; CHECK: BUFFER_LOAD_UBYTE v{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0 offset:0xfff ; encoding: [0xff,0x0f,0x20,0xe0
define void @mubuf_load1(i8 addrspace(1)* %out, i8 addrspace(1)* %in) {
entry:
  %0 = getelementptr i8 addrspace(1)* %in, i64 4095
  %1 = load i8 addrspace(1)* %0
  store i8 %1, i8 addrspace(1)* %out
  ret void
}

; MUBUF load with an immediate byte offset that doesn't fit into 12-bits
; CHECK-LABEL: {{^}}mubuf_load2:
; CHECK: BUFFER_LOAD_DWORD v{{[0-9]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 ; encoding: [0x00,0x80,0x30,0xe0
define void @mubuf_load2(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = getelementptr i32 addrspace(1)* %in, i64 1024
  %1 = load i32 addrspace(1)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; MUBUF load with a 12-bit immediate offset and a register offset
; CHECK-LABEL: {{^}}mubuf_load3:
; CHECK-NOT: ADD
; CHECK: BUFFER_LOAD_DWORD v{{[0-9]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:0x4 ; encoding: [0x04,0x80,0x30,0xe0
define void @mubuf_load3(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i64 %offset) {
entry:
  %0 = getelementptr i32 addrspace(1)* %in, i64 %offset
  %1 = getelementptr i32 addrspace(1)* %0, i64 1
  %2 = load i32 addrspace(1)* %1
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

;;;==========================================================================;;;
;;; MUBUF STORE TESTS
;;;==========================================================================;;;

; MUBUF store with an immediate byte offset that fits into 12-bits
; CHECK-LABEL: {{^}}mubuf_store0:
; CHECK: BUFFER_STORE_DWORD v{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0 offset:0x4 ; encoding: [0x04,0x00,0x70,0xe0
define void @mubuf_store0(i32 addrspace(1)* %out) {
entry:
  %0 = getelementptr i32 addrspace(1)* %out, i64 1
  store i32 0, i32 addrspace(1)* %0
  ret void
}

; MUBUF store with the largest possible immediate offset
; CHECK-LABEL: {{^}}mubuf_store1:
; CHECK: BUFFER_STORE_BYTE v{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0 offset:0xfff ; encoding: [0xff,0x0f,0x60,0xe0

define void @mubuf_store1(i8 addrspace(1)* %out) {
entry:
  %0 = getelementptr i8 addrspace(1)* %out, i64 4095
  store i8 0, i8 addrspace(1)* %0
  ret void
}

; MUBUF store with an immediate byte offset that doesn't fit into 12-bits
; CHECK-LABEL: {{^}}mubuf_store2:
; CHECK: BUFFER_STORE_DWORD v{{[0-9]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]:[0-9]}}], 0 addr64 ; encoding: [0x00,0x80,0x70,0xe0
define void @mubuf_store2(i32 addrspace(1)* %out) {
entry:
  %0 = getelementptr i32 addrspace(1)* %out, i64 1024
  store i32 0, i32 addrspace(1)* %0
  ret void
}

; MUBUF store with a 12-bit immediate offset and a register offset
; CHECK-LABEL: {{^}}mubuf_store3:
; CHECK-NOT: ADD
; CHECK: BUFFER_STORE_DWORD v{{[0-9]}}, v[{{[0-9]:[0-9]}}], s[{{[0-9]:[0-9]}}], 0 addr64 offset:0x4 ; encoding: [0x04,0x80,0x70,0xe0
define void @mubuf_store3(i32 addrspace(1)* %out, i64 %offset) {
entry:
  %0 = getelementptr i32 addrspace(1)* %out, i64 %offset
  %1 = getelementptr i32 addrspace(1)* %0, i64 1
  store i32 0, i32 addrspace(1)* %1
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr:
; CHECK: BUFFER_STORE_DWORD v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0
define void @store_sgpr_ptr(i32 addrspace(1)* %out) #0 {
  store i32 99, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr_offset:
; CHECK: BUFFER_STORE_DWORD v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:0x28
define void @store_sgpr_ptr_offset(i32 addrspace(1)* %out) #0 {
  %out.gep = getelementptr i32 addrspace(1)* %out, i32 10
  store i32 99, i32 addrspace(1)* %out.gep, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr_large_offset:
; CHECK: BUFFER_STORE_DWORD v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
define void @store_sgpr_ptr_large_offset(i32 addrspace(1)* %out) #0 {
  %out.gep = getelementptr i32 addrspace(1)* %out, i32 32768
  store i32 99, i32 addrspace(1)* %out.gep, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_vgpr_ptr:
; CHECK: BUFFER_STORE_DWORD v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
define void @store_vgpr_ptr(i32 addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %out.gep = getelementptr i32 addrspace(1)* %out, i32 %tid
  store i32 99, i32 addrspace(1)* %out.gep, align 4
  ret void
}
