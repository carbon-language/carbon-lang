; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI %s


declare i32 @llvm.r600.read.tidig.x() #1

; FIXME: This currently doesn't do a great job of clustering the
; loads, which end up with extra moves between them. Right now, it
; seems the only things areLoadsFromSameBasePtr is accomplishing is
; ordering the loads so that the lower address loads come first.

; FUNC-LABEL: @cluster_global_arg_loads
; SI: BUFFER_LOAD_DWORD [[REG0:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI: BUFFER_LOAD_DWORD [[REG1:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:0x4
; SI: BUFFER_STORE_DWORD [[REG0]]
; SI: BUFFER_STORE_DWORD [[REG1]]
define void @cluster_global_arg_loads(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 addrspace(1)* %ptr) #0 {
  %load0 = load i32 addrspace(1)* %ptr, align 4
  %gep = getelementptr i32 addrspace(1)* %ptr, i32 1
  %load1 = load i32 addrspace(1)* %gep, align 4
  store i32 %load0, i32 addrspace(1)* %out0, align 4
  store i32 %load1, i32 addrspace(1)* %out1, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
