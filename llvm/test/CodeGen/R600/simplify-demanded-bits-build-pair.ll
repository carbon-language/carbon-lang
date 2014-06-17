; RUN: llc -verify-machineinstrs -march=r600 -mcpu=SI < %s | FileCheck -check-prefix=SI %s

; XFAIL: *

; 64-bit select was originally lowered with a build_pair, and this
; could be simplified to 1 cndmask instead of 2, but that broken when
; it started being implemented with a v2i32 build_vector and
; bitcasting.
define void @trunc_select_i64(i32 addrspace(1)* %out, i64 %a, i64 %b, i32 %c) {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, i64 %a, i64 %b
  %trunc = trunc i64 %select to i32
  store i32 %trunc, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: Fix truncating store for local memory
; SI-LABEL: @trunc_load_alloca_i64:
; SI: DS_READ_B32
; SI-NOT: DS_READ_B64
; SI: S_ENDPGM
define void @trunc_load_alloca_i64(i64 addrspace(1)* %out, i32 %a, i32 %b) {
  %idx = add i32 %a, %b
  %alloca = alloca i64, i32 4
  %gep0 = getelementptr i64* %alloca, i64 0
  %gep1 = getelementptr i64* %alloca, i64 1
  %gep2 = getelementptr i64* %alloca, i64 2
  %gep3 = getelementptr i64* %alloca, i64 3
  store i64 24, i64* %gep0, align 8
  store i64 9334, i64* %gep1, align 8
  store i64 3935, i64* %gep2, align 8
  store i64 9342, i64* %gep3, align 8
  %gep = getelementptr i64* %alloca, i32 %idx
  %load = load i64* %gep, align 8
  %mask = and i64 %load, 4294967296
  %add = add i64 %mask, -1
  store i64 %add, i64 addrspace(1)* %out, align 4
  ret void
}
