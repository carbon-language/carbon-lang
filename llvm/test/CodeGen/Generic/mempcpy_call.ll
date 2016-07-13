; RUN: llc < %s -O2 | FileCheck %s

; This test just checks that mempcpy is lowered as memcpy.
; The test to check that the return value of mempcpy is the dst pointer adjusted
; by the copy size is done by Codegen/X86/mempcpy_ret_val.ll

; CHECK-LABEL: CALL_MEMPCPY:
; CHECK: callq memcpy
;
define void @CALL_MEMPCPY(i8* %DST, i8* %SRC, i64 %N) {
entry:
  %call = tail call i8* @mempcpy(i8* %DST, i8* %SRC, i64 %N)
  ret void
}

declare i8* @mempcpy(i8*, i8*, i64)
