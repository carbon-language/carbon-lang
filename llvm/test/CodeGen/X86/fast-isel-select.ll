; RUN: llc -mtriple x86_64-apple-darwin -O0 -o - < %s | FileCheck %s
; Make sure we only use the less significant bit of the value that feeds the
; select. Otherwise, we may account for a non-zero value whereas the
; lsb is zero.
; <rdar://problem/15651765>

; CHECK-LABEL: fastisel_select:
; CHECK: subb {{%[a-z0-9]+}}, [[RES:%[a-z0-9]+]]
; CHECK: testb $1, [[RES]]
; CHECK: cmovnel %edi, %esi
define i32 @fastisel_select(i1 %exchSub2211_, i1 %trunc_8766) {
  %shuffleInternal15257_8932 = sub i1 %exchSub2211_, %trunc_8766
  %counter_diff1345 = select i1 %shuffleInternal15257_8932, i32 1204476887, i32 0
  ret i32 %counter_diff1345
}

