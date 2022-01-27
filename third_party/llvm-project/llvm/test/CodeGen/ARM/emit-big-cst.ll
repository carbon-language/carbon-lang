; RUN: llc -mtriple=thumbv7-unknown-unknown -target-abi apcs < %s | FileCheck %s
; Check assembly printing of odd constants.

; CHECK: bigCst:
; CHECK-NEXT: .long 1694510592
; CHECK-NEXT: .long 2960197
; CHECK-NEXT: .short 26220
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .size bigCst, 12

@bigCst = internal constant i82 483673642326615442599424

define void @accessBig(i64* %storage) {
  %addr = bitcast i64* %storage to i82*
  %bigLoadedCst = load volatile i82, i82* @bigCst
  %tmp = add i82 %bigLoadedCst, 1
  store i82 %tmp, i82* %addr
  ret void
}
