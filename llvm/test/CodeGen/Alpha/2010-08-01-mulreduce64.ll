; RUN: llc < %s -march=alpha | FileCheck %s

define fastcc i64 @getcount(i64 %s) {
	%tmp431 = mul i64 %s, 12884901888
	ret i64 %tmp431
}

; CHECK: sll $16,33,$0
; CHECK-NEXT: sll $16,32,$1
; CHECK-NEXT: addq $0,$1,$0

