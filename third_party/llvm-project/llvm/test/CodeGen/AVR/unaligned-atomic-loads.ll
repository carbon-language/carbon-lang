; RUN: llc -mattr=addsubiw < %s -march=avr | FileCheck %s

; This verifies that the middle end can handle an unaligned atomic load.
;
; In the past, an assertion inside the SelectionDAGBuilder would always
; hit an assertion for unaligned loads and stores.

%AtomicI16 = type { %CellI16, [0 x i8] }
%CellI16 = type { i16, [0 x i8] }

; CHECK-LABEL: foo
; CHECK: ret
define void @foo(%AtomicI16* %self) {
start:
  %a = getelementptr inbounds %AtomicI16, %AtomicI16* %self, i16 0, i32 0, i32 0
  load atomic i16, i16* %a seq_cst, align 1
  ret void
}

