; RUN: llc  -march=mipsel -mcpu=mips16 < %s | FileCheck %s -check-prefix=16

; FIXME: Disabled temporarily because it should not have worked previously
; and will be fixed after a subsequent patch
; REQUIRES: disabled


define i32 @main() nounwind {
entry:
  ret i32 0

; 16: 	.set	mips16                  # @main


; 16:	jr	$ra

}
