; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=CHECK-PIC
; RUN: llc -march=mipsel -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC
; RUN: llc -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=CHECK-PIC16
; RUN: llc -march=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC16

define void @count(i32 %x, i32 %y, i32 %z) noreturn nounwind readnone {
entry:
  br label %bosco

bosco:                                            ; preds = %bosco, %entry
  br label %bosco
}

; CHECK-PIC: b	$BB0_1
; CHECK-STATIC: j	$BB0_1
; CHECK-PIC16: b	$BB0_1
; CHECK-STATIC16: b	$BB0_1

