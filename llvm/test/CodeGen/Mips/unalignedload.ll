; RUN: llc  < %s -march=mipsel  | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips    | FileCheck %s -check-prefix=CHECK-EB
%struct.S2 = type { %struct.S1, %struct.S1 }
%struct.S1 = type { i8, i8 }
%struct.S4 = type { [7 x i8] }

@s2 = common global %struct.S2 zeroinitializer, align 1
@s4 = common global %struct.S4 zeroinitializer, align 1

define void @foo1() nounwind {
entry:
; CHECK-EL: lw  $25, %call16(foo2)
; CHECK-EL: ulhu  $4, 2
; CHECK-EL: lw  $[[R0:[0-9]+]], %got(s4)
; CHECK-EL: lbu $[[R1:[0-9]+]], 6($[[R0]])
; CHECK-EL: ulhu  $[[R2:[0-9]+]], 4($[[R0]])
; CHECK-EL: sll $[[R3:[0-9]+]], $[[R1]], 16
; CHECK-EL: ulw $4, 0($[[R0]])
; CHECK-EL: lw  $25, %call16(foo4)
; CHECK-EL: or  $5, $[[R2]], $[[R3]]

; CHECK-EB: ulhu  $[[R0:[0-9]+]], 2
; CHECK-EB: lw  $25, %call16(foo2)
; CHECK-EB: sll $4, $[[R0]], 16
; CHECK-EB: lw  $[[R1:[0-9]+]], %got(s4)
; CHECK-EB: ulhu  $[[R2:[0-9]+]], 4($[[R1]])
; CHECK-EB: lbu $[[R3:[0-9]+]], 6($[[R1]])
; CHECK-EB: sll $[[R4:[0-9]+]], $[[R2]], 16
; CHECK-EB: sll $[[R5:[0-9]+]], $[[R3]], 8
; CHECK-EB: ulw $4, 0($[[R1]])
; CHECK-EB: lw  $25, %call16(foo4)
; CHECK-EB: or  $5, $[[R4]], $[[R5]]

  tail call void @foo2(%struct.S1* byval getelementptr inbounds (%struct.S2* @s2, i32 0, i32 1)) nounwind
  tail call void @foo4(%struct.S4* byval @s4) nounwind
  ret void
}

declare void @foo2(%struct.S1* byval)

declare void @foo4(%struct.S4* byval)
