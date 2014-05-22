; RUN: llc  < %s -march=mipsel  | FileCheck %s -check-prefix=ALL -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips    | FileCheck %s -check-prefix=ALL -check-prefix=CHECK-EB
%struct.S2 = type { %struct.S1, %struct.S1 }
%struct.S1 = type { i8, i8 }
%struct.S4 = type { [7 x i8] }

@s2 = common global %struct.S2 zeroinitializer, align 1
@s4 = common global %struct.S4 zeroinitializer, align 1

define void @bar1() nounwind {
entry:
; ALL-LABEL: bar1:

; ALL-DAG:      lw $[[R0:[0-9]+]], %got(s2)(

; ALL-DAG:      lbu $[[PART1:[0-9]+]], 2($[[R0]])
; ALL-DAG:      lbu $[[PART2:[0-9]+]], 3($[[R0]])

; CHECK-EL-DAG: sll $[[T0:[0-9]+]], $[[PART2]], 8
; CHECK-EL-DAG: or  $4, $[[T0]], $[[PART1]]

; CHECK-EB-DAG: sll $[[T0:[0-9]+]], $[[PART1]], 8
; CHECK-EB-DAG: or  $[[T1:[0-9]+]], $[[T0]], $[[PART2]]
; CHECK-EB-DAG: sll $4, $[[T1]], 16

  tail call void @foo2(%struct.S1* byval getelementptr inbounds (%struct.S2* @s2, i32 0, i32 1)) nounwind
  ret void
}

define void @bar2() nounwind {
entry:
; ALL-LABEL: bar2:

; ALL-DAG:      lw $[[R2:[0-9]+]], %got(s4)(

; CHECK-EL-DAG: lwl $[[R1:4]], 3($[[R2]])
; CHECK-EL-DAG: lwr $[[R1]], 0($[[R2]])

; CHECK-EB-DAG: lwl $[[R1:4]], 0($[[R2]])
; CHECK-EB-DAG: lwr $[[R1]], 3($[[R2]])

  tail call void @foo4(%struct.S4* byval @s4) nounwind
  ret void
}

declare void @foo2(%struct.S1* byval)

declare void @foo4(%struct.S4* byval)
