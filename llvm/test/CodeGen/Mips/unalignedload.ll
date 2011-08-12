; RUN: llc -march=mips < %s | FileCheck %s

%struct.S2 = type { %struct.S1, %struct.S1 }
%struct.S1 = type { i8, i8 }

@s2 = common global %struct.S2 zeroinitializer, align 1

define void @foo1() nounwind {
entry:
; CHECK: ulw  ${{[0-9]+}}, 2

  tail call void @foo2(%struct.S1* byval getelementptr inbounds (%struct.S2* @s2, i32 0, i32 1)) nounwind
  ret void
}

declare void @foo2(%struct.S1* byval)
