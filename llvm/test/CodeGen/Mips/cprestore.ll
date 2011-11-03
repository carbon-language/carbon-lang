; RUN: llc -march=mipsel < %s | FileCheck %s

; CHECK: .set macro
; CHECK-NEXT: .cprestore
; CHECK-NEXT: .set nomacro

%struct.S = type { [16384 x i32] }

define void @foo2() nounwind {
entry:
  %s = alloca %struct.S, align 4
  call void @foo1(%struct.S* byval %s)
  ret void
}

declare void @foo1(%struct.S* byval)
