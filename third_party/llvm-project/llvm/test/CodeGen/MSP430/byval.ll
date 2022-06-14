; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

%struct.Foo = type { i16, i16, i16 }
@foo = global %struct.Foo { i16 1, i16 2, i16 3 }, align 2

define i16 @callee(%struct.Foo* byval(%struct.Foo) %f) nounwind {
entry:
; CHECK-LABEL: callee:
; CHECK: mov 2(r1), r12
  %0 = getelementptr inbounds %struct.Foo, %struct.Foo* %f, i32 0, i32 0
  %1 = load i16, i16* %0, align 2
  ret i16 %1
}

define void @caller() nounwind {
entry:
; CHECK-LABEL: caller:
; CHECK: mov &foo+4, 4(r1)
; CHECK-NEXT: mov &foo+2, 2(r1)
; CHECK-NEXT: mov &foo, 0(r1)
  %call = call i16 @callee(%struct.Foo* byval(%struct.Foo) @foo)
  ret void
}
