; RUN: llc < %s -march=x86 | FileCheck %s

%struct.s = type { double, float }

define void @g() nounwind {
entry:
  %tmp = alloca %struct.s, align 4
  call void @f(%struct.s* inreg sret %tmp, i32 inreg 41, i32 inreg 42, i32 43)
  ret void
  ; CHECK: g:
  ; CHECK: subl {{.*}}, %esp
  ; CHECK-NEXT: $43, (%esp)
  ; CHECK-NEXT: leal    16(%esp), %eax
  ; CHECK-NEXT: movl    $41, %edx
  ; CHECK-NEXT: movl    $42, %ecx
  ; CHECK-NEXT: calll   f
}

declare void @f(%struct.s* inreg sret, i32 inreg, i32 inreg, i32)
