; RUN: llc < %s -march=x86-64 | FileCheck %s
; RUN: llc < %s -march=x86 | FileCheck %s
; CHECK: _memcpy
define void @foo([40000 x i32] *%P) nounwind {
  call void @bar([40000 x i32] * byval align 1 %P)
  ret void
}

declare void @bar([40000 x i32] *%P )
    
