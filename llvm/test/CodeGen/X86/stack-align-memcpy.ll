; RUN: llc < %s -force-align-stack -mtriple i386-apple-darwin -mcpu=i486 | FileCheck %s

%struct.foo = type { [88 x i8] }

; PR15249
; We can't use rep;movsl here because it clobbers the base pointer in %esi.
define void @test1(%struct.foo* nocapture %x, i32 %y) nounwind {
  %dynalloc = alloca i8, i32 %y, align 1
  call void @bar(i8* %dynalloc, %struct.foo* align 4 byval %x)
  ret void

; CHECK: test1:
; CHECK: andl $-16, %esp
; CHECK: movl %esp, %esi
; CHECK-NOT: rep;movsl
}

declare void @bar(i8* nocapture, %struct.foo* align 4 byval) nounwind
