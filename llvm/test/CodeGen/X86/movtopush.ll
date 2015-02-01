; RUN: llc < %s -mtriple=i686-windows | FileCheck %s -check-prefix=NORMAL
; RUN: llc < %s -mtriple=i686-windows -force-align-stack -stack-alignment=32 | FileCheck %s -check-prefix=ALIGNED 
declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)
declare void @inreg(i32 %a, i32 inreg %b, i32 %c, i32 %d)

; Here, we should have a reserved frame, so we don't expect pushes
; NORMAL-LABEL: test1
; NORMAL: subl    $16, %esp
; NORMAL-NEXT: movl    $4, 12(%esp)
; NORMAL-NEXT: movl    $3, 8(%esp)
; NORMAL-NEXT: movl    $2, 4(%esp)
; NORMAL-NEXT: movl    $1, (%esp)
; NORMAL-NEXT: call
define void @test1() {
entry:
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Here, we expect a sequence of 4 immediate pushes
; NORMAL-LABEL: test2
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   $1
; NORMAL-NEXT: call
define void @test2(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Again, we expect a sequence of 4 immediate pushes
; Checks that we generate the right pushes for >8bit immediates
; NORMAL-LABEL: test2b
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4096
; NORMAL-NEXT: pushl   $3072
; NORMAL-NEXT: pushl   $2048
; NORMAL-NEXT: pushl   $1024
; NORMAL-NEXT: call
define void @test2b(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 1024, i32 2048, i32 3072, i32 4096)
  ret void
}

; The first push should push a register
; NORMAL-LABEL: test3
; NORMAL-NOT: subl {{.*}} %esp
; NORMAL: pushl   $4
; NORMAL-NEXT: pushl   $3
; NORMAL-NEXT: pushl   $2
; NORMAL-NEXT: pushl   %e{{..}}
; NORMAL-NEXT: call
define void @test3(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 %k, i32 2, i32 3, i32 4)
  ret void
}

; We don't support weird calling conventions
; NORMAL-LABEL: test4
; NORMAL: subl    $12, %esp
; NORMAL-NEXT: movl    $4, 8(%esp)
; NORMAL-NEXT: movl    $3, 4(%esp)
; NORMAL-NEXT: movl    $1, (%esp)
; NORMAL-NEXT: movl    $2, %eax
; NORMAL-NEXT: call
define void @test4(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @inreg(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Check that additional alignment is added when the pushes
; don't add up to the required alignment.
; ALIGNED-LABEL: test5
; ALIGNED: subl    $16, %esp
; ALIGNED-NEXT: pushl   $4
; ALIGNED-NEXT: pushl   $3
; ALIGNED-NEXT: pushl   $2
; ALIGNED-NEXT: pushl   $1
; ALIGNED-NEXT: call
define void @test5(i32 %k) {
entry:
  %a = alloca i32, i32 %k
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Check that pushing the addresses of globals (Or generally, things that 
; aren't exactly immediates) isn't broken.
; Fixes PR21878.
; NORMAL-LABEL: test6
; NORMAL: pushl    $_ext
; NORMAL-NEXT: call
declare void @f(i8*)
@ext = external constant i8

define void @test6() {
  call void @f(i8* @ext)
  br label %bb
bb:
  alloca i32
  ret void
}
