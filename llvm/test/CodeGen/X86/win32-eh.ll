; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck %s

declare void @may_throw_or_crash()
declare i32 @_except_handler3(...)
declare i32 @_except_handler4(...)
declare i32 @__CxxFrameHandler3(...)
declare void @llvm.eh.begincatch(i8*, i8*)
declare void @llvm.eh.endcatch()

define void @use_except_handler3() {
  invoke void @may_throw_or_crash()
      to label %cont unwind label %catchall
cont:
  ret void
catchall:
  landingpad { i8*, i32 } personality i32 (...)* @_except_handler3
      catch i8* null
  br label %cont
}

; CHECK-LABEL: _use_except_handler3:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl $-1, -4(%ebp)
; CHECK: movl $L__ehtable$use_except_handler3, -8(%ebp)
; CHECK: leal -16(%ebp), %[[node:[^ ,]*]]
; CHECK: movl $__except_handler3, -12(%ebp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], -16(%ebp)
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl -16(%ebp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler3:
; CHECK-NEXT:  .long   -1
; CHECK-NEXT:  .long   1
; CHECK-NEXT:  .long   Ltmp{{[0-9]+}}

define void @use_except_handler4() {
  invoke void @may_throw_or_crash()
      to label %cont unwind label %catchall
cont:
  ret void
catchall:
  landingpad { i8*, i32 } personality i32 (...)* @_except_handler4
      catch i8* null
  br label %cont
}

; CHECK-LABEL: _use_except_handler4:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl %esp, -24(%ebp)
; CHECK: movl $-2, -4(%ebp)
; CHECK: movl $L__ehtable$use_except_handler4, %[[lsda:[^ ,]*]]
; CHECK: xorl ___security_cookie, %[[lsda]]
; CHECK: movl %[[lsda]], -8(%ebp)
; CHECK: leal -16(%ebp), %[[node:[^ ,]*]]
; CHECK: movl $__except_handler4, -12(%ebp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], -16(%ebp)
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl -16(%ebp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_except_handler4:
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   9999
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   -2
; CHECK-NEXT:  .long   1
; CHECK-NEXT:  .long   Ltmp{{[0-9]+}}

define void @use_CxxFrameHandler3() {
  invoke void @may_throw_or_crash()
      to label %cont unwind label %catchall
cont:
  ret void
catchall:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__CxxFrameHandler3
      catch i8* null
  %ehptr = extractvalue { i8*, i32 } %ehvals, 0
  call void @llvm.eh.begincatch(i8* %ehptr, i8* null)
  call void @llvm.eh.endcatch()
  br label %cont
}

; CHECK-LABEL: _use_CxxFrameHandler3:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl %esp, -16(%ebp)
; CHECK: movl $-1, -4(%ebp)
; CHECK: leal -12(%ebp), %[[node:[^ ,]*]]
; CHECK: movl $___ehhandler$use_CxxFrameHandler3, -8(%ebp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], -12(%ebp)
; CHECK: movl %[[node]], %fs:0
; CHECK: movl $0, -4(%ebp)
; CHECK: calll _may_throw_or_crash
; CHECK: movl -12(%ebp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

; CHECK: .section .xdata,"dr"
; CHECK-LABEL: L__ehtable$use_CxxFrameHandler3:
; CHECK-NEXT:  .long   429065506
; CHECK-NEXT:  .long   2
; CHECK-NEXT:  .long   ($stateUnwindMap$use_CxxFrameHandler3)
; CHECK-NEXT:  .long   1
; CHECK-NEXT:  .long   ($tryMap$use_CxxFrameHandler3)
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   1

; CHECK-LABEL: ___ehhandler$use_CxxFrameHandler3:
; CHECK: movl $L__ehtable$use_CxxFrameHandler3, %eax
; CHECK: jmp  ___CxxFrameHandler3 # TAILCALL
