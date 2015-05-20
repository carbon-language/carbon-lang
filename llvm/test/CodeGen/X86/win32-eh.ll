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
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl $-1, 12(%esp)
; CHECK: movl $L__ehtable$use_except_handler3, 8(%esp)
; CHECK: movl $__except_handler3, 4(%esp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], (%esp)
; CHECK: leal (%esp), %[[node:[^ ,]*]]
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl (%esp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

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
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl %esp, (%esp)
; CHECK: movl $-2, 20(%esp)
; CHECK: movl $L__ehtable$use_except_handler4, 4(%esp)
; CHECK: leal 8(%esp), %[[node:[^ ,]*]]
; CHECK: movl $__except_handler4, 12(%esp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], 8(%esp)
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl 8(%esp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

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
; CHECK: subl ${{[0-9]+}}, %esp
; CHECK: movl %esp, (%esp)
; CHECK: movl $-1, 12(%esp)
; CHECK: leal 4(%esp), %[[node:[^ ,]*]]
; CHECK: movl $___ehhandler$use_CxxFrameHandler3, 8(%esp)
; CHECK: movl %fs:0, %[[next:[^ ,]*]]
; CHECK: movl %[[next]], 4(%esp)
; CHECK: movl %[[node]], %fs:0
; CHECK: calll _may_throw_or_crash
; CHECK: movl 4(%esp), %[[next:[^ ,]*]]
; CHECK: movl %[[next]], %fs:0
; CHECK: retl

; CHECK-LABEL: ___ehhandler$use_CxxFrameHandler3:
; CHECK: movl $L__ehtable$use_CxxFrameHandler3, %eax
; CHECK: jmp  ___CxxFrameHandler3 # TAILCALL
