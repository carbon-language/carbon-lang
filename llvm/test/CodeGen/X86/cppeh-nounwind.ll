; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck %s

; Sometimes invokes of nounwind functions make it through to CodeGen, especially
; at -O0, where Clang sometimes optimistically annotates functions as nounwind.
; WinEHPrepare ends up outlining functions, and emitting references to LSDA
; labels. Make sure we emit the LSDA in that case.

declare i32 @__CxxFrameHandler3(...)
declare void @nounwind_func() nounwind
declare void @cleanup()

define void @should_emit_tables() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @nounwind_func()
      to label %done unwind label %lpad

done:
  ret void

lpad:
  %vals = landingpad { i8*, i32 }
      cleanup
  call void @cleanup()
  resume { i8*, i32 } %vals
}

; CHECK: _should_emit_tables:
; CHECK: calll _nounwind_func
; CHECK: retl

; CHECK: L__ehtable$should_emit_tables:

; CHECK: ___ehhandler$should_emit_tables:
; CHECK: movl $L__ehtable$should_emit_tables, %eax
; CHECK: jmp ___CxxFrameHandler3 # TAILCALL
