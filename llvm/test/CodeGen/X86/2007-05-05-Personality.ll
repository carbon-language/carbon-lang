; RUN: llc < %s -mtriple=i686-pc-linux-gnu -o - | FileCheck %s

; CHECK: .cfi_personality 0, __gnat_eh_personality
; CHECK: .cfi_lsda 0, .Lexception0

@error = external global i8

define void @_ada_x() {
entry:
  invoke void @raise()
          to label %eh_then unwind label %unwind

unwind:                                           ; preds = %entry
  %eh_ptr = tail call i8* @llvm.eh.exception()
  %eh_select = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gnat_eh_personality to i8*), i8* @error)
  %eh_typeid = tail call i32 @llvm.eh.typeid.for(i8* @error)
  %tmp2 = icmp eq i32 %eh_select, %eh_typeid
  br i1 %tmp2, label %eh_then, label %Unwind

eh_then:                                          ; preds = %unwind, %entry
  ret void

Unwind:                                           ; preds = %unwind
  %0 = tail call i32 (...)* @_Unwind_Resume(i8* %eh_ptr)
  unreachable
}

declare void @raise()

declare i8* @llvm.eh.exception() nounwind readonly

declare i32 @llvm.eh.selector(i8*, i8*, ...) nounwind

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare i32 @__gnat_eh_personality(...)

declare i32 @_Unwind_Resume(...)
