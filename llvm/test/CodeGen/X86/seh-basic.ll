; RUN: llc -mtriple x86_64-pc-windows-msvc < %s | FileCheck %s

define void @two_invoke_merged() {
entry:
  invoke void @try_body()
          to label %again unwind label %lpad

again:
  invoke void @try_body()
          to label %done unwind label %lpad

done:
  ret void

lpad:
  %vals = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          catch i8* bitcast (i32 (i8*, i8*)* @filt0 to i8*)
          catch i8* bitcast (i32 (i8*, i8*)* @filt1 to i8*)
  %sel = extractvalue { i8*, i32 } %vals, 1
  call void @use_selector(i32 %sel)
  ret void
}

; Normal path code

; CHECK-LABEL: {{^}}two_invoke_merged:
; CHECK: .seh_proc two_invoke_merged
; CHECK: .seh_handler __C_specific_handler, @unwind, @except
; CHECK: .Ltmp0:
; CHECK: callq try_body
; CHECK-NEXT: .Ltmp1:
; CHECK: .Ltmp2:
; CHECK: callq try_body
; CHECK-NEXT: .Ltmp3:
; CHECK: retq

; Landing pad code

; CHECK: .Ltmp5:
; CHECK: movl $1, %ecx
; CHECK: jmp
; CHECK: .Ltmp6:
; CHECK: movl $2, %ecx
; CHECK: callq use_selector

; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 2
; CHECK-NEXT: .long .Ltmp0@IMGREL
; CHECK-NEXT: .long .Ltmp3@IMGREL+1
; CHECK-NEXT: .long filt0@IMGREL
; CHECK-NEXT: .long .Ltmp5@IMGREL
; CHECK-NEXT: .long .Ltmp0@IMGREL
; CHECK-NEXT: .long .Ltmp3@IMGREL+1
; CHECK-NEXT: .long filt1@IMGREL
; CHECK-NEXT: .long .Ltmp6@IMGREL
; CHECK: .text
; CHECK: .seh_endproc

define void @two_invoke_gap() {
entry:
  invoke void @try_body()
          to label %again unwind label %lpad

again:
  call void @do_nothing_on_unwind()
  invoke void @try_body()
          to label %done unwind label %lpad

done:
  ret void

lpad:
  %vals = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          catch i8* bitcast (i32 (i8*, i8*)* @filt0 to i8*)
  %sel = extractvalue { i8*, i32 } %vals, 1
  call void @use_selector(i32 %sel)
  ret void
}

; Normal path code

; CHECK-LABEL: {{^}}two_invoke_gap:
; CHECK: .seh_proc two_invoke_gap
; CHECK: .seh_handler __C_specific_handler, @unwind, @except
; CHECK: .Ltmp11:
; CHECK: callq try_body
; CHECK-NEXT: .Ltmp12:
; CHECK: callq do_nothing_on_unwind
; CHECK: .Ltmp13:
; CHECK: callq try_body
; CHECK-NEXT: .Ltmp14:
; CHECK: retq

; Landing pad code

; CHECK: .Ltmp16:
; CHECK: movl $1, %ecx
; CHECK: callq use_selector

; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 2
; CHECK-NEXT: .long .Ltmp11@IMGREL
; CHECK-NEXT: .long .Ltmp12@IMGREL+1
; CHECK-NEXT: .long filt0@IMGREL
; CHECK-NEXT: .long .Ltmp16@IMGREL
; CHECK-NEXT: .long .Ltmp13@IMGREL
; CHECK-NEXT: .long .Ltmp14@IMGREL+1
; CHECK-NEXT: .long filt0@IMGREL
; CHECK-NEXT: .long .Ltmp16@IMGREL
; CHECK: .text
; CHECK: .seh_endproc

define void @two_invoke_nounwind_gap() {
entry:
  invoke void @try_body()
          to label %again unwind label %lpad

again:
  call void @cannot_unwind()
  invoke void @try_body()
          to label %done unwind label %lpad

done:
  ret void

lpad:
  %vals = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          catch i8* bitcast (i32 (i8*, i8*)* @filt0 to i8*)
  %sel = extractvalue { i8*, i32 } %vals, 1
  call void @use_selector(i32 %sel)
  ret void
}

; Normal path code

; CHECK-LABEL: {{^}}two_invoke_nounwind_gap:
; CHECK: .seh_proc two_invoke_nounwind_gap
; CHECK: .seh_handler __C_specific_handler, @unwind, @except
; CHECK: .Ltmp21:
; CHECK: callq try_body
; CHECK-NEXT: .Ltmp22:
; CHECK: callq cannot_unwind
; CHECK: .Ltmp23:
; CHECK: callq try_body
; CHECK-NEXT: .Ltmp24:
; CHECK: retq

; Landing pad code

; CHECK: .Ltmp26:
; CHECK: movl $1, %ecx
; CHECK: callq use_selector

; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 1
; CHECK-NEXT: .long .Ltmp21@IMGREL
; CHECK-NEXT: .long .Ltmp24@IMGREL+1
; CHECK-NEXT: .long filt0@IMGREL
; CHECK-NEXT: .long .Ltmp26@IMGREL
; CHECK: .text
; CHECK: .seh_endproc

declare void @try_body()
declare void @do_nothing_on_unwind()
declare void @cannot_unwind() nounwind
declare void @use_selector(i32)

declare i32 @filt0(i8* %eh_info, i8* %rsp)
declare i32 @filt1(i8* %eh_info, i8* %rsp)

declare void @handler0()
declare void @handler1()

declare i32 @__C_specific_handler(...)
declare i32 @llvm.eh.typeid.for(i8*) readnone nounwind
