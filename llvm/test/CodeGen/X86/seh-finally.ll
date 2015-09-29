; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: sed -e 's/__C_specific_handler/_except_handler3/' %s | \
; RUN:        llc -mtriple=i686-windows-msvc | FileCheck %s --check-prefix=X86

@str_recovered = internal unnamed_addr constant [10 x i8] c"recovered\00", align 1

declare void @crash()

define i32 @main() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @crash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call = call i32 @puts(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @str_recovered, i64 0, i64 0))
  call void @abort()
  ret i32 0

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          cleanup
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  %call2 = invoke i32 @puts(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @str_recovered, i64 0, i64 0))
          to label %invoke.cont1 unwind label %terminate.lpad

invoke.cont1:                                     ; preds = %lpad
  resume { i8*, i32 } %0

terminate.lpad:                                   ; preds = %lpad
  %3 = landingpad { i8*, i32 }
          catch i8* null
  call void @abort()
  unreachable
}

; X64-LABEL: main:
; X64: retq

; X64: .seh_handlerdata
; X64-NEXT: .text
; X64-NEXT: .Ltmp{{[0-9]+}}:
; X64-NEXT: .seh_endproc
; X64-NEXT: .section .xdata,"dr"
; X64-NEXT: .long 1
; X64-NEXT: .long .Ltmp0@IMGREL
; X64-NEXT: .long .Ltmp1@IMGREL
; X64-NEXT: .long main.cleanup@IMGREL
; X64-NEXT: .long 0

; X64-LABEL: main.cleanup:
; X64: callq puts
; X64: retq

; X86-LABEL: _main:
; X86: retl

; X86: .section .xdata,"dr"
; X86: L__ehtable$main:
; X86-NEXT: .long -1
; X86-NEXT: .long 0
; X86-NEXT: .long _main.cleanup

; X86-LABEL: _main.cleanup:
; X86: calll _puts
; X86: retl

declare i32 @__C_specific_handler(...)

declare i32 @puts(i8*)

declare void @abort()
