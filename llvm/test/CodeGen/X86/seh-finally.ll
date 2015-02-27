; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

@str_recovered = internal unnamed_addr constant [10 x i8] c"recovered\00", align 1

declare void @crash()

define i32 @main() {
entry:
  invoke void @crash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call = call i32 @puts(i8* getelementptr inbounds ([10 x i8]* @str_recovered, i64 0, i64 0))
  call void @abort()
  ret i32 0

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          cleanup
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  %call2 = invoke i32 @puts(i8* getelementptr inbounds ([10 x i8]* @str_recovered, i64 0, i64 0))
          to label %invoke.cont1 unwind label %terminate.lpad

invoke.cont1:                                     ; preds = %lpad
  resume { i8*, i32 } %0

terminate.lpad:                                   ; preds = %lpad
  %3 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          catch i8* null
  call void @abort()
  unreachable
}

; CHECK: main:

; FIXME: No handlers yet!
; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 0

declare i32 @__C_specific_handler(...)

declare i32 @puts(i8*)

declare void @abort()
