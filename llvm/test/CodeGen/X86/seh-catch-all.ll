; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

@str = internal unnamed_addr constant [10 x i8] c"recovered\00", align 1

declare i32 @__C_specific_handler(...)
declare void @crash()
declare i32 @puts(i8*)

define i32 @main() {
entry:
  invoke void @crash()
          to label %__try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
          catch i8* null
  call i32 @puts(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @str, i64 0, i64 0))
  br label %__try.cont

__try.cont:
  ret i32 0

eh.resume:
  resume { i8*, i32 } %0
}

; CHECK-LABEL: main:
; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 1
; CHECK-NEXT: .Ltmp{{[0-9]+}}@IMGREL
; CHECK-NEXT: .Ltmp{{[0-9]+}}@IMGREL+1
; CHECK-NEXT: 1
; CHECK-NEXT: .Ltmp{{[0-9]+}}@IMGREL
