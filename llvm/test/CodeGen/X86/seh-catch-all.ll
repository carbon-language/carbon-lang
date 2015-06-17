; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s

@str = linkonce_odr unnamed_addr constant [27 x i8] c"GetExceptionCode(): 0x%lx\0A\00", align 1

declare i32 @__C_specific_handler(...)
declare void @crash()
declare i32 @printf(i8* nocapture readonly, ...) nounwind

define i32 @main() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @crash()
          to label %__try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = ptrtoint i8* %1 to i64
  %3 = trunc i64 %2 to i32
  call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @str, i64 0, i64 0), i32 %3)
  br label %__try.cont

__try.cont:
  ret i32 0

eh.resume:
  resume { i8*, i32 } %0
}

; Check that we can get the exception code from eax to the printf.

; CHECK-LABEL: main:
; CHECK: callq crash
; CHECK: retq
; CHECK: # Block address taken
; CHECK: leaq str(%rip), %rcx
; CHECK: movl %eax, %edx
; CHECK: callq printf

; CHECK: .seh_handlerdata
; CHECK-NEXT: .long 1
; CHECK-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL
; CHECK-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL+1
; CHECK-NEXT: .long 1
; CHECK-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL
