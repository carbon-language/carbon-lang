; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: sed -e 's/__C_specific_handler/_except_handler3/' %s | \
; RUN:         llc -mtriple=i686-windows-msvc | FileCheck %s --check-prefix=X86

@str = linkonce_odr unnamed_addr constant [27 x i8] c"GetExceptionCode(): 0x%lx\0A\00", align 1

declare i32 @__C_specific_handler(...)
declare void @crash()
declare i32 @printf(i8* nocapture readonly, ...) nounwind

define i32 @main() {
entry:
  invoke void @crash()
          to label %__try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
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

; X64-LABEL: main:
; X64: callq crash
; X64: retq
; X64: # Block address taken
; X64: leaq str(%rip), %rcx
; X64: movl %eax, %edx
; X64: callq printf

; X64: .seh_handlerdata
; X64-NEXT: .long 1
; X64-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL
; X64-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL+1
; X64-NEXT: .long 1
; X64-NEXT: .long .Ltmp{{[0-9]+}}@IMGREL

; X86-LABEL: _main:
; 	The EH code load should be this offset +4.
; X86: movl %esp, -24(%ebp)
; X86: movl $L__ehtable$main,
; 	EH state 0
; X86: movl $0, -4(%ebp)
; X86: calll _crash
; X86: retl
; X86: # Block address taken
; X86: movl -20(%ebp), %[[ptrs:[^ ,]*]]
; X86: movl (%[[ptrs]]), %[[rec:[^ ,]*]]
; X86: movl (%[[rec]]), %[[code:[^ ,]*]]
; 	EH state -1
; X86: movl $-1, -4(%ebp)
; X86-DAG: movl %[[code]], 4(%esp)
; X86-DAG: movl $_str, (%esp)
; X86: calll _printf

; X86: .section .xdata,"dr"
; X86-NEXT: L__ehtable$main
; X86-NEXT: .long -1
; X86-NEXT: .long 1
; X86-NEXT: .long Ltmp{{[0-9]+}}
