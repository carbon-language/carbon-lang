; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck --check-prefix=X86 %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck --check-prefix=X64 %s

; Loosely based on IR for this C++ source code:
;   void f(int p);
;   int main() {
;     try {
;       f(1);
;     } catch (int) {
;       f(2);
;     } catch (...) {
;       f(3);
;     }
;   }

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i8*, i32, i32, i32, i32, i8* }
%eh.CatchableTypeArray.1 = type { i32, [1 x %eh.CatchableType*] }
%eh.ThrowInfo = type { i32, i8*, i8*, i8* }
%eh.CatchHandlerType = type { i32, i8* }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"
@llvm.eh.handlertype.H.1 = private unnamed_addr constant %eh.CatchHandlerType { i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"

declare void @f(i32 %p, i32* %l)
declare i32 @__CxxFrameHandler3(...)

define i32 @try_catch_catch() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %local = alloca i32
  invoke void @f(i32 1, i32* %local)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [%eh.CatchHandlerType* @llvm.eh.handlertype.H.0, i8* null]
          to label %catch unwind label %catch.dispatch.2

catch:                                            ; preds = %catch.dispatch
  invoke void @f(i32 2, i32* %local)
          to label %invoke.cont.2 unwind label %catchendblock

invoke.cont.2:                                    ; preds = %catch
  catchret %0 to label %try.cont

catch.dispatch.2:                                   ; preds = %catch.dispatch
  %1 = catchpad [%eh.CatchHandlerType* @llvm.eh.handlertype.H.0, i8* null]
          to label %catch.2 unwind label %catchendblock

catch.2:                                            ; preds = %catch.dispatch.2
  invoke void @f(i32 3, i32* %local)
          to label %invoke.cont.3 unwind label %catchendblock

invoke.cont.3:                                    ; preds = %catch.2
  catchret %1 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.2, %invoke.cont.3
  ret i32 0

catchendblock:                                    ; preds = %catch, %catch.2, %catch.dispatch.2
  catchendpad unwind to caller
}

; X86-LABEL: _try_catch_catch:
; X86: movl $0, -{{[0-9]+}}(%ebp)
; X86: leal -[[local_offs:[0-9]+]](%ebp), %[[addr_reg:[a-z]+]]
; X86-DAG: movl %[[addr_reg]], 4(%esp)
; X86-DAG: movl $1, (%esp)
; X86: calll _f
; X86: [[contbb:Ltmp[0-9]+]]: # Block address taken
; X86: movl -{{[0-9]+}}(%ebp), %esp
; X86: retl

; X86: [[catch1bb:LBB0_[0-9]+]]: # %catch{{$}}
; X86: addl $12, %ebp
; X86: subl $8, %esp
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86: leal -[[local_offs]](%ebp), %[[addr_reg:[a-z]+]]
; X86-DAG: movl %[[addr_reg]], 4(%esp)
; X86-DAG: movl $2, (%esp)
; X86: calll _f
; X86: movl $[[contbb]], %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: retl

; X86: [[catch2bb:LBB0_[0-9]+]]: # %catch.2{{$}}
; X86: addl $12, %ebp
; X86: subl $8, %esp
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86: leal -[[local_offs]](%ebp), %[[addr_reg:[a-z]+]]
; X86-DAG: movl %[[addr_reg]], 4(%esp)
; X86-DAG: movl $3, (%esp)
; X86: calll _f
; X86: movl $[[contbb]], %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: retl

; X86: L__ehtable$try_catch_catch:
; X86: $handlerMap$0$try_catch_catch:
; X86:   .long   0
; X86:   .long   "??_R0H@8"
; X86:   .long   0
; X86:   .long   [[catch1bb]]
; X86:   .long   0
; X86:   .long   "??_R0H@8"
; X86:   .long   0
; X86:   .long   [[catch2bb]]

; X64-LABEL: try_catch_catch:
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $48, %rsp
; X64: .seh_stackalloc 48
; X64: leaq 48(%rsp), %rbp
; X64: .seh_setframe 5, 48
; X64-DAG: leaq -[[local_offs:[0-9]+]](%rbp), %rdx
; X64-DAG: movl $1, %ecx
; X64: callq f
; X64: [[contbb:.Ltmp[0-9]+]]: # Block address taken
; X64: addq $48, %rsp
; X64: popq %rbp
; X64: retq

; X64: [[catch1bb:\.LBB0_[0-9]+]]: # %catch{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: movq %rdx, %rbp
; X64: subq $32, %rsp
; X64-DAG: leaq -[[local_offs]](%rbp), %rdx
; X64-DAG: movl $2, %ecx
; X64: callq f
; X64: leaq [[contbb]](%rip), %rax
; X64: addq $32, %rsp
; X64: popq %rbp
; X64: retq

; X64: [[catch2bb:\.LBB0_[0-9]+]]: # %catch.2{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: movq %rdx, %rbp
; X64: subq $32, %rsp
; X64-DAG: leaq -[[local_offs]](%rbp), %rdx
; X64-DAG: movl $3, %ecx
; X64: callq f
; X64: leaq [[contbb]](%rip), %rax
; X64: addq $32, %rsp
; X64: popq %rbp
; X64: retq

; X64: $handlerMap$0$try_catch_catch:
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   [[catch1bb]]@IMGREL
; X64:   .long   56
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   [[catch2bb]]@IMGREL
; X64:   .long   56
