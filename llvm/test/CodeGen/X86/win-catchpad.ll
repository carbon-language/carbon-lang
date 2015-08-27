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

declare void @f(i32 %p)
declare i32 @__CxxFrameHandler3(...)

define i32 @try_catch_catch() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @f(i32 1)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [%eh.CatchHandlerType* @llvm.eh.handlertype.H.0, i8* null]
          to label %catch unwind label %catch.dispatch.2

catch:                                            ; preds = %catch.dispatch
  invoke void @f(i32 2)
          to label %invoke.cont.2 unwind label %catchendblock

invoke.cont.2:                                    ; preds = %catch
  catchret %0 to label %try.cont

catch.dispatch.2:                                   ; preds = %catch.dispatch
  %1 = catchpad [%eh.CatchHandlerType* @llvm.eh.handlertype.H.0, i8* null]
          to label %catch.2 unwind label %catchendblock

catch.2:                                            ; preds = %catch.dispatch.2
  invoke void @f(i32 3)
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
; X86: movl $1, (%esp)
; X86: calll _f
; X86: [[contbb:LBB0_[0-9]+]]:
; X86: movl -{{[0-9]+}}(%ebp), %esp
; X86: retl

; X86: [[catch1bb:LBB0_[0-9]+]]: # %catch{{$}}
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86: movl $2, (%esp)
; X86: calll _f
; X86: movl $[[contbb]], %eax
; X86-NEXT: retl

; X86: [[catch2bb:LBB0_[0-9]+]]: # %catch.2{{$}}
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86: movl $3, (%esp)
; X86: calll _f
; X86: movl $[[contbb]], %eax
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
; X64: movl $1, %ecx
; X64: callq f
; X64: [[contbb:\.LBB0_[0-9]+]]:
; X64: retq

; X64: [[catch1bb:\.LBB0_[0-9]+]]: # %catch{{$}}
; X64: movl $2, %ecx
; X64: callq f
; X64: leaq [[contbb]](%rip), %rax
; X64: retq

; X64: [[catch2bb:\.LBB0_[0-9]+]]: # %catch.2{{$}}
; X64: movl $3, %ecx
; X64: callq f
; X64: leaq [[contbb]](%rip), %rax
; X64: retq

; FIXME: Get rid of these parent_frame_offset things below. They are leftover
; from our IR outlining strategy.
; X64: $handlerMap$0$try_catch_catch:
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   [[catch1bb]]@IMGREL
; X64    .long   .Lcatch$parent_frame_offset
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   [[catch2bb]]@IMGREL
; X64    .long   .Lcatch.2$parent_frame_offset
