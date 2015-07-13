; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=CHECK --check-prefix=X64
; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s --check-prefix=CHECK --check-prefix=X86

; This test case is equivalent to:
; void f() {
;   try {
;     try {
;       may_throw();
;     } catch (int &) {
;       may_throw();
;     }
;     may_throw();
;   } catch (double) {
;   }
; }


%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchHandlerType = type { i32, i8* }

$"\01??_R0N@8" = comdat any

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0N@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".N\00" }, comdat
@llvm.eh.handlertype.N.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0N@8" to i8*) }, section "llvm.metadata"
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@llvm.eh.handlertype.H.8 = private unnamed_addr constant %eh.CatchHandlerType { i32 8, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"

define internal i8* @"\01?f@@YAXXZ.catch"(i8*, i8*) #4 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %.i8 = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?f@@YAXXZ" to i8*), i8* %1, i32 0)
  %bc2 = bitcast i8* %.i8 to i32**
  %bc3 = bitcast i32** %bc2 to i8*
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %entry
  ret i8* blockaddress(@"\01?f@@YAXXZ", %try.cont)

lpad1:                                            ; preds = %entry
  %lp4 = landingpad { i8*, i32 }
          cleanup
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.N.0
  %recover = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.N.0 to i8*), i32 1, i8* (i8*, i8*)* @"\01?f@@YAXXZ.catch1")
  indirectbr i8* %recover, [label %invoke.cont2]
}

; CHECK-LABEL: "?f@@YAXXZ.catch":
; No code should be generated for the indirectbr.
; CHECK-NOT: jmp{{[ql]}} *
; X64:        .seh_handlerdata
; X64-NEXT:        .long   ("$cppxdata$?f@@YAXXZ")@IMGREL


define internal i8* @"\01?f@@YAXXZ.catch1"(i8*, i8*) #4 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %.i8 = call i8* @llvm.localrecover(i8* bitcast (void ()* @"\01?f@@YAXXZ" to i8*), i8* %1, i32 1)
  %2 = bitcast i8* %.i8 to double*
  %3 = bitcast double* %2 to i8*
  invoke void () @llvm.donothing()
          to label %done unwind label %lpad

done:
  ret i8* blockaddress(@"\01?f@@YAXXZ", %try.cont8)

lpad:                                             ; preds = %entry
  %4 = landingpad { i8*, i32 }
          cleanup
  %recover = call i8* (...) @llvm.eh.actions()
  unreachable
}

; CHECK-LABEL: "?f@@YAXXZ.catch1":
; No code should be generated for the indirectbr.
; CHECK-NOT: jmp{{[ql]}} *
; X64: ".L?f@@YAXXZ.catch1$parent_frame_offset" = 16
; X64:         movq    %rdx, 16(%rsp)
; X64:        .seh_handlerdata
; X64:        .long   ("$cppxdata$?f@@YAXXZ")@IMGREL

define void @"\01?f@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %0 = alloca i32*, align 8
  %1 = alloca double, align 8
  call void (...) @llvm.localescape(i32** %0, double* %1)
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad2

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad2:                                            ; preds = %entry
  %2 = landingpad { i8*, i32 }
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.8
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.N.0
  %recover = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.8 to i8*), i32 0, i8* (i8*, i8*)* @"\01?f@@YAXXZ.catch", i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.N.0 to i8*), i32 1, i8* (i8*, i8*)* @"\01?f@@YAXXZ.catch1")
  indirectbr i8* %recover, [label %try.cont, label %try.cont8]

try.cont:                                         ; preds = %lpad2, %invoke.cont
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %try.cont8 unwind label %lpad1

lpad1:
  %3 = landingpad { i8*, i32 }
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.N.0
  %recover2 = call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.N.0 to i8*), i32 1, i8* (i8*, i8*)* @"\01?f@@YAXXZ.catch1")
  indirectbr i8* %recover2, [label %try.cont8]

try.cont8:                                        ; preds = %lpad2, %try.cont
  ret void
}

; CHECK-LABEL: "?f@@YAXXZ":
; No code should be generated for the indirectbr.
; CHECK-NOT: jmp{{[ql]}} *

; X64:             .seh_handlerdata
; X64-NEXT:        .long   ("$cppxdata$?f@@YAXXZ")@IMGREL
; X86:             .section .xdata,"dr"

; CHECK: .align 4

; X64: "$cppxdata$?f@@YAXXZ":
; X64-NEXT:          .long   429065506
; X64-NEXT:          .long   4
; X64-NEXT:          .long   ("$stateUnwindMap$?f@@YAXXZ")@IMGREL
; X64-NEXT:          .long   2
; X64-NEXT:          .long   ("$tryMap$?f@@YAXXZ")@IMGREL
; X64-NEXT:          .long   6
; X64-NEXT:          .long   ("$ip2state$?f@@YAXXZ")@IMGREL
; X64-NEXT:          .long   32
; X64-NEXT:          .long   0
; X64-NEXT:          .long   1

; X86: "L__ehtable$?f@@YAXXZ":
; X86-NEXT:          .long   429065506
; X86-NEXT:          .long   4
; X86-NEXT:          .long   ("$stateUnwindMap$?f@@YAXXZ")
; X86-NEXT:          .long   2
; X86-NEXT:          .long   ("$tryMap$?f@@YAXXZ")
; X86-NEXT:          .long   0
; X86-NEXT:          .long   0
; X86-NEXT:          .long   0
; X86-NEXT:          .long   1


; CHECK-NEXT:"$stateUnwindMap$?f@@YAXXZ":
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   0
; CHECK-NEXT:"$tryMap$?f@@YAXXZ":
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   ("$handlerMap$0$?f@@YAXXZ")
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   ("$handlerMap$1$?f@@YAXXZ")
; CHECK-NEXT:"$handlerMap$0$?f@@YAXXZ":
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   "??_R0H@8"
; CHECK-NEXT:        .long   "{{.?}}L?f@@YAXXZ$frame_escape_0"
; CHECK-NEXT:        .long   "?f@@YAXXZ.catch"
; X64-NEXT:          .long   ".L?f@@YAXXZ.catch$parent_frame_offset"
; CHECK-NEXT:"$handlerMap$1$?f@@YAXXZ":
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   "??_R0N@8"
; CHECK-NEXT:        .long   "{{.?}}L?f@@YAXXZ$frame_escape_1"
; CHECK-NEXT:        .long   "?f@@YAXXZ.catch1"
; X64-NEXT:          .long   ".L?f@@YAXXZ.catch1$parent_frame_offset"

; X64-NEXT:"$ip2state$?f@@YAXXZ":
; X64-NEXT:        .long   .Lfunc_begin0
; X64-NEXT:        .long   2
; X64-NEXT:        .long   .Ltmp0
; X64-NEXT:        .long   0
; X64-NEXT:        .long   .Lfunc_begin1
; X64-NEXT:        .long   3
; X64-NEXT:        .long   .Lfunc_begin2
; X64-NEXT:        .long   -1
; X64-NEXT:        .long   .Ltmp13
; X64-NEXT:        .long   1
; X64-NEXT:        .long   .Ltmp16
; X64-NEXT:        .long   0


; X86: "___ehhandler$?f@@YAXXZ": # @"__ehhandler$?f@@YAXXZ"
; X86: movl $"L__ehtable$?f@@YAXXZ", %eax
; X86: jmp ___CxxFrameHandler3 # TAILCALL


declare void @"\01?may_throw@@YAXXZ"() #1

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

; Function Attrs: nounwind
declare i8* @llvm.eh.actions(...) #3

; Function Attrs: nounwind
declare void @llvm.localescape(...) #3

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #2

declare void @llvm.donothing()

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" "wineh-parent"="?f@@YAXXZ" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
attributes #4 = { "wineh-parent"="?f@@YAXXZ" }
