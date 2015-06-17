; RUN: llc -mtriple=i686-pc-windows-msvc < %s | FileCheck %s

; Based on this source:
; extern "C" void may_throw(int);
; void f() {
;   try {
;     may_throw(1);
;     try {
;       may_throw(2);
;     } catch (int) {
;       may_throw(3);
;     }
;   } catch (int) {
;     may_throw(4);
;   }
; }

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchHandlerType = type { i32, i8* }

declare void @may_throw(i32)
declare i32 @__CxxFrameHandler3(...)
declare void @llvm.eh.begincatch(i8*, i8*)
declare void @llvm.eh.endcatch()
declare i32 @llvm.eh.typeid.for(i8*)

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"

define void @f() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @may_throw(i32 1)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  invoke void @may_throw(i32 2)
          to label %try.cont.9 unwind label %lpad.1

try.cont.9:                                       ; preds = %invoke.cont.3, %invoke.cont, %catch.7
  ; FIXME: Something about our CFG breaks TailDuplication. This empy asm blocks
  ; it so we can focus on testing the state numbering.
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"()
  ret void

lpad:                                             ; preds = %catch, %entry
  %0 = landingpad { i8*, i32 }
          catch %eh.CatchHandlerType* @llvm.eh.handlertype.H.0
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  br label %catch.dispatch.4

lpad.1:                                           ; preds = %invoke.cont
  %3 = landingpad { i8*, i32 }
          catch i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)
  %4 = extractvalue { i8*, i32 } %3, 0
  %5 = extractvalue { i8*, i32 } %3, 1
  %6 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)) #3
  %matches = icmp eq i32 %5, %6
  br i1 %matches, label %catch, label %catch.dispatch.4

catch.dispatch.4:                                 ; preds = %lpad.1, %lpad
  %exn.slot.0 = phi i8* [ %4, %lpad.1 ], [ %1, %lpad ]
  %ehselector.slot.0 = phi i32 [ %5, %lpad.1 ], [ %2, %lpad ]
  %.pre = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%eh.CatchHandlerType* @llvm.eh.handlertype.H.0 to i8*)) #3
  %matches6 = icmp eq i32 %ehselector.slot.0, %.pre
  br i1 %matches6, label %catch.7, label %eh.resume

catch.7:                                          ; preds = %catch.dispatch.4
  tail call void @llvm.eh.begincatch(i8* %exn.slot.0, i8* null) #3
  tail call void @may_throw(i32 4)
  tail call void @llvm.eh.endcatch() #3
  br label %try.cont.9

catch:                                            ; preds = %lpad.1
  tail call void @llvm.eh.begincatch(i8* %4, i8* null) #3
  invoke void @may_throw(i32 3)
          to label %invoke.cont.3 unwind label %lpad

invoke.cont.3:                                    ; preds = %catch
  tail call void @llvm.eh.endcatch() #3
  br label %try.cont.9

eh.resume:                                        ; preds = %catch.dispatch.4
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val.12 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val.12
}

; CHECK-LABEL: _f:
; CHECK: movl $-1, [[state:[-0-9]+]](%ebp)
; CHECK: movl $___ehhandler$f, {{.*}}
;
; CHECK: movl $0, [[state]](%ebp)
; CHECK: movl $1, (%esp)
; CHECK: calll _may_throw
;
; CHECK: movl $1, [[state]](%ebp)
; CHECK: movl $2, (%esp)
; CHECK: calll _may_throw

; CHECK-LABEL: _f.catch:
; CHECK: movl $4, Lf$frame_escape_{{[0-9]+.*}}
; CHECK: movl $4, (%esp)
; CHECK: calll _may_throw

; CHECK-LABEL: _f.catch.1:
; CHECK: movl $3, Lf$frame_escape_{{[0-9]+.*}}
; CHECK: movl $3, (%esp)
; CHECK: calll _may_throw

; CHECK: .safeseh ___ehhandler$f
