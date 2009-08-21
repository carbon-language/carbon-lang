; RUN: llvm-as < %s | llc
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:16:16-f128:128:128"
target triple = "s390x-ibm-linux-gnu"

%0 = type { i64 }
%struct.anon = type { %struct.dwarf_fde* }
%struct.dwarf_fde = type <{ i32, i32, [0 x i8] }>
%struct.object = type { i8*, i8*, i8*, %struct.anon, %0, %struct.object* }

@__dso_handle = hidden global i8* null            ; <i8**> [#uses=0]
@__CTOR_LIST__ = internal global [1 x void ()*] [void ()* inttoptr (i64 -1 to void ()*)], section ".ctors", align 8 ; <[1 x void ()*]*> [#uses=1]
@__DTOR_LIST__ = internal global [1 x void ()*] [void ()* inttoptr (i64 -1 to void ()*)], section ".dtors", align 8 ; <[1 x void ()*]*> [#uses=2]
@__EH_FRAME_BEGIN__ = internal constant [0 x i8] zeroinitializer, section ".eh_frame", align 4 ; <[0 x i8]*> [#uses=1]
@__JCR_LIST__ = internal global [0 x i8*] zeroinitializer, section ".jcr", align 8 ; <[0 x i8*]*> [#uses=2]
@completed.1298.b = internal global i1 false      ; <i1*> [#uses=2]
@p.1296 = internal global void ()** getelementptr ([1 x void ()*]* @__DTOR_LIST__, i64 1, i64 0) ; <void ()***> [#uses=3]
@object.1314 = internal global %struct.object zeroinitializer ; <%struct.object*> [#uses=1]
@llvm.used = appending global [7 x i8*] [i8* bitcast ([1 x void ()*]* @__CTOR_LIST__ to i8*), i8* bitcast ([1 x void ()*]* @__DTOR_LIST__ to i8*), i8* getelementptr inbounds ([0 x i8]* @__EH_FRAME_BEGIN__, i32 0, i32 0), i8* bitcast (void ()* @__do_global_dtors_aux to i8*), i8* bitcast (void ()* @call___do_global_dtors_aux to i8*), i8* bitcast (void ()* @frame_dummy to i8*), i8* bitcast (void ()* @call_frame_dummy to i8*)], section "llvm.metadata" ; <[7 x i8*]*> [#uses=0]

define internal void @__do_global_dtors_aux() nounwind {
entry:
  %.b = load i1* @completed.1298.b                ; <i1> [#uses=1]
  br i1 %.b, label %return, label %bb1.preheader

bb1.preheader:                                    ; preds = %entry
  %0 = load void ()*** @p.1296, align 8           ; <void ()**> [#uses=2]
  %1 = load void ()** %0, align 8                 ; <void ()*> [#uses=2]
  %2 = icmp eq void ()* %1, null                  ; <i1> [#uses=1]
  br i1 %2, label %bb2, label %bb

bb:                                               ; preds = %bb, %bb1.preheader
  %3 = phi void ()** [ %0, %bb1.preheader ], [ %6, %bb ] ; <void ()**> [#uses=1]
  %4 = phi void ()* [ %1, %bb1.preheader ], [ %7, %bb ] ; <void ()*> [#uses=1]
  %5 = getelementptr inbounds void ()** %3, i64 1 ; <void ()**> [#uses=1]
  store void ()** %5, void ()*** @p.1296, align 8
  tail call void %4() nounwind
  %6 = load void ()*** @p.1296, align 8           ; <void ()**> [#uses=2]
  %7 = load void ()** %6, align 8                 ; <void ()*> [#uses=2]
  %8 = icmp eq void ()* %7, null                  ; <i1> [#uses=1]
  br i1 %8, label %bb2, label %bb

bb2:                                              ; preds = %bb, %bb1.preheader
  br i1 icmp ne (i8* (i8*)* @__deregister_frame_info, i8* (i8*)* null), label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  %9 = tail call i8* @__deregister_frame_info(i8* getelementptr inbounds ([0 x i8]* @__EH_FRAME_BEGIN__, i32 0, i32 0)) nounwind ; <i8*> [#uses=0]
  br label %bb4

bb4:                                              ; preds = %bb2, %bb3
  store i1 true, i1* @completed.1298.b
  ret void

return:                                           ; preds = %entry
  ret void
}

declare extern_weak i8* @__deregister_frame_info(i8*)

define internal void @call___do_global_dtors_aux() nounwind {
entry:
  tail call void asm sideeffect "\09.section\09.fini", ""() nounwind
  tail call void @__do_global_dtors_aux() nounwind
  tail call void asm sideeffect ".text", ""() nounwind
  ret void
}

define internal void @frame_dummy() nounwind {
entry:
  br i1 icmp ne (void (i8*, %struct.object*)* @__register_frame_info, void (i8*, %struct.object*)* null), label %bb, label %bb1

bb:                                               ; preds = %entry
  tail call void @__register_frame_info(i8* getelementptr inbounds ([0 x i8]* @__EH_FRAME_BEGIN__, i32 0, i32 0), %struct.object* @object.1314) nounwind
  br label %bb1

bb1:                                              ; preds = %entry, %bb
  %0 = load i8** getelementptr inbounds ([0 x i8*]* @__JCR_LIST__, i64 0, i64 0), align 8 ; <i8*> [#uses=1]
  %1 = icmp eq i8* %0, null                       ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb2

bb2:                                              ; preds = %bb1
  %asmtmp = tail call void (i8*)* (void (i8*)*)* asm "", "=r,0"(void (i8*)* @_Jv_RegisterClasses) nounwind ; <void (i8*)*> [#uses=2]
  %2 = icmp eq void (i8*)* %asmtmp, null          ; <i1> [#uses=1]
  br i1 %2, label %return, label %bb3

bb3:                                              ; preds = %bb2
  tail call void %asmtmp(i8* bitcast ([0 x i8*]* @__JCR_LIST__ to i8*)) nounwind
  ret void

return:                                           ; preds = %bb2, %bb1
  ret void
}

declare extern_weak void @__register_frame_info(i8*, %struct.object*)

declare extern_weak void @_Jv_RegisterClasses(i8*)

define internal void @call_frame_dummy() nounwind {
entry:
  tail call void asm sideeffect "\09.section\09.init", ""() nounwind
  tail call void @frame_dummy() nounwind
  tail call void asm sideeffect ".text", ""() nounwind
  ret void
}
