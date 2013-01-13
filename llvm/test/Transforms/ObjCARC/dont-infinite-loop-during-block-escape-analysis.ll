; RUN: opt -S -objc-arc < %s
; bugzilla://14551
; rdar://12851911

; Make sure that we do not hang clang during escape analysis.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-darwin"

%struct.__block_descriptor = type { i64, i64 }
%struct.__block_byref_foo = type { i8*, %struct.__block_byref_foo*, i32, i32, i32 }

@_NSConcreteGlobalBlock = external global i8*
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1
@__block_descriptor_tmp = internal constant { i64, i64, i8*, i8* } { i64 0, i64 32, i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i8* null }
@__block_literal_global = internal constant { i8**, i32, i32, i8*, %struct.__block_descriptor* } { i8** @_NSConcreteGlobalBlock, i32 1342177280, i32 0, i8* bitcast (void (i8*)* @__hang_clang_block_invoke to i8*), %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8* }* @__block_descriptor_tmp to %struct.__block_descriptor*) }, align 8

define void @hang_clang() uwtable optsize ssp {
entry:
  %foo = alloca %struct.__block_byref_foo, align 8
  %byref.isa = getelementptr inbounds %struct.__block_byref_foo* %foo, i64 0, i32 0
  store i8* null, i8** %byref.isa, align 8
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_foo* %foo, i64 0, i32 1
  store %struct.__block_byref_foo* %foo, %struct.__block_byref_foo** %byref.forwarding, align 8
  %byref.flags = getelementptr inbounds %struct.__block_byref_foo* %foo, i64 0, i32 2
  store i32 536870912, i32* %byref.flags, align 8
  %byref.size = getelementptr inbounds %struct.__block_byref_foo* %foo, i64 0, i32 3
  store i32 32, i32* %byref.size, align 4
  %foo1 = getelementptr inbounds %struct.__block_byref_foo* %foo, i64 0, i32 4
  store i32 0, i32* %foo1, align 8, !tbaa !4
  br label %for.body

for.body:                                         ; preds = %for.inc.for.body_crit_edge, %entry
  %0 = phi i1 [ true, %entry ], [ %phitmp, %for.inc.for.body_crit_edge ]
  %i.06 = phi i32 [ 1, %entry ], [ %phitmp8, %for.inc.for.body_crit_edge ]
  %block.05 = phi void (...)* [ null, %entry ], [ %block.1, %for.inc.for.body_crit_edge ]
  br i1 %0, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %1 = call i8* @objc_retainBlock(i8* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global to i8*)) nounwind, !clang.arc.copy_on_escape !7
  %2 = bitcast i8* %1 to void (...)*
  %3 = bitcast void (...)* %block.05 to i8*
  call void @objc_release(i8* %3) nounwind, !clang.imprecise_release !7
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %block.1 = phi void (...)* [ %2, %if.then ], [ %block.05, %for.body ]
  %exitcond = icmp eq i32 %i.06, 10
  br i1 %exitcond, label %for.end, label %for.inc.for.body_crit_edge

for.inc.for.body_crit_edge:                       ; preds = %for.inc
  %.pre = load %struct.__block_byref_foo** %byref.forwarding, align 8
  %foo2.phi.trans.insert = getelementptr inbounds %struct.__block_byref_foo* %.pre, i64 0, i32 4
  %.pre7 = load i32* %foo2.phi.trans.insert, align 4, !tbaa !4
  %phitmp = icmp eq i32 %.pre7, 0
  %phitmp8 = add i32 %i.06, 1
  br label %for.body

for.end:                                          ; preds = %for.inc
  %4 = bitcast %struct.__block_byref_foo* %foo to i8*
  call void @_Block_object_dispose(i8* %4, i32 8)
  %5 = bitcast void (...)* %block.1 to i8*
  call void @objc_release(i8* %5) nounwind, !clang.imprecise_release !7
  ret void
}

define internal void @__hang_clang_block_invoke(i8* nocapture %.block_descriptor) nounwind uwtable readnone optsize ssp {
entry:
  ret void
}

declare i8* @objc_retainBlock(i8*)

declare void @objc_release(i8*) nonlazybind

declare void @_Block_object_dispose(i8*, i32)

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!1 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!2 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!4 = metadata !{metadata !"int", metadata !5}
!5 = metadata !{metadata !"omnipotent char", metadata !6}
!6 = metadata !{metadata !"Simple C/C++ TBAA"}
!7 = metadata !{}
