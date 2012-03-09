; RUN: opt -objc-arc -S < %s | FileCheck %s

%struct.__objcFastEnumerationState = type { i64, i8**, i64*, [5 x i64] }

@"\01L_OBJC_METH_VAR_NAME_" = internal global [43 x i8] c"countByEnumeratingWithState:objects:count:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global i8* getelementptr inbounds ([43 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i64 0, i64 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@g = common global i8* null, align 8
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"

declare void @callee()
declare i8* @returner()
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_retain(i8*)
declare void @objc_enumerationMutation(i8*)
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind
declare void @use(i8*)
declare void @objc_release(i8*)

!0 = metadata !{}

; Delete a nested retain+release pair.

; CHECK: define void @test0(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test0(i8* %a) nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %0 = call i8* @objc_retain(i8* %a) nounwind
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp2, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call, %forcoll.loopinit ], [ %call6, %forcoll.refetch ]
  %tmp7 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp7, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr3 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr3, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  call void @use(i8* %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp5 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call6 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp5, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call6, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK: define void @test2(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test2() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp2, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  call void @use(i8* %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK: define void @test4(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test4() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %tmp = load i8** @g, align 8
  %0 = call i8* @objc_retain(i8* %tmp) nounwind
  %tmp2 = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp2, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp4 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp4, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call, %forcoll.loopinit ], [ %call8, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr5 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr5, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  call void @use(i8* %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp7 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call8 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp7, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call8, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK: define void @test5(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test5() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  call void @callee()
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp2, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  call void @use(i8* %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK: define void @test6(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test6() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp2, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  call void @use(i8* %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @callee()
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK: define void @test7(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test7() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  call void @callee()
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp2, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.notmutated ]
  %mutationsptr4 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  call void @use(i8* %3)
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @callee()
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Delete a nested retain+release pair.

; CHECK: define void @test8(
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test8() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %1 = call i8* @objc_retain(i8* %0) nounwind
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call3 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp2, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call3, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  %stateitems.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 1
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call3, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp8 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp8, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ 0, %forcoll.loopbody.outer ], [ %4, %forcoll.next ]
  %mutationsptr4 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr4, align 8
  %2 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %2, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %1)
  br label %forcoll.notmutated

forcoll.notmutated:
  %stateitems = load i8*** %stateitems.ptr, align 8
  %currentitem.ptr = getelementptr i8** %stateitems, i64 %forcoll.index
  %3 = load i8** %currentitem.ptr, align 8
  %tobool = icmp eq i8* %3, null
  br i1 %tobool, label %forcoll.next, label %if.then

if.then:
  call void @callee()
  br label %forcoll.next

forcoll.next:
  %4 = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %4, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %1, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %5 = icmp eq i64 %call7, 0
  br i1 %5, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %1) nounwind
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; TODO: Delete a nested retain+release pair.
; The optimizer currently can't do this, because of a split loop backedge.
; See test9b for the same testcase without a split backedge.

; CHECK: define void @test9(
; CHECK: call i8* @objc_retain
; CHECK: call i8* @objc_retain
; CHECK: call i8* @objc_retain
; CHECK: }
define void @test9() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %call1 = call i8* @returner()
  %1 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call1) nounwind
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %2 = call i8* @objc_retain(i8* %0) nounwind
  %tmp3 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp3, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated.forcoll.loopbody_crit_edge ], [ 1, %forcoll.loopbody.outer ]
  %mutationsptr5 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %exitcond = icmp eq i64 %forcoll.index, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.notmutated.forcoll.loopbody_crit_edge

forcoll.notmutated.forcoll.loopbody_crit_edge:
  %phitmp = add i64 %forcoll.index, 1
  br label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %2) nounwind
  call void @objc_release(i8* %1) nounwind, !clang.imprecise_release !0
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test9, but without a split backedge. This we can optimize.

; CHECK: define void @test9b(
; CHECK: call i8* @objc_retain
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test9b() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %call1 = call i8* @returner()
  %1 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call1) nounwind
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %2 = call i8* @objc_retain(i8* %0) nounwind
  %tmp3 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp3, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated ], [ 0, %forcoll.loopbody.outer ]
  %mutationsptr5 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %phitmp = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %phitmp, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %2) nounwind
  call void @objc_release(i8* %1) nounwind, !clang.imprecise_release !0
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; TODO: Delete a nested retain+release pair.
; The optimizer currently can't do this, because of a split loop backedge.
; See test10b for the same testcase without a split backedge.

; CHECK: define void @test10(
; CHECK: call i8* @objc_retain
; CHECK: call i8* @objc_retain
; CHECK: call i8* @objc_retain
; CHECK: }
define void @test10() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %call1 = call i8* @returner()
  %1 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call1) nounwind
  call void @callee()
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %2 = call i8* @objc_retain(i8* %0) nounwind
  %tmp3 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp3, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated.forcoll.loopbody_crit_edge ], [ 1, %forcoll.loopbody.outer ]
  %mutationsptr5 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %exitcond = icmp eq i64 %forcoll.index, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.notmutated.forcoll.loopbody_crit_edge

forcoll.notmutated.forcoll.loopbody_crit_edge:
  %phitmp = add i64 %forcoll.index, 1
  br label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %2) nounwind
  call void @objc_release(i8* %1) nounwind, !clang.imprecise_release !0
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test10, but without a split backedge. This we can optimize.

; CHECK: define void @test10b(
; CHECK: call i8* @objc_retain
; CHECK: call i8* @objc_retain
; CHECK-NOT: @objc_retain
; CHECK: }
define void @test10b() nounwind {
entry:
  %state.ptr = alloca %struct.__objcFastEnumerationState, align 8
  %items.ptr = alloca [16 x i8*], align 8
  %call = call i8* @returner()
  %0 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %call1 = call i8* @returner()
  %1 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call1) nounwind
  call void @callee()
  %tmp = bitcast %struct.__objcFastEnumerationState* %state.ptr to i8*
  call void @llvm.memset.p0i8.i64(i8* %tmp, i8 0, i64 64, i32 8, i1 false)
  %2 = call i8* @objc_retain(i8* %0) nounwind
  %tmp3 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call4 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp3, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %iszero = icmp eq i64 %call4, 0
  br i1 %iszero, label %forcoll.empty, label %forcoll.loopinit

forcoll.loopinit:
  %mutationsptr.ptr = getelementptr inbounds %struct.__objcFastEnumerationState* %state.ptr, i64 0, i32 2
  %mutationsptr = load i64** %mutationsptr.ptr, align 8
  %forcoll.initial-mutations = load i64* %mutationsptr, align 8
  br label %forcoll.loopbody.outer

forcoll.loopbody.outer:
  %forcoll.count.ph = phi i64 [ %call4, %forcoll.loopinit ], [ %call7, %forcoll.refetch ]
  %tmp9 = icmp ugt i64 %forcoll.count.ph, 1
  %umax = select i1 %tmp9, i64 %forcoll.count.ph, i64 1
  br label %forcoll.loopbody

forcoll.loopbody:
  %forcoll.index = phi i64 [ %phitmp, %forcoll.notmutated ], [ 0, %forcoll.loopbody.outer ]
  %mutationsptr5 = load i64** %mutationsptr.ptr, align 8
  %statemutations = load i64* %mutationsptr5, align 8
  %3 = icmp eq i64 %statemutations, %forcoll.initial-mutations
  br i1 %3, label %forcoll.notmutated, label %forcoll.mutated

forcoll.mutated:
  call void @objc_enumerationMutation(i8* %2)
  br label %forcoll.notmutated

forcoll.notmutated:
  %phitmp = add i64 %forcoll.index, 1
  %exitcond = icmp eq i64 %phitmp, %umax
  br i1 %exitcond, label %forcoll.refetch, label %forcoll.loopbody

forcoll.refetch:
  %tmp6 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8
  %call7 = call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)(i8* %2, i8* %tmp6, %struct.__objcFastEnumerationState* %state.ptr, [16 x i8*]* %items.ptr, i64 16)
  %4 = icmp eq i64 %call7, 0
  br i1 %4, label %forcoll.empty, label %forcoll.loopbody.outer

forcoll.empty:
  call void @objc_release(i8* %2) nounwind
  call void @objc_release(i8* %1) nounwind, !clang.imprecise_release !0
  call void @objc_release(i8* %0) nounwind, !clang.imprecise_release !0
  ret void
}
