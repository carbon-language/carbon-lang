; RUN: not --crash opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -atomic-expand %s 2>&1 | FileCheck %s
; The AtomicExpand pass cannot handle missing libcalls (yet) so reports a fatal error.
; CHECK: LLVM ERROR: expandAtomicOpToLibcall shouldn't fail for Load

define i32 @atomic_load_global_align1(i32 addrspace(1)* %ptr) {
; GCN-LABEL: @atomic_load_global_align1(
; GCN-NEXT:    [[TMP1:%.*]] = bitcast i32 addrspace(1)* [[PTR:%.*]] to i8 addrspace(1)*
; GCN-NEXT:    [[TMP2:%.*]] = addrspacecast i8 addrspace(1)* [[TMP1]] to i8*
; GCN-NEXT:    [[TMP3:%.*]] = alloca i32, align 4
; GCN-NEXT:    [[TMP4:%.*]] = bitcast i32* [[TMP3]] to i8*
; GCN-NEXT:    call void @llvm.lifetime.start.p0i8(i64 4, i8* [[TMP4]])
; GCN-NEXT:    call void @0(i64 4, i8* [[TMP2]], i8* [[TMP4]], i32 5)
; GCN-NEXT:    [[TMP5:%.*]] = load i32, i32* [[TMP3]], align 4
; GCN-NEXT:    call void @llvm.lifetime.end.p0i8(i64 4, i8* [[TMP4]])
; GCN-NEXT:    ret i32 [[TMP5]]
;
  %val = load atomic i32, i32 addrspace(1)* %ptr  seq_cst, align 1
  ret i32 %val
}

define void @atomic_store_global_align1(i32 addrspace(1)* %ptr, i32 %val) {
; GCN-LABEL: @atomic_store_global_align1(
; GCN-NEXT:    [[TMP1:%.*]] = bitcast i32 addrspace(1)* [[PTR:%.*]] to i8 addrspace(1)*
; GCN-NEXT:    [[TMP2:%.*]] = addrspacecast i8 addrspace(1)* [[TMP1]] to i8*
; GCN-NEXT:    [[TMP3:%.*]] = alloca i32, align 4
; GCN-NEXT:    [[TMP4:%.*]] = bitcast i32* [[TMP3]] to i8*
; GCN-NEXT:    call void @llvm.lifetime.start.p0i8(i64 4, i8* [[TMP4]])
; GCN-NEXT:    store i32 [[VAL:%.*]], i32* [[TMP3]], align 4
; GCN-NEXT:    call void @1(i64 4, i8* [[TMP2]], i8* [[TMP4]], i32 0)
; GCN-NEXT:    call void @llvm.lifetime.end.p0i8(i64 4, i8* [[TMP4]])
; GCN-NEXT:    ret void
;
  store atomic i32 %val, i32 addrspace(1)* %ptr monotonic, align 1
  ret void
}
