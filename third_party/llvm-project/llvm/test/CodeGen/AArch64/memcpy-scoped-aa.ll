; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -stop-after=finalize-isel -o - %s | FileCheck --check-prefix=MIR %s

; MIR-DAG: ![[DOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"bax"}
; MIR-DAG: ![[SCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"bax: %p"}
; MIR-DAG: ![[SCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"bax: %q"}
; MIR-DAG: ![[SET0:[0-9]+]] = !{![[SCOPE0]]}
; MIR-DAG: ![[SET1:[0-9]+]] = !{![[SCOPE1]]}

; MIR-LABEL: name: test_memcpy
; MIR:      %2:fpr128 = LDRQui %0, 1 :: (load (s128) from %ir.p1, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: STRQui killed %2, %0, 0 :: (store (s128) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memcpy(i32* nocapture %p, i32* nocapture readonly %q) {
; CHECK-LABEL: test_memcpy:
; CHECK-DAG:    ldp [[Q0:w[0-9]+]], [[Q1:w[0-9]+]], [x1]
; CHECK-DAG:    ldr [[PVAL:q[0-9]+]], [x0, #16]
; CHECK-DAG:    add w8, [[Q0]], [[Q1]]
; CHECK:        str [[PVAL]], [x0]
; CHECK:        mov w0, w8
; CHECK:        ret
  %p0 = bitcast i32* %p to i8*
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 4
  %p1 = bitcast i32* %add.ptr to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %p0, i8* noundef nonnull align 4 dereferenceable(16) %p1, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, i32* %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, i32* %q, i64 1
  %v1 = load i32, i32* %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memcpy_inline
; MIR:      %2:fpr128 = LDRQui %0, 1 :: (load (s128) from %ir.p1, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: STRQui killed %2, %0, 0 :: (store (s128) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memcpy_inline(i32* nocapture %p, i32* nocapture readonly %q) {
; CHECK-LABEL: test_memcpy_inline:
; CHECK-DAG:    ldp [[Q0:w[0-9]+]], [[Q1:w[0-9]+]], [x1]
; CHECK-DAG:    ldr [[PVAL:q[0-9]+]], [x0, #16]
; CHECK-DAG:    add w8, [[Q0]], [[Q1]]
; CHECK:        str [[PVAL]], [x0]
; CHECK:        mov w0, w8
; CHECK:        ret
  %p0 = bitcast i32* %p to i8*
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 4
  %p1 = bitcast i32* %add.ptr to i8*
  tail call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %p0, i8* noundef nonnull align 4 dereferenceable(16) %p1, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, i32* %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, i32* %q, i64 1
  %v1 = load i32, i32* %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memmove
; MIR:      %2:fpr128 = LDRQui %0, 1 :: (load (s128) from %ir.p1, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: STRQui killed %2, %0, 0 :: (store (s128) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memmove(i32* nocapture %p, i32* nocapture readonly %q) {
; CHECK-LABEL: test_memmove:
; CHECK-DAG:    ldp [[Q0:w[0-9]+]], [[Q1:w[0-9]+]], [x1]
; CHECK-DAG:    ldr [[PVAL:q[0-9]+]], [x0, #16]
; CHECK-DAG:    add w8, [[Q0]], [[Q1]]
; CHECK:        str [[PVAL]], [x0]
; CHECK:        mov w0, w8
; CHECK:        ret
  %p0 = bitcast i32* %p to i8*
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 4
  %p1 = bitcast i32* %add.ptr to i8*
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %p0, i8* noundef nonnull align 4 dereferenceable(16) %p1, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, i32* %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, i32* %q, i64 1
  %v1 = load i32, i32* %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memset
; MIR:      %2:gpr64 = MOVi64imm -6148914691236517206
; MIR-NEXT: STRXui %2, %0, 1 :: (store (s64) into %ir.p0 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: STRXui %2, %0, 0 :: (store (s64) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memset(i32* nocapture %p, i32* nocapture readonly %q) {
; CHECK-LABEL: test_memset:
; CHECK-DAG:    ldp [[Q0:w[0-9]+]], [[Q1:w[0-9]+]], [x1]
; CHECK-DAG:    mov [[PVAL:x[0-9]+]], #-6148914691236517206
; CHECK:        stp [[PVAL]], [[PVAL]], [x0]
; CHECK:        add w8, [[Q0]], [[Q1]]
; CHECK:        mov w0, w8
; CHECK:        ret
  %p0 = bitcast i32* %p to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %p0, i8 170, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, i32* %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, i32* %q, i64 1
  %v1 = load i32, i32* %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_mempcpy
; MIR:      %2:fpr128 = LDRQui %0, 1 :: (load (s128) from %ir.p1, align 1, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: STRQui killed %2, %0, 0 :: (store (s128) into %ir.p0, align 1, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_mempcpy(i32* nocapture %p, i32* nocapture readonly %q) {
; CHECK-LABEL: test_mempcpy:
; CHECK-DAG:    ldp [[Q0:w[0-9]+]], [[Q1:w[0-9]+]], [x1]
; CHECK-DAG:    ldr [[PVAL:q[0-9]+]], [x0, #16]
; CHECK-DAG:    add w8, [[Q0]], [[Q1]]
; CHECK:        str [[PVAL]], [x0]
; CHECK:        mov w0, w8
; CHECK:        ret
  %p0 = bitcast i32* %p to i8*
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 4
  %p1 = bitcast i32* %add.ptr to i8*
  %call = tail call i8* @mempcpy(i8* noundef nonnull align 4 dereferenceable(16) %p0, i8* noundef nonnull align 4 dereferenceable(16) %p1, i64 16), !alias.scope !2, !noalias !4
  %v0 = load i32, i32* %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, i32* %q, i64 1
  %v1 = load i32, i32* %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)

declare i8* @mempcpy(i8*, i8*, i64)

!0 = distinct !{!0, !"bax"}
!1 = distinct !{!1, !0, !"bax: %p"}
!2 = !{!1}
!3 = distinct !{!3, !0, !"bax: %q"}
!4 = !{!3}
