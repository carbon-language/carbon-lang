; RUN: opt -licm -S <%s | FileCheck %s
; Test that licm doesn't sink/delete debug info.
define i32 @foo(i32 %a, i32 %j) nounwind {
entry:
;CHECK: entry:
  call void @llvm.dbg.func.start(metadata !0)
  call void @llvm.dbg.stoppoint(i32 3, i32 5, metadata !1)
;CHECK: %mul = mul i32 %j, %j
  br label %for.cond

for.cond:
;CHECK: for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %s.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  call void @llvm.dbg.stoppoint(i32 4, i32 5, metadata !1)
; CHECK: call void @llvm.dbg.stoppoint(i32 4, i32 5, metadata !1)
  %cmp = icmp slt i32 %i.0, %a
  br i1 %cmp, label %for.body, label %for.end

for.body:
;CHECK: for.body:
  call void @llvm.dbg.stoppoint(i32 5, i32 2, metadata !1)
;CHECK: call void @llvm.dbg.stoppoint(i32 5, i32 2, metadata !1)
  %mul = mul i32 %j, %j
  %add = add nsw i32 %s.0, %mul
  br label %for.inc

for.inc:
;CHECK: for.inc:
  call void @llvm.dbg.stoppoint(i32 4, i32 18, metadata !1)
;CHECK: call void @llvm.dbg.stoppoint(i32 4, i32 18, metadata !1)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  call void @llvm.dbg.stoppoint(i32 7, i32 5, metadata !1)
  br label %0

; <label>:0                                       ; preds = %for.end
  call void @llvm.dbg.stoppoint(i32 8, i32 1, metadata !1)
  call void @llvm.dbg.region.end(metadata !0)
  ret i32 %s.0
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.declare({ }*, metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!0 = metadata !{i32 458798, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 2, metadata !2, i1 false, i1 true}; [DW_TAG_subprogram ]
!1 = metadata !{i32 458769, i32 0, i32 12, metadata !"licm.c", metadata !"/home/edwin", metadata !"clang 1.1", i1 true, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!2 = metadata !{i32 458788, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5}; [DW_TAG_base_type ]
