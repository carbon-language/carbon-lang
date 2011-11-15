; RUN: llc < %s -march=x86-64 | FileCheck %s

%struct.obj = type { i64 }

define void @_Z7releaseP3obj(%struct.obj* nocapture %o) nounwind uwtable ssp {
entry:
; CHECK: decq	(%rdi)
; CHECK-NEXT: je
  %refcnt = getelementptr inbounds %struct.obj* %o, i64 0, i32 0
  %0 = load i64* %refcnt, align 8, !tbaa !0
  %dec = add i64 %0, -1
  store i64 %dec, i64* %refcnt, align 8, !tbaa !0
  %tobool = icmp eq i64 %dec, 0
  br i1 %tobool, label %if.end, label %return

if.end:                                           ; preds = %entry
  %1 = bitcast %struct.obj* %o to i8*
  tail call void @free(i8* %1)
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

declare void @free(i8* nocapture) nounwind

!0 = metadata !{metadata !"long", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
