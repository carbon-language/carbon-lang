; RUN: llc < %s -march=x86-64 | FileCheck %s

%struct.obj = type { i64 }

; CHECK: _Z7releaseP3obj
define void @_Z7releaseP3obj(%struct.obj* nocapture %o) nounwind uwtable ssp {
entry:
; CHECK: decq	(%{{rdi|rcx}})
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

@c = common global i64 0, align 8
@a = common global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1
@b = common global i32 0, align 4

; CHECK: test
define i32 @test() nounwind uwtable ssp {
entry:
; CHECK: decq
; CHECK-NOT: decq
%0 = load i64* @c, align 8, !tbaa !0
%dec.i = add nsw i64 %0, -1
store i64 %dec.i, i64* @c, align 8, !tbaa !0
%tobool.i = icmp ne i64 %dec.i, 0
%lor.ext.i = zext i1 %tobool.i to i32
store i32 %lor.ext.i, i32* @a, align 4, !tbaa !3
%call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([5 x i8]* @.str, i64 0, i64 0), i64 %dec.i) nounwind
ret i32 0
}

; CHECK: test2
define i32 @test2() nounwind uwtable ssp {
entry:
; CHECK-NOT: decq ({{.*}})
%0 = load i64* @c, align 8, !tbaa !0
%dec.i = add nsw i64 %0, -1
store i64 %dec.i, i64* @c, align 8, !tbaa !0
%tobool.i = icmp ne i64 %0, 0
%lor.ext.i = zext i1 %tobool.i to i32
store i32 %lor.ext.i, i32* @a, align 4, !tbaa !3
%call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([5 x i8]* @.str, i64 0, i64 0), i64 %dec.i) nounwind
ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @free(i8* nocapture) nounwind

!0 = metadata !{metadata !"long", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"int", metadata !1}
