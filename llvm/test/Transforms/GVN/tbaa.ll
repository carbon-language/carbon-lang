; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

define i32 @test1(i8* %p, i8* %q) {
; CHECK: @test1(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p)
; CHECK-NOT: tbaa
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p)
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test2(i8* %p, i8* %q) {
; CHECK: @test2(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGC:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test3(i8* %p, i8* %q) {
; CHECK: @test3(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGB:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !3
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test4(i8* %p, i8* %q) {
; CHECK: @test4(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !1
  %b = call i32 @foo(i8* %p), !tbaa !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test5(i8* %p, i8* %q) {
; CHECK: @test5(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !1
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test6(i8* %p, i8* %q) {
; CHECK: @test6(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p), !tbaa [[TAGA:!.*]]
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !0
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test7(i8* %p, i8* %q) {
; CHECK: @test7(i8* %p, i8* %q)
; CHECK: call i32 @foo(i8* %p)
; CHECK-NOT: tbaa
; CHECK: %c = add i32 %a, %a
  %a = call i32 @foo(i8* %p), !tbaa !4
  %b = call i32 @foo(i8* %p), !tbaa !3
  %c = add i32 %a, %b
  ret i32 %c
}

declare i32 @foo(i8*) readonly

; CHECK: [[TAGC]] = metadata !{metadata [[TYPEC:!.*]], metadata [[TYPEC]], i64 0}
; CHECK: [[TYPEC]] = metadata !{metadata !"C", metadata [[TYPEA:!.*]]}
; CHECK: [[TYPEA]] = metadata !{metadata !"A", metadata !{{.*}}}
; CHECK: [[TAGB]] = metadata !{metadata [[TYPEB:!.*]], metadata [[TYPEB]], i64 0}
; CHECK: [[TYPEB]] = metadata !{metadata !"B", metadata [[TYPEA]]}
; CHECK: [[TAGA]] = metadata !{metadata [[TYPEA]], metadata [[TYPEA]], i64 0}
!0 = metadata !{metadata !5, metadata !5, i64 0}
!1 = metadata !{metadata !6, metadata !6, i64 0}
!2 = metadata !{metadata !"tbaa root", null}
!3 = metadata !{metadata !7, metadata !7, i64 0}
!4 = metadata !{metadata !8, metadata !8, i64 0}
!5 = metadata !{metadata !"C", metadata !6}
!6 = metadata !{metadata !"A", metadata !2}
!7 = metadata !{metadata !"B", metadata !6}
!8 = metadata !{metadata !"another root", null}
