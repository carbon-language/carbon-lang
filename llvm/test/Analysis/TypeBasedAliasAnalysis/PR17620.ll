; RUN: opt < %s -O2 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%"structA" = type { %"structB" }
%"structB" = type { i32*, %classT }
%classT = type { %classO, %classJ*, i8 }
%classO = type { i32 }
%classJ = type { i8 }
%"classA" = type { %"classB" }
%"classB" = type { i8 }
%"classC" = type { %"classD", %"structA" }
%"classD" = type { %"structA"* }

; Function Attrs: ssp uwtable
define %"structA"** @test(%"classA"* %this, i32** %p1) #0 align 2 {
entry:
; CHECK-LABEL: @test
; CHECK: load i32** %p1, align 8, !tbaa
; CHECK: load i32** inttoptr (i64 8 to i32**), align 8, !tbaa
; CHECK: call void @callee
  %p1.addr = alloca i32**, align 8
  store i32** %p1, i32*** %p1.addr, align 8, !tbaa !1
  %0 = load i32*** %p1.addr, align 8
  %1 = load i32** %0, align 8, !tbaa !4
  %__value_ = getelementptr inbounds %"classC"* null, i32 0, i32 1
  %__cc = getelementptr inbounds %"structA"* %__value_, i32 0, i32 0
  %first = getelementptr inbounds %"structB"* %__cc, i32 0, i32 0
  %2 = load i32** %first, align 8, !tbaa !6
  call void @callee(i32* %1, i32* %2)
  unreachable
}

declare void @callee(i32*, i32*) #1

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.4"}
!1 = metadata !{metadata !2, metadata !2, i64 0}
!2 = metadata !{metadata !"omnipotent char", metadata !3, i64 0}
!3 = metadata !{metadata !"Simple C/C++ TBAA"}
!4 = metadata !{metadata !5, metadata !5, i64 0}
!5 = metadata !{metadata !"any pointer", metadata !2, i64 0}
!6 = metadata !{metadata !7, metadata !5, i64 8}
!7 = metadata !{metadata !"_ZTSN12_GLOBAL__N_11RINS_1FIPi8TreeIterN1I1S1LENS_1KINS_1DIKS2_S3_EEEEE1GEPSD_EE", metadata !8, i64 8}
!8 = metadata !{metadata !"_ZTSN12_GLOBAL__N_11FIPi8TreeIterN1I1S1LENS_1KINS_1DIKS1_S2_EEEEE1GE", metadata !9, i64 0}
!9 = metadata !{metadata !"_ZTSN12_GLOBAL__N_11DIKPi8TreeIterEE", metadata !5, i64 0, metadata !10, i64 8}
!10 = metadata !{metadata !"_ZTS8TreeIter", metadata !5, i64 8, metadata !11, i64 16}
!11 = metadata !{metadata !"bool", metadata !2, i64 0}
