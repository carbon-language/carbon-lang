; ModuleID = 'vector.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define <4 x float> @_Z3fooDv2_fS_(double %a.coerce, double %b.coerce) #0 {
entry:
  %a = alloca <2 x float>, align 8
  %a.addr = alloca <2 x float>, align 8
  %b = alloca <2 x float>, align 8
  %b.addr = alloca <2 x float>, align 8
  %c = alloca <4 x float>, align 16
  %0 = bitcast <2 x float>* %a to double*
  store double %a.coerce, double* %0, align 1
  %a1 = load <2 x float>* %a, align 8
  store <2 x float> %a1, <2 x float>* %a.addr, align 8
  call void @llvm.dbg.declare(metadata !{<2 x float>* %a.addr}, metadata !21), !dbg !22
  %1 = bitcast <2 x float>* %b to double*
  store double %b.coerce, double* %1, align 1
  %b2 = load <2 x float>* %b, align 8
  store <2 x float> %b2, <2 x float>* %b.addr, align 8
  call void @llvm.dbg.declare(metadata !{<2 x float>* %b.addr}, metadata !23), !dbg !22
  call void @llvm.dbg.declare(metadata !{<4 x float>* %c}, metadata !24), !dbg !25
  %2 = load <2 x float>* %a.addr, align 8, !dbg !26
  %3 = load <4 x float>* %c, align 16, !dbg !26
  %4 = shufflevector <2 x float> %2, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>, !dbg !26
  %5 = shufflevector <4 x float> %3, <4 x float> %4, <4 x i32> <i32 4, i32 1, i32 5, i32 3>, !dbg !26
  store <4 x float> %5, <4 x float>* %c, align 16, !dbg !26
  %6 = load <2 x float>* %b.addr, align 8, !dbg !27
  %7 = load <4 x float>* %c, align 16, !dbg !27
  %8 = shufflevector <2 x float> %6, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>, !dbg !27
  %9 = shufflevector <4 x float> %7, <4 x float> %8, <4 x i32> <i32 0, i32 4, i32 2, i32 5>, !dbg !27
  store <4 x float> %9, <4 x float>* %c, align 16, !dbg !27
  %10 = load <4 x float>* %c, align 16, !dbg !28
  ret <4 x float> %10, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %a = alloca <2 x float>, align 8
  %b = alloca <2 x float>, align 8
  %x = alloca <4 x float>, align 16
  %coerce = alloca <2 x float>, align 8
  %coerce1 = alloca <2 x float>, align 8
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{<2 x float>* %a}, metadata !29), !dbg !30
  store <2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float>* %a, align 8, !dbg !30
  call void @llvm.dbg.declare(metadata !{<2 x float>* %b}, metadata !31), !dbg !32
  store <2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float>* %b, align 8, !dbg !32
  call void @llvm.dbg.declare(metadata !{<4 x float>* %x}, metadata !33), !dbg !34
  %0 = load <2 x float>* %a, align 8, !dbg !34
  %1 = load <2 x float>* %b, align 8, !dbg !34
  store <2 x float> %0, <2 x float>* %coerce, align 8, !dbg !34
  %2 = bitcast <2 x float>* %coerce to double*, !dbg !34
  %3 = load double* %2, align 1, !dbg !34
  store <2 x float> %1, <2 x float>* %coerce1, align 8, !dbg !34
  %4 = bitcast <2 x float>* %coerce1 to double*, !dbg !34
  %5 = load double* %4, align 1, !dbg !34
  %call = call <4 x float> @_Z3fooDv2_fS_(double %3, double %5), !dbg !34
  store <4 x float> %call, <4 x float>* %x, align 16, !dbg !34
  %6 = load <4 x float>* %x, align 16, !dbg !35
  %7 = extractelement <4 x float> %6, i32 0, !dbg !35
  %8 = load <4 x float>* %x, align 16, !dbg !35
  %9 = extractelement <4 x float> %8, i32 1, !dbg !35
  %add = fadd float %7, %9, !dbg !35
  %10 = load <4 x float>* %x, align 16, !dbg !35
  %11 = extractelement <4 x float> %10, i32 2, !dbg !35
  %add2 = fadd float %add, %11, !dbg !35
  %12 = load <4 x float>* %x, align 16, !dbg !35
  %13 = extractelement <4 x float> %12, i32 3, !dbg !35
  %add3 = fadd float %add2, %13, !dbg !35
  %conv = fptosi float %add3 to i32, !dbg !35
  ret i32 %conv, !dbg !35
}

attributes #0 = { noinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/vector.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"vector.cpp", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !17}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"_Z3fooDv2_fS_", i32 6, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, <4 x float> (double, double)* @_Z3fooDv2_fS_, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/vector.cpp]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !13, metadata !13}
!8 = metadata !{i32 786454, metadata !1, null, metadata !"float4", i32 2, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_typedef ] [float4] [line 2, size 0, align 0, offset 0] [from ]
!9 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 128, i64 128, i32 0, i32 2048, metadata !10, metadata !11, i32 0, i32 0} ; [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [vector] [from float]
!10 = metadata !{i32 786468, null, null, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786465, i64 0, i64 4}        ; [ DW_TAG_subrange_type ] [0, 3]
!13 = metadata !{i32 786454, metadata !1, null, metadata !"float2", i32 3, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_typedef ] [float2] [line 3, size 0, align 0, offset 0] [from ]
!14 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 64, i64 64, i32 0, i32 2048, metadata !10, metadata !15, i32 0, i32 0} ; [ DW_TAG_array_type ] [line 0, size 64, align 64, offset 0] [vector] [from float]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786465, i64 0, i64 2}        ; [ DW_TAG_subrange_type ] [0, 1]
!17 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 13, metadata !18, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 13} ; [ DW_TAG_subprogram ] [line 13] [def] [main]
!18 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !19, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!21 = metadata !{i32 786689, metadata !4, metadata !"a", metadata !5, i32 16777222, metadata !13, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 6]
!22 = metadata !{i32 6, i32 0, metadata !4, null}
!23 = metadata !{i32 786689, metadata !4, metadata !"b", metadata !5, i32 33554438, metadata !13, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 6]
!24 = metadata !{i32 786688, metadata !4, metadata !"c", metadata !5, i32 7, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [c] [line 7]
!25 = metadata !{i32 7, i32 0, metadata !4, null}
!26 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
!27 = metadata !{i32 9, i32 0, metadata !4, null}
!28 = metadata !{i32 10, i32 0, metadata !4, null}
!29 = metadata !{i32 786688, metadata !17, metadata !"a", metadata !5, i32 14, metadata !13, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 14]
!30 = metadata !{i32 14, i32 0, metadata !17, null}
!31 = metadata !{i32 786688, metadata !17, metadata !"b", metadata !5, i32 15, metadata !13, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 15]
!32 = metadata !{i32 15, i32 0, metadata !17, null}
!33 = metadata !{i32 786688, metadata !17, metadata !"x", metadata !5, i32 16, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [x] [line 16]
!34 = metadata !{i32 16, i32 0, metadata !17, null}
!35 = metadata !{i32 18, i32 0, metadata !17, null}
; RUN: opt < %s -debug-ir -S | FileCheck %s.check
