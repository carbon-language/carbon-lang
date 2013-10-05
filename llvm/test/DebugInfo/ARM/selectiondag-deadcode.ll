; RUN: llc -filetype=asm < %s | FileCheck %s
target triple = "thumbv7-apple-ios7.0.0"
%"matrix" = type { [9 x float] }
%"ellipse" = type { %"plane" }
%"plane" = type { %"vector" }
%"vector" = type { %union.anon.2.26.2 }
%union.anon.2.26.2 = type { %struct.anon.1.25.1 }
%struct.anon.1.25.1 = type { float, float, float }

define void @_Z7Ellipse9GetMatrixEv(%"matrix"* noalias nocapture sret %agg.result, %"ellipse"* nocapture readonly %this) #0 align 2 {
  br i1 undef, label %_Z7Vector39NormalizeEv.exit, label %1
  %2 = call double @_Z7StdMath4SqrtEd(double undef) #4
  br label %_Z7Vector39NormalizeEv.exit
_Z7Vector39NormalizeEv.exit:                 ; preds = %1, %0
  call void @llvm.dbg.declare(metadata !{%"matrix"* %agg.result}, metadata !74)
  ; rdar://problem/15094721.  When this dead use gets eliminated (and
  ; thus the def of the vreg holding %agg.result) the dbg_value becomes
  ; dangling and SelectionDAGISel crashes.
  ; It should definitely not crash. Drop the dbg_value instead.
  ; CHECK: "matrix"
  %3 = getelementptr inbounds %"matrix"* %agg.result, i32 0, i32 0, i32 8
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata) #1

declare double @_Z7StdMath4SqrtEd(double) #3

!llvm.dbg.cu = !{!0}
!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 true, metadata !"", i32 0, metadata !2, metadata !3, metadata !67, metadata !2, metadata !104, metadata !""} ; [ DW_TAG_compile_unit ] [/Volumes/Data/radar/15094721/Ellipse.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"Ellipse.cpp", metadata !"/Volumes/Data/radar/15094721"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !9, metadata !12, metadata !15, metadata !18, metadata !33, metadata !58}
!4 = metadata !{i32 786434, metadata !5, metadata !6, metadata !"Ellipse", i32 307, i64 96, i64 32, i32 0, i32 0, null, metadata !7, i32 0, null, null, metadata !"_ZTS7EllipseE"} ; [ DW_TAG_class_type ] [Ellipse] [line 307, size 96, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"test.ii", metadata !"/Volumes/Data/radar/15094721"}
!6 = metadata !{i32 786489, metadata !5, null, metadata !"Namespace", i32 158} ; [ DW_TAG_namespace ] [Namespace] [line 158]
!7 = metadata !{metadata !8, metadata !30}
!8 = metadata !{i32 786460, null, metadata !"_ZTS7EllipseE", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from Plane]
!9 = metadata !{i32 786434, metadata !5, metadata !6, metadata !"Plane", i32 302, i64 96, i64 32, i32 0, i32 0, null, metadata !10, i32 0, null, null, metadata !"_ZTS5PlaneE"} ; [ DW_TAG_class_type ] [Plane] [line 302, size 96, align 32, offset 0] [def] [from ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786445, metadata !5, metadata !"_ZTS5PlaneE", metadata !"normal", i32 305, i64 96, i64 32, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ] [normal] [line 305, size 96, align 32, offset 0] [from Vector3]
!12 = metadata !{i32 786434, metadata !5, metadata !6, metadata !"Vector3", i32 182, i64 96, i64 32, i32 0, i32 0, null, metadata !13, i32 0, null, null, metadata !"_ZTS7Vector3E"} ; [ DW_TAG_class_type ] [Vector3] [line 182, size 96, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !24}
!14 = metadata !{i32 786445, metadata !5, metadata !"_ZTS7Vector3E", metadata !"", i32 185, i64 96, i64 32, i64 0, i32 0, metadata !15} ; [ DW_TAG_member ] [line 185, size 96, align 32, offset 0] [from ]
!15 = metadata !{i32 786455, metadata !5, metadata !12, metadata !"", i32 185, i64 96, i64 32, i64 0, i32 0, null, metadata !16, i32 0, null, null, metadata !"_ZTS7Vector3Ut_E"} ; [ DW_TAG_union_type ] [line 185, size 96, align 32, offset 0] [def] [from ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786445, metadata !5, metadata !"_ZTS7Vector3Ut_E", metadata !"", i32 187, i64 96, i64 32, i64 0, i32 0, metadata !18} ; [ DW_TAG_member ] [line 187, size 96, align 32, offset 0] [from ]
!18 = metadata !{i32 786451, metadata !5, metadata !15, metadata !"", i32 187, i64 96, i64 32, i32 0, i32 0, null, metadata !19, i32 0, null, null, metadata !"_ZTS7Vector3Ut_Ut_E"} ; [ DW_TAG_structure_type ] [line 187, size 96, align 32, offset 0] [def] [from ]
!19 = metadata !{metadata !20, metadata !22, metadata !23}
!20 = metadata !{i32 786445, metadata !5, metadata !"_ZTS7Vector3Ut_Ut_E", metadata !"x", i32 189, i64 32, i64 32, i64 0, i32 0, metadata !21} ; [ DW_TAG_member ] [x] [line 189, size 32, align 32, offset 0] [from float]
!21 = metadata !{i32 786468, null, null, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!22 = metadata !{i32 786445, metadata !5, metadata !"_ZTS7Vector3Ut_Ut_E", metadata !"y", i32 190, i64 32, i64 32, i64 32, i32 0, metadata !21} ; [ DW_TAG_member ] [y] [line 190, size 32, align 32, offset 32] [from float]
!23 = metadata !{i32 786445, metadata !5, metadata !"_ZTS7Vector3Ut_Ut_E", metadata !"z", i32 191, i64 32, i64 32, i64 64, i32 0, metadata !21} ; [ DW_TAG_member ] [z] [line 191, size 32, align 32, offset 64] [from float]
!24 = metadata !{i32 786478, metadata !5, metadata !12, metadata !"Normalize", metadata !"Normalize", metadata !"_Z7Vector39NormalizeEv", i32 194, metadata !25, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 194} ; [ DW_TAG_subprogram ] [line 194] [Normalize]
!25 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !26, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = metadata !{metadata !27, metadata !28}
!27 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from Vector3]
!28 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 1088, metadata !"_ZTS7Vector3E"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [artificial] [from _ZTS7Vector3E]
!29 = metadata !{i32 786468}
!30 = metadata !{i32 786478, metadata !5, metadata !4, metadata !"GetMatrix", metadata !"GetMatrix", metadata !"_Z7Ellipse9GetMatrixEv", i32 309, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i32 257, i1 true, null, null, i32 0, metadata !29, i32 309} ; [ DW_TAG_subprogram ] [line 309] [private] [GetMatrix]
!31 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !32, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!32 = metadata !{metadata !33, metadata !57}
!33 = metadata !{i32 786434, metadata !5, metadata !6, metadata !"Matrix3", i32 202, i64 288, i64 32, i32 0, i32 0, null, metadata !34, i32 0, null, null, metadata !"_ZTS7Matrix3E"} ; [ DW_TAG_class_type ] [Matrix3] [line 202, size 288, align 32, offset 0] [def] [from ]
!34 = metadata !{metadata !35, metadata !39, metadata !43, metadata !46, metadata !51}
!35 = metadata !{i32 786445, metadata !5, metadata !"_ZTS7Matrix3E", metadata !"data", i32 211, i64 288, i64 32, i64 0, i32 0, metadata !36} ; [ DW_TAG_member ] [data] [line 211, size 288, align 32, offset 0] [from ]
!36 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 288, i64 32, i32 0, i32 0, metadata !21, metadata !37, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 288, align 32, offset 0] [from float]
!37 = metadata !{metadata !38}
!38 = metadata !{i32 786465, i64 0, i64 9}        ; [ DW_TAG_subrange_type ] [0, 8]
!39 = metadata !{i32 786478, metadata !5, metadata !33, metadata !"Matrix3", metadata !"Matrix3", metadata !"", i32 205, metadata !40, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 205} ; [ DW_TAG_subprogram ] [line 205] [Matrix3]
!40 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !41, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = metadata !{null, metadata !42}
!42 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 1088, metadata !"_ZTS7Matrix3E"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [artificial] [from _ZTS7Matrix3E]
!43 = metadata !{i32 786478, metadata !5, metadata !33, metadata !"Matrix3", metadata !"Matrix3", metadata !"", i32 206, metadata !44, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 206} ; [ DW_TAG_subprogram ] [line 206] [Matrix3]
!44 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !45, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{null, metadata !42, metadata !21, metadata !21, metadata !21, metadata !21, metadata !21, metadata !21, metadata !21, metadata !21, metadata !21}
!46 = metadata !{i32 786478, metadata !5, metadata !33, metadata !"Multiply", metadata !"Multiply", metadata !"_Z7Matrix38MultiplyEPKf", i32 209, metadata !47, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 209} ; [ DW_TAG_subprogram ] [line 209] [Multiply]
!47 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !48, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!48 = metadata !{null, metadata !42, metadata !49}
!49 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !50} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from ]
!50 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !21} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from float]
!51 = metadata !{i32 786478, metadata !5, metadata !33, metadata !"operator*=", metadata !"operator*=", metadata !"_Z7Matrix3mLERKS0_", i32 210, metadata !52, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 210} ; [ DW_TAG_subprogram ] [line 210] [operator*=]
!52 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !53, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!53 = metadata !{metadata !54, metadata !42, metadata !55}
!54 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !33} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from Matrix3]
!55 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !56} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!56 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !33} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from Matrix3]
!57 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 1088, metadata !"_ZTS7EllipseE"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [artificial] [from _ZTS7EllipseE]
!58 = metadata !{i32 786434, metadata !5, metadata !6, metadata !"StdMath", i32 159, i64 8, i64 8, i32 0, i32 0, null, metadata !59, i32 0, null, null, metadata !"_ZTS7StdMathE"} ; [ DW_TAG_class_type ] [StdMath] [line 159, size 8, align 8, offset 0] [def] [from ]
!59 = metadata !{metadata !60, metadata !63}
!60 = metadata !{i32 786478, metadata !5, metadata !58, metadata !"InvSqrt", metadata !"InvSqrt", metadata !"_Z7StdMath7InvSqrtEf", i32 162, metadata !61, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 162} ; [ DW_TAG_subprogram ] [line 162] [InvSqrt]
!61 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !62, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!62 = metadata !{metadata !21, metadata !21}
!63 = metadata !{i32 786478, metadata !5, metadata !58, metadata !"Sqrt", metadata !"Sqrt", metadata !"_Z7StdMath4SqrtEd", i32 163, metadata !64, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, metadata !29, i32 163} ; [ DW_TAG_subprogram ] [line 163] [Sqrt]
!64 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !65, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = metadata !{metadata !66, metadata !66}
!66 = metadata !{i32 786468, null, null, metadata !"double", i32 0, i64 64, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 32, offset 0, enc DW_ATE_float]
!67 = metadata !{metadata !68, metadata !76, metadata !81, metadata !89, metadata !92, metadata !95, metadata !101}
!68 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"GetMatrix", metadata !"GetMatrix", metadata !"_Z7Ellipse9GetMatrixEv", i32 313, metadata !31, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (%"matrix"*, %"ellipse"*)* @_Z7Ellipse9GetMatrixEv, null, metadata !30, metadata !69, i32 314} ; [ DW_TAG_subprogram ] [line 313] [def] [scope 314] [GetMatrix]
!69 = metadata !{metadata !70, metadata !72, metadata !74, metadata !75}
!70 = metadata !{i32 786689, metadata !68, metadata !"this", null, i32 16777216, metadata !71, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!71 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !"_ZTS7EllipseE"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from _ZTS7EllipseE]
!72 = metadata !{i32 786688, metadata !68, metadata !"vert", metadata !73, i32 315, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [vert] [line 315]
!73 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [/Volumes/Data/radar/15094721/test.ii]
!74 = metadata !{i32 786688, metadata !68, metadata !"matrix", metadata !73, i32 317, metadata !33, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [matrix] [line 317]
!75 = metadata !{i32 786688, metadata !68, metadata !"rotateZ", metadata !73, i32 318, metadata !33, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [rotateZ] [line 318]
!76 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"operator*=", metadata !"operator*=", metadata !"_Z7Matrix3mLERKS0_", i32 219, metadata !52, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !51, metadata !77, i32 220} ; [ DW_TAG_subprogram ] [line 219] [def] [scope 220] [operator*=]
!77 = metadata !{metadata !78, metadata !80}
!78 = metadata !{i32 786689, metadata !76, metadata !"this", null, i32 16777216, metadata !79, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!79 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !"_ZTS7Matrix3E"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from _ZTS7Matrix3E]
!80 = metadata !{i32 786689, metadata !76, metadata !"matrix", metadata !73, i32 33554651, metadata !55, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [matrix] [line 219]
!81 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"Multiply", metadata !"Multiply", metadata !"_Z7Matrix38MultiplyEPKf", i32 214, metadata !47, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !46, metadata !82, i32 215} ; [ DW_TAG_subprogram ] [line 214] [def] [scope 215] [Multiply]
!82 = metadata !{metadata !83, metadata !84, metadata !85}
!83 = metadata !{i32 786689, metadata !81, metadata !"this", null, i32 16777216, metadata !79, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!84 = metadata !{i32 786689, metadata !81, metadata !"matrix", metadata !73, i32 33554646, metadata !49, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [matrix] [line 214]
!85 = metadata !{i32 786688, metadata !81, metadata !"temp", metadata !73, i32 216, metadata !86, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [temp] [line 216]
!86 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 512, i64 32, i32 0, i32 0, metadata !21, metadata !87, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 512, align 32, offset 0] [from float]
!87 = metadata !{metadata !88}
!88 = metadata !{i32 786465, i64 0, i64 16}       ; [ DW_TAG_subrange_type ] [0, 15]
!89 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"Matrix3", metadata !"Matrix3", metadata !"_Z7Matrix3C1Ev", i32 213, metadata !40, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !39, metadata !90, i32 213} ; [ DW_TAG_subprogram ] [line 213] [def] [Matrix3]
!90 = metadata !{metadata !91}
!91 = metadata !{i32 786689, metadata !89, metadata !"this", null, i32 16777216, metadata !79, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!92 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"Matrix3", metadata !"Matrix3", metadata !"_Z7Matrix3C2Ev", i32 213, metadata !40, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !39, metadata !93, i32 213} ; [ DW_TAG_subprogram ] [line 213] [def] [Matrix3]
!93 = metadata !{metadata !94}
!94 = metadata !{i32 786689, metadata !92, metadata !"this", null, i32 16777216, metadata !79, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!95 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"Normalize", metadata !"Normalize", metadata !"_Z7Vector39NormalizeEv", i32 196, metadata !25, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !24, metadata !96, i32 197} ; [ DW_TAG_subprogram ] [line 196] [def] [scope 197] [Normalize]
!96 = metadata !{metadata !97, metadata !99, metadata !100}
!97 = metadata !{i32 786689, metadata !95, metadata !"this", null, i32 16777216, metadata !98, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!98 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !"_ZTS7Vector3E"} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from _ZTS7Vector3E]
!99 = metadata !{i32 786688, metadata !95, metadata !"len2", metadata !73, i32 198, metadata !21, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [len2] [line 198]
!100 = metadata !{i32 786688, metadata !95, metadata !"invLen", metadata !73, i32 199, metadata !21, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [invLen] [line 199]
!101 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"InvSqrt", metadata !"InvSqrt", metadata !"_Z7StdMath7InvSqrtEf", i32 165, metadata !61, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, null, null, metadata !60, metadata !102, i32 166} ; [ DW_TAG_subprogram ] [line 165] [def] [scope 166] [InvSqrt]
!102 = metadata !{metadata !103}
!103 = metadata !{i32 786689, metadata !101, metadata !"a_val", metadata !73, i32 16777381, metadata !21, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a_val] [line 165]
!104 = metadata !{metadata !105}
!105 = metadata !{i32 786490, metadata !0, metadata !6, i32 312} ; [ DW_TAG_imported_module ]
