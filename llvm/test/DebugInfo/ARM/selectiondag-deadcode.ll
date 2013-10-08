; RUN: llc -filetype=asm < %s | FileCheck %s
target triple = "thumbv7-apple-ios7.0.0"
%class.Matrix3.0.6.10 = type { [9 x float] }
define arm_aapcscc void @_Z9GetMatrixv(%class.Matrix3.0.6.10* noalias nocapture sret %agg.result) #0 {
  br i1 fcmp oeq (float fadd (float fadd (float fmul (float undef, float undef), float fmul (float undef, float undef)), float fmul (float undef, float undef)), float 0.000000e+00), label %_ZN7Vector39NormalizeEv.exit, label %1
  tail call arm_aapcscc void @_ZL4Sqrtd() #3
  br label %_ZN7Vector39NormalizeEv.exit
_ZN7Vector39NormalizeEv.exit:                     ; preds = %1, %0
  ; rdar://problem/15094721.
  ;
  ; When this (partially) dead use gets eliminated (and thus the def
  ; of the vreg holding %agg.result) the dbg_value becomes dangling
  ; and SelectionDAGISel crashes.  It should definitely not
  ; crash. Drop the dbg_value instead.
  ; CHECK-NOT: "matrix"
  tail call void @llvm.dbg.declare(metadata !{%class.Matrix3.0.6.10* %agg.result}, metadata !45)
  %2 = getelementptr inbounds %class.Matrix3.0.6.10* %agg.result, i32 0, i32 0, i32 8
  ret void
}
declare void @llvm.dbg.declare(metadata, metadata) #1
declare arm_aapcscc void @_ZL4Sqrtd() #2
!4 = metadata !{i32 786434, metadata !5, null, metadata !"Matrix3", i32 20, i64 288, i64 32, i32 0, i32 0, null, null, i32 0, null, null, metadata !"_ZTS7Matrix3"} ; [ DW_TAG_class_type ] [Matrix3] [line 20, size 288, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"test.ii", metadata !"/Volumes/Data/radar/15094721"}
!39 = metadata !{i32 786478, metadata !5, metadata !40, metadata !"GetMatrix", metadata !"GetMatrix", metadata !"_Z9GetMatrixv", i32 32, metadata !41, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (%class.Matrix3.0.6.10*)* @_Z9GetMatrixv, null, null, null, i32 32} ; [ DW_TAG_subprogram ] [line 32] [def] [GetMatrix]
!40 = metadata !{i32 786473, metadata !5}         ; [ DW_TAG_file_type ] [/Volumes/Data/radar/15094721/test.ii]
!41 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, null, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{i32 786688, metadata !39, metadata !"matrix", metadata !40, i32 35, metadata !4, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [matrix] [line 35]
