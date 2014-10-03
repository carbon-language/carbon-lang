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
  tail call void @llvm.dbg.declare(metadata !{%class.Matrix3.0.6.10* %agg.result}, metadata !45, metadata !{metadata !"0x102"})
  %2 = getelementptr inbounds %class.Matrix3.0.6.10* %agg.result, i32 0, i32 0, i32 8
  ret void
}
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare arm_aapcscc void @_ZL4Sqrtd() #2
!4 = metadata !{metadata !"0x2\00Matrix3\0020\00288\0032\000\000\000", metadata !5, null, null, null, null, null, metadata !"_ZTS7Matrix3"} ; [ DW_TAG_class_type ] [Matrix3] [line 20, size 288, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"test.ii", metadata !"/Volumes/Data/radar/15094721"}
!39 = metadata !{metadata !"0x2e\00GetMatrix\00GetMatrix\00_Z9GetMatrixv\0032\000\001\000\006\00256\001\0032", metadata !5, metadata !40, metadata !41, null, void (%class.Matrix3.0.6.10*)* @_Z9GetMatrixv, null, null, null} ; [ DW_TAG_subprogram ] [line 32] [def] [GetMatrix]
!40 = metadata !{metadata !"0x29", metadata !5}         ; [ DW_TAG_file_type ] [/Volumes/Data/radar/15094721/test.ii]
!41 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, null, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{metadata !"0x100\00matrix\0035\008192", metadata !39, metadata !40, metadata !4} ; [ DW_TAG_auto_variable ] [matrix] [line 35]
