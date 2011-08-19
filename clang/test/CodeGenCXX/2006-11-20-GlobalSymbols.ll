; ModuleID = '/Volumes/Data/sources/llvm/tools/clang/test/CodeGenCXX/2006-11-20-GlobalSymbols.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.1.0"

@"\01f\01oo" = global i32 0, align 4

define i32 @_Z3barv() nounwind {
entry:
  %tmp = load i32* @"\01f\01oo", align 4, !dbg !13
  ret i32 %tmp, !dbg !13
}

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 720913, i32 0, i32 4, metadata !"/Volumes/Data/sources/llvm/tools/clang/test/CodeGenCXX/<unknown>", metadata !"/Volumes/Data/builds/build-llvm/tools/clang/test/CodeGenCXX", metadata !"clang version 3.0 (trunk 138139)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !10} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 720942, i32 0, metadata !6, metadata !"bar", metadata !"bar", metadata !"_Z3barv", metadata !6, i32 8, metadata !7, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 ()* @_Z3barv, null, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 720937, metadata !"/Volumes/Data/sources/llvm/tools/clang/test/CodeGenCXX/2006-11-20-GlobalSymbols.cpp", metadata !"/Volumes/Data/builds/build-llvm/tools/clang/test/CodeGenCXX", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 720917, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 720932, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !12}
!12 = metadata !{i32 720948, i32 0, null, metadata !"foo", metadata !"foo", metadata !"\01f\01oo", metadata !6, i32 6, metadata !9, i32 0, i32 1, i32* @"\01f\01oo"} ; [ DW_TAG_variable ]
!13 = metadata !{i32 9, i32 3, metadata !14, null}
!14 = metadata !{i32 720907, metadata !5, i32 8, i32 11, metadata !6, i32 0} ; [ DW_TAG_lexical_block ]
