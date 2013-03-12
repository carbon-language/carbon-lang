; RUN: opt -std-compile-opts < %s | llvm-dis | not grep badref 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

%struct.anon = type { i32, i32 }
%struct.test = type { i64, %struct.anon, %struct.test* }

@TestArrayPtr = global %struct.test* getelementptr inbounds ([10 x %struct.test]* @TestArray, i64 0, i64 3) ; <%struct.test**> [#uses=1]
@TestArray = common global [10 x %struct.test] zeroinitializer, align 32 ; <[10 x %struct.test]*> [#uses=2]

define i32 @main() nounwind readonly {
  %diff1 = alloca i64                             ; <i64*> [#uses=2]
  call void @llvm.dbg.declare(metadata !{i64* %diff1}, metadata !0)
  store i64 72, i64* %diff1, align 8
  %v1 = load %struct.test** @TestArrayPtr, align 8 ; <%struct.test*> [#uses=1]
  %v2 = ptrtoint %struct.test* %v1 to i64 ; <i64> [#uses=1]
  %v3 = sub i64 %v2, ptrtoint ([10 x %struct.test]* @TestArray to i64) ; <i64> [#uses=1]
  store i64 %v3, i64* %diff1, align 8
  ret i32 4
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!7 = metadata !{metadata !1}
!6 = metadata !{i32 786449, i32 0, i32 12, metadata !"/d/j/debug-test.c", metadata !"/Volumes/Data/b", metadata !"clang version 3.0 (trunk 131941)", i1 true, i1 false, metadata !"", i32 0, null, null, metadata !7, null, null} ; [ DW_TAG_compile_unit ]
!0 = metadata !{i32 786688, metadata !1, metadata !"c", metadata !2, i32 2, metadata !5, i32 0, null} ; [ DW_TAG_auto_variable ]
!1 = metadata !{i32 786478, i32 0, metadata !2, metadata !"main", metadata !"main", metadata !"", metadata !2, i32 1, metadata !3, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 false, i32 ()* @main, null, null, null, i32 1} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 786473, metadata !"/d/j/debug-test.c", metadata !"/Volumes/Data/b", metadata !0} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 786453, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !4, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786468, metadata !6, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
