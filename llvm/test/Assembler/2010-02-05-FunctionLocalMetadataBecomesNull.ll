; RUN: opt -S -O3 < %s | FileCheck %s
; RUN: verify-uselistorder %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

%struct.anon = type { i32, i32 }
%struct.test = type { i64, %struct.anon, %struct.test* }

@TestArrayPtr = global %struct.test* getelementptr inbounds ([10 x %struct.test]* @TestArray, i64 0, i64 3) ; <%struct.test**> [#uses=1]
@TestArray = common global [10 x %struct.test] zeroinitializer, align 32 ; <[10 x %struct.test]*> [#uses=2]

define i32 @main() nounwind readonly {
  %diff1 = alloca i64                             ; <i64*> [#uses=2]
; CHECK: call void @llvm.dbg.value(metadata i64 72,
  call void @llvm.dbg.declare(metadata i64* %diff1, metadata !0, metadata !{!"0x102"})
  store i64 72, i64* %diff1, align 8
  %v1 = load %struct.test*, %struct.test** @TestArrayPtr, align 8 ; <%struct.test*> [#uses=1]
  %v2 = ptrtoint %struct.test* %v1 to i64 ; <i64> [#uses=1]
  %v3 = sub i64 %v2, ptrtoint ([10 x %struct.test]* @TestArray to i64) ; <i64> [#uses=1]
  store i64 %v3, i64* %diff1, align 8
  ret i32 4
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!7 = !{!1}
!6 = !{!"0x11\0012\00clang version 3.0 (trunk 131941)\001\00\000\00\000", !8, !9, !9, !7, null, null} ; [ DW_TAG_compile_unit ]
!0 = !{!"0x100\00c\002\000", !1, !2, !5} ; [ DW_TAG_auto_variable ]
!1 = !{!"0x2e\00main\00main\00\001\000\001\000\006\00256\000\001", !8, !2, !3, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ]
!2 = !{!"0x29", !8} ; [ DW_TAG_file_type ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !8, !2, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !6} ; [ DW_TAG_base_type ]
!8 = !{!"/d/j/debug-test.c", !"/Volumes/Data/b"}
!9 = !{i32 0}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"Debug Info Version", i32 2}
