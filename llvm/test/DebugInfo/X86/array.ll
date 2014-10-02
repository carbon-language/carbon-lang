; ModuleID = 'array.c'
;
; From (clang -g -c -O1):
;
; void f(int* p) {
;   p[0] = 42;
; }
;
; int main(int argc, char** argv) {
;   int array[4] = { 0, 1, 2, 3 };
;   f(array);
;   return array[0];
; }
;
; RUN: llc -filetype=asm %s -o - | FileCheck %s
; Test that we only emit register-indirect locations for the array array.
; rdar://problem/14874886
;
; CHECK:     ##DEBUG_VALUE: main:array <- [R{{.*}}+0]
; CHECK-NOT: ##DEBUG_VALUE: main:array <- R{{.*}}
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@main.array = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16

; Function Attrs: nounwind ssp uwtable
define void @f(i32* nocapture %p) #0 {
  tail call void @llvm.dbg.value(metadata !{i32* %p}, i64 0, metadata !11, metadata !{metadata !"0x102"}), !dbg !28
  store i32 42, i32* %p, align 4, !dbg !29, !tbaa !30
  ret void, !dbg !34
}

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) #0 {
  %array = alloca [4 x i32], align 16
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !19, metadata !{metadata !"0x102"}), !dbg !35
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !20, metadata !{metadata !"0x102"}), !dbg !35
  tail call void @llvm.dbg.value(metadata !{[4 x i32]* %array}, i64 0, metadata !21, metadata !{metadata !"0x102"}), !dbg !36
  %1 = bitcast [4 x i32]* %array to i8*, !dbg !36
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ([4 x i32]* @main.array to i8*), i64 16, i32 16, i1 false), !dbg !36
  tail call void @llvm.dbg.value(metadata !{[4 x i32]* %array}, i64 0, metadata !21, metadata !{metadata !"0x102"}), !dbg !36
  %2 = getelementptr inbounds [4 x i32]* %array, i64 0, i64 0, !dbg !37
  call void @f(i32* %2), !dbg !37
  tail call void @llvm.dbg.value(metadata !{[4 x i32]* %array}, i64 0, metadata !21, metadata !{metadata !"0x102"}), !dbg !36
  %3 = load i32* %2, align 16, !dbg !38, !tbaa !30
  ret i32 %3, !dbg !38
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26}
!llvm.ident = !{!27}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 \001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/array.c] [DW_LANG_C99]
!1 = metadata !{metadata !"array.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !12}
!4 = metadata !{metadata !"0x2e\00f\00f\00\001\000\001\000\006\00256\001\001", metadata !1, metadata !5, metadata !6, null, void (i32*)* @f, null, null, metadata !10} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/array.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"0x101\00p\0016777217\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [p] [line 1]
!12 = metadata !{metadata !"0x2e\00main\00main\00\005\000\001\000\006\00256\001\005", metadata !1, metadata !5, metadata !13, null, i32 (i32, i8**)* @main, null, null, metadata !18} ; [ DW_TAG_subprogram ] [line 5] [def] [main]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !9, metadata !9, metadata !15}
!15 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!16 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !17} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!17 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!18 = metadata !{metadata !19, metadata !20, metadata !21}
!19 = metadata !{metadata !"0x101\00argc\0016777221\000", metadata !12, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [argc] [line 5]
!20 = metadata !{metadata !"0x101\00argv\0033554437\000", metadata !12, metadata !5, metadata !15} ; [ DW_TAG_arg_variable ] [argv] [line 5]
!21 = metadata !{metadata !"0x100\00array\006\000", metadata !12, metadata !5, metadata !22} ; [ DW_TAG_auto_variable ] [array] [line 6]
!22 = metadata !{metadata !"0x1\00\000\00128\0032\000\000", null, null, metadata !9, metadata !23, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 32, offset 0] [from int]
!23 = metadata !{metadata !24}
!24 = metadata !{metadata !"0x21\000\004"}        ; [ DW_TAG_subrange_type ] [0, 3]
!25 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!27 = metadata !{metadata !"clang version 3.5.0 "}
!28 = metadata !{i32 1, i32 0, metadata !4, null}
!29 = metadata !{i32 2, i32 0, metadata !4, null}
!30 = metadata !{metadata !31, metadata !31, i64 0}
!31 = metadata !{metadata !"int", metadata !32, i64 0}
!32 = metadata !{metadata !"omnipotent char", metadata !33, i64 0}
!33 = metadata !{metadata !"Simple C/C++ TBAA"}
!34 = metadata !{i32 3, i32 0, metadata !4, null}
!35 = metadata !{i32 5, i32 0, metadata !12, null}
!36 = metadata !{i32 6, i32 0, metadata !12, null}
!37 = metadata !{i32 7, i32 0, metadata !12, null}
!38 = metadata !{i32 8, i32 0, metadata !12, null}
