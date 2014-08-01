; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      long int b;
;    } Inner;
;
;    typedef struct {
;      Inner inner[2];
;    } Outer;
;
;    int foo(Outer outer) {
;      Inner i1 = outer.inner[1];
;      return i1.a;
;    }
;
;
; CHECK: DW_TAG_variable [4]
;                                                  rax, piece 0x00000004
; CHECK-NEXT: DW_AT_location [DW_FORM_block1]{{.*}}50 93 04
; CHECK-NEXT:  DW_AT_name {{.*}}"i1"
;
; ModuleID = '/Volumes/Data/llvm/test/DebugInfo/X86/sroasplit-1.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i64 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(%struct.Outer* byval align 8 %outer) #0 {
entry:
  call void @llvm.dbg.declare(metadata !{%struct.Outer* %outer}, metadata !25), !dbg !26
  %i1.sroa.0.0..sroa_idx = getelementptr inbounds %struct.Outer* %outer, i64 0, i32 0, i64 1, i32 0, !dbg !27
  %i1.sroa.0.0.copyload = load i32* %i1.sroa.0.0..sroa_idx, align 8, !dbg !27
  call void @llvm.dbg.value(metadata !{i32 %i1.sroa.0.0.copyload}, i64 0, metadata !28), !dbg !27
  %i1.sroa.2.0..sroa_raw_cast = bitcast %struct.Outer* %outer to i8*, !dbg !27
  %i1.sroa.2.0..sroa_raw_idx = getelementptr inbounds i8* %i1.sroa.2.0..sroa_raw_cast, i64 20, !dbg !27
  ret i32 %i1.sroa.0.0.copyload, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/sroasplit-1.c] [DW_LANG_C99]
!1 = metadata !{metadata !"sroasplit-1.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 10, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%struct.Outer*)* @foo, null, null, metadata !2, i32 10} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/sroasplit-1.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786454, metadata !1, null, metadata !"Outer", i32 8, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ] [Outer] [line 8, size 0, align 0, offset 0] [from ]
!10 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 6, i64 256, i64 64, i32 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 6, size 256, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"inner", i32 7, i64 256, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ] [inner] [line 7, size 256, align 64, offset 0] [from ]
!13 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 256, i64 64, i32 0, i32 0, metadata !14, metadata !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 256, align 64, offset 0] [from Inner]
!14 = metadata !{i32 786454, metadata !1, null, metadata !"Inner", i32 4, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_typedef ] [Inner] [line 4, size 0, align 0, offset 0] [from ]
!15 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 1, i64 128, i64 64, i32 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 128, align 64, offset 0] [def] [from ]
!16 = metadata !{metadata !17, metadata !18}
!17 = metadata !{i32 786445, metadata !1, metadata !15, metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!18 = metadata !{i32 786445, metadata !1, metadata !15, metadata !"b", i32 3, i64 64, i64 64, i64 64, i32 0, metadata !19} ; [ DW_TAG_member ] [b] [line 3, size 64, align 64, offset 64] [from long int]
!19 = metadata !{i32 786468, null, null, metadata !"long int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!20 = metadata !{metadata !21}
!21 = metadata !{i32 786465, i64 0, i64 2}        ; [ DW_TAG_subrange_type ] [0, 1]
!22 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!23 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!24 = metadata !{metadata !"clang version 3.5.0 "}
!25 = metadata !{i32 786689, metadata !4, metadata !"outer", metadata !5, i32 16777226, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [outer] [line 10]
!26 = metadata !{i32 10, i32 0, metadata !4, null}
!27 = metadata !{i32 11, i32 0, metadata !4, null}
!28 = metadata !{i32 786688, metadata !4, metadata !"i1", metadata !5, i32 11, metadata !14, i32 0, i32 0, metadata !29} ; [ DW_TAG_auto_variable ] [i1] [line 11] [piece, size 4, offset 0]
!29 = metadata !{i32 3, i32 0, i32 4}
!30 = metadata !{i32 786688, metadata !4, metadata !"i1", metadata !5, i32 11, metadata !14, i32 0, i32 0, metadata !31} ; [ DW_TAG_auto_variable ] [i1] [line 11] [piece, size 12, offset 0]
!31 = metadata !{i32 3, i32 0, i32 12}
!32 = metadata !{i32 12, i32 0, metadata !4, null}
