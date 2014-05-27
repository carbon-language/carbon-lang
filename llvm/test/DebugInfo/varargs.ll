; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; REQUIRES: object-emission
;
; Test debug info for variadic function arguments.
; Created from tools/clang/tests/CodeGenCXX/debug-info-varargs.cpp
;
; The ... parameter of variadic should be emitted as
; DW_TAG_unspecified_parameters.
;
; Normal variadic function.
; void b(int c, ...);
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "a"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "b"
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; Variadic C++ member function.
; struct A { void a(int c, ...); }
;
; Variadic function pointer.
; void (*fptr)(int, ...);
;
; CHECK: DW_TAG_subroutine_type
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_unspecified_parameters
;
; ModuleID = 'llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp'

%struct.A = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_Z1biz(i32 %c, ...) #0 {
  %1 = alloca i32, align 4
  %a = alloca %struct.A, align 1
  %fptr = alloca void (i32, ...)*, align 8
  store i32 %c, i32* %1, align 4
  call void @llvm.dbg.declare(metadata !{i32* %1}, metadata !21), !dbg !22
  call void @llvm.dbg.declare(metadata !{%struct.A* %a}, metadata !23), !dbg !24
  call void @llvm.dbg.declare(metadata !{void (i32, ...)** %fptr}, metadata !25), !dbg !27
  store void (i32, ...)* @_Z1biz, void (i32, ...)** %fptr, align 8, !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !13, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp", metadata !"radar/13690847"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"A", i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"a", metadata !"a", metadata !"_ZN1A1aEiz", i32 6, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !12, i32 6} ; [ DW_TAG_subprogram ] [line 6] [a]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !10, metadata !11}
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 786456}
!12 = metadata !{i32 786468}
!13 = metadata !{metadata !14}
!14 = metadata !{i32 786478, metadata !1, metadata !15, metadata !"b", metadata !"b", metadata !"_Z1biz", i32 13, metadata !16, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32, ...)* @_Z1biz, null, null, metadata !2, i32 13} ; [ DW_TAG_subprogram ] [line 13] [def] [b]
!15 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [llvm/tools/clang/test/CodeGenCXX/debug-info-varargs.cpp]
!16 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null, metadata !10, metadata !11}
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!20 = metadata !{metadata !"clang version 3.5 "}
!21 = metadata !{i32 786689, metadata !14, metadata !"c", metadata !15, i32 16777229, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [c] [line 13]
!22 = metadata !{i32 13, i32 0, metadata !14, null}
!23 = metadata !{i32 786688, metadata !14, metadata !"a", metadata !15, i32 16, metadata !4, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 16]
!24 = metadata !{i32 16, i32 0, metadata !14, null}
!25 = metadata !{i32 786688, metadata !14, metadata !"fptr", metadata !15, i32 18, metadata !26, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [fptr] [line 18]
!26 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!27 = metadata !{i32 18, i32 0, metadata !14, null}
!28 = metadata !{i32 22, i32 0, metadata !14, null}
