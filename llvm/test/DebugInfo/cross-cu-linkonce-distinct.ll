; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Testing that two distinct (distinct by writing them in separate files, while
; still fulfilling C++'s ODR by having identical token sequences) functions,
; linked under LTO, get plausible debug info (and don't crash).

; Built from source:
; $ clang++ a.cpp b.cpp -g -c -emit-llvm
; $ llvm-link a.bc b.bc -o ab.bc

; This change is intended to tickle a case where the subprogram MDNode
; associated with the llvm::Function will differ from the subprogram
; referenced by the DbgLocs in the function.

; $ sed -ie "s/!12, !0/!0, !12/" ab.ll
; $ cat a.cpp
; inline int func(int i) {
;   return i * 2;
; }
; int (*x)(int) = &func;
; $ cat b.cpp
; inline int func(int i) {
;   return i * 2;
; }
; int (*y)(int) = &func;

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "func"
; CHECK: DW_TAG_compile_unit

; FIXME: Maybe we should drop the subprogram here - since the function was
; emitted in one CU, due to linkonce_odr uniquing. We certainly don't emit the
; subprogram here if the source location for this definition is the same (see
; test/DebugInfo/cross-cu-linkonce.ll), though it's very easy to tickle that
; into failing even without duplicating the source as has been done in this
; case (two cpp files in different directories, including the same header that
; contains an inline function - clang will produce distinct subprogram metadata
; that won't deduplicate owing to the file location information containing the
; directory of the source file even though the file name is absolute, not
; relative)

; CHECK: DW_TAG_subprogram

@x = global i32 (i32)* @_Z4funci, align 8
@y = global i32 (i32)* @_Z4funci, align 8

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr i32 @_Z4funci(i32 %i) #0 {
  %1 = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  call void @llvm.dbg.declare(metadata !{i32* %1}, metadata !22), !dbg !23
  %2 = load i32* %1, align 4, !dbg !24
  %3 = mul nsw i32 %2, 2, !dbg !24
  ret i32 %3, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!12, !0}
!llvm.module.flags = !{!19, !20}
!llvm.ident = !{!21, !21}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !9, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/a.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"a.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func", metadata !"func", metadata !"_Z4funci", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z4funci, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [func]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/a.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786484, i32 0, null, metadata !"x", metadata !"x", metadata !"", metadata !5, i32 4, metadata !11, i32 0, i32 1, i32 (i32)** @x, null} ; [ DW_TAG_variable ] [x] [line 4] [def]
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!12 = metadata !{i32 786449, metadata !13, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !14, metadata !17, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/b.cpp] [DW_LANG_C_plus_plus]
!13 = metadata !{metadata !"b.cpp", metadata !"/tmp/dbginfo"}
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786478, metadata !13, metadata !16, metadata !"func", metadata !"func", metadata !"_Z4funci", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z4funci, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [func]
!16 = metadata !{i32 786473, metadata !13}        ; [ DW_TAG_file_type ] [/tmp/dbginfo/b.cpp]
!17 = metadata !{metadata !18}
!18 = metadata !{i32 786484, i32 0, null, metadata !"y", metadata !"y", metadata !"", metadata !16, i32 4, metadata !11, i32 0, i32 1, i32 (i32)** @y, null} ; [ DW_TAG_variable ] [y] [line 4] [def]
!19 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!21 = metadata !{metadata !"clang version 3.5.0 "}
!22 = metadata !{i32 786689, metadata !4, metadata !"i", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [i] [line 1]
!23 = metadata !{i32 1, i32 0, metadata !4, null}
!24 = metadata !{i32 2, i32 0, metadata !4, null}
