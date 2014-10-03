; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=line - | FileCheck %s
; RUN: %llc_dwarf < %s -filetype=asm | FileCheck --check-prefix=ASM %s

; If multiple line tables are emitted, one per CU, those line tables can
; unambiguously rely on the comp_dir of their owning CU and use directory '0'
; to refer to it.

; CHECK: .debug_line contents:
; CHECK-NEXT: Line table prologue:
; CHECK-NOT: include_directories
; CHECK: file_names[   1]   0 {{.*}} a.cpp
; CHECK-NOT: file_names

; CHECK: Line table prologue:
; CHECK-NOT: include_directories
; CHECK: file_names[   1]   0 {{.*}} b.cpp
; CHECK-NOT: file_names

; However, if a single line table is emitted and shared between CUs, the
; comp_dir is ambiguous and relying on it would lead to different path
; interpretations depending on which CU lead to the table - so ensure that
; full paths are always emitted in this case, never comp_dir relative.

; ASM: .file   1 "/tmp/dbginfo/a{{[/\\]+}}a.cpp"
; ASM: .file   2 "/tmp/dbginfo/b{{[/\\]+}}b.cpp"

; Generated from the following source compiled to bitcode from within their
; respective directories (with debug info) and linked together with llvm-link

; a/a.cpp
; void func() {
; }

; b/b.cpp
; void func();
; int main() {
;   func();
; }

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  ret void, !dbg !19
}

; Function Attrs: uwtable
define i32 @main() #1 {
entry:
  call void @_Z4funcv(), !dbg !20
  ret i32 0, !dbg !21
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0, !8}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18, !18}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"a.cpp", metadata !"/tmp/dbginfo/a"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00func\00func\00_Z4funcv\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, void ()* @_Z4funcv, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [func]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/a/a.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !9, metadata !2, metadata !2, metadata !10, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ]
!9 = metadata !{metadata !"b.cpp", metadata !"/tmp/dbginfo/b"}
!10 = metadata !{metadata !11}
!11 = metadata !{metadata !"0x2e\00main\00main\00\002\000\001\000\006\00256\000\002", metadata !9, metadata !12, metadata !13, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [def] [main]
!12 = metadata !{metadata !"0x29", metadata !9}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/b/b.cpp]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !15}
!15 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!16 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!18 = metadata !{metadata !"clang version 3.5.0 "}
!19 = metadata !{i32 2, i32 0, metadata !4, null}
!20 = metadata !{i32 3, i32 0, metadata !11, null}
!21 = metadata !{i32 4, i32 0, metadata !11, null}

