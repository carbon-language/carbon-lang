; RUN: llc -mtriple=i686-w64-mingw32 -o %t -filetype=obj %s
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s
; RUN: llc -mtriple=i686-w64-mingw32 -o %t -filetype=obj -dwarf-version=3 %s
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s -check-prefix=DWARF3

; CHECK:         DW_AT_stmt_list [DW_FORM_sec_offset]
; DWARF3:        DW_AT_stmt_list [DW_FORM_data4]
;
; generated from:
; clang -g -S -emit-llvm test.c -o test.ll
; int main()
; {
;       return 0;
; }

; ModuleID = 'test.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"
target triple = "i686-pc-win32"

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0, !dbg !10
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !11}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.4 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [C:\Projects/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"C:\5CProjects"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00main\00main\00\001\000\001\000\006\000\000\002", metadata !1, metadata !5, metadata !6, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [main]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [C:\Projects/test.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 3, i32 0, metadata !4, null}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
