; REQUIRES: object-emission
; RUN: %llc_dwarf -O2  -dwarf-version 2 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s  --check-prefix=DWARF23
; RUN: %llc_dwarf -O2  -dwarf-version 3 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s  --check-prefix=DWARF23
; RUN: %llc_dwarf -O2  -dwarf-version 4 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s  --check-prefix=DWARF4

; This is a test for PR21176.
; DW_OP_const <const> doesn't describe a constant value, but a value at a constant address. 
; The proper way to describe a constant value is DW_OP_constu <const>, DW_OP_stack_value.

; Generated with clang -S -emit-llvm -g -O2 test.cpp

; extern int func();
; 
; int main()
; {
;   volatile int c = 13;
;   c = func();
;   return c;
; }

; DWARF23: Location description: 10 0d {{$}}
; DWARF4: Location description: 10 0d 9f

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %c = alloca i32, align 4
  tail call void @llvm.dbg.value(metadata !15, i64 0, metadata !10, metadata !16), !dbg !17
  store volatile i32 13, i32* %c, align 4, !dbg !18
  %call = tail call i32 @_Z4funcv(), !dbg !19
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !10, metadata !16), !dbg !17
  store volatile i32 %call, i32* %c, align 4, !dbg !19
  tail call void @llvm.dbg.value(metadata !{i32* %c}, i64 0, metadata !10, metadata !16), !dbg !17
  %c.0.c.0. = load volatile i32* %c, align 4, !dbg !20
  ret i32 %c.0.c.0., !dbg !20
}

declare i32 @_Z4funcv() #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 (trunk 223522)\001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/home/kromanova/ngh/ToT_latest/llvm/test/DebugInfo/test.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"test.cpp", metadata !"/home/kromanova/ngh/ToT_latest/llvm/test/DebugInfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00main\00main\00\003\000\001\000\000\00256\001\004", metadata !1, metadata !5, metadata !6, null, i32 ()* @main, null, null, metadata !9} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [main]
!5 = metadata !{metadata !"0x29", metadata !1}    ; [ DW_TAG_file_type ] [/home/kromanova/ngh/ToT_latest/llvm/test/DebugInfo/test.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x100\00c\005\000", metadata !4, metadata !5, metadata !11} ; [ DW_TAG_auto_variable ] [c] [line 5]
!11 = metadata !{metadata !"0x35\00\000\000\000\000\000", null, null, metadata !8} ; [ DW_TAG_volatile_type ] [line 0, size 0, align 0, offset 0] [from int]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!13 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!14 = metadata !{metadata !"clang version 3.6.0 (trunk 223522)"}
!15 = metadata !{i32 13}
!16 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!17 = metadata !{i32 5, i32 16, metadata !4, null}
!18 = metadata !{i32 5, i32 3, metadata !4, null}
!19 = metadata !{i32 6, i32 7, metadata !4, null}
!20 = metadata !{i32 7, i32 3, metadata !4, null}

