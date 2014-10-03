; RUN: opt < %s -add-discriminators -S | FileCheck %s

; We should not generate discriminators for DWARF versions prior to 4.
;
; Original code:
;
; int foo(long i) {
;   if (i < 5) return 2; else return 90;
; }
;
; None of the !dbg nodes associated with the if() statement should be
; altered. If they are, it means that the discriminators pass added a
; new lexical scope.

define i32 @foo(i64 %i) #0 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i64, align 8
  store i64 %i, i64* %i.addr, align 8
  call void @llvm.dbg.declare(metadata !{i64* %i.addr}, metadata !13, metadata !{}), !dbg !14
  %0 = load i64* %i.addr, align 8, !dbg !15
; CHECK:  %0 = load i64* %i.addr, align 8, !dbg !15
  %cmp = icmp slt i64 %0, 5, !dbg !15
; CHECK:  %cmp = icmp slt i64 %0, 5, !dbg !15
  br i1 %cmp, label %if.then, label %if.else, !dbg !15
; CHECK:  br i1 %cmp, label %if.then, label %if.else, !dbg !15

if.then:                                          ; preds = %entry
  store i32 2, i32* %retval, !dbg !15
  br label %return, !dbg !15

if.else:                                          ; preds = %entry
  store i32 90, i32* %retval, !dbg !15
  br label %return, !dbg !15

return:                                           ; preds = %if.else, %if.then
  %1 = load i32* %retval, !dbg !17
  ret i32 %1, !dbg !17
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [./no-discriminators] [DW_LANG_C99]
!1 = metadata !{metadata !"no-discriminators", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, i32 (i64)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [./no-discriminators]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
; CHECK: !10 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!11 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!12 = metadata !{metadata !"clang version 3.5.0 "}
!13 = metadata !{metadata !"0x101\00i\0016777217\000", metadata !4, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [i] [line 1]
!14 = metadata !{i32 1, i32 0, metadata !4, null}
!15 = metadata !{i32 2, i32 0, metadata !16, null}
; CHECK: !15 = metadata !{i32 2, i32 0, metadata !16, null}
!16 = metadata !{metadata !"0xb\002\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [./no-discriminators]
; CHECK: !16 = metadata !{metadata !"0xb\002\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [./no-discriminators]
!17 = metadata !{i32 3, i32 0, metadata !4, null}
