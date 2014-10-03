; RUN: llc -filetype=obj -O0 < %s -mtriple mips-unknown-linux-gnu | llvm-dwarfdump - | FileCheck %s
; PR19815

; Generated using clang -target mips-linux-gnu -g test.c -S -o - -flto|opt -sroa -S
; test.c:
;
; int foo(int x) {
;  if (x)
;    return 0;
;  return 1;
; }

; CHECK: Address            Line   Column File   ISA Discriminator Flags
; CHECK: ------------------ ------ ------ ------ --- ------------- -------------
; CHECK: 0x0000000000000000      1      0      1   0             0  is_stmt
; CHECK: 0x0000000000000000      1      0      1   0             0  is_stmt prologue_end
; CHECK: 0x0000000000000008      2      0      1   0             0  is_stmt
; CHECK: 0x0000000000000020      3      0      1   0             0  is_stmt
; CHECK: 0x0000000000000030      4      0      1   0             0  is_stmt
; CHECK: 0x0000000000000040      5      0      1   0             0  is_stmt
; CHECK: 0x0000000000000050      5      0      1   0             0  is_stmt end_sequence

target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips--linux-gnu"

; Function Attrs: nounwind
define i32 @foo(i32 %x) #0 {
entry:
  call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !12, metadata !{metadata !"0x102"}), !dbg !13
  %tobool = icmp ne i32 %x, 0, !dbg !14
  br i1 %tobool, label %if.then, label %if.end, !dbg !14

if.then:                                          ; preds = %entry
  br label %return, !dbg !16

if.end:                                           ; preds = %entry
  br label %return, !dbg !17

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.end ]
  ret i32 %retval.0, !dbg !18
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, i32 (i32)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/test.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!11 = metadata !{metadata !"clang version 3.5.0"}
!12 = metadata !{metadata !"0x101\00x\0016777217\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [x] [line 1]
!13 = metadata !{i32 1, i32 0, metadata !4, null}
!14 = metadata !{i32 2, i32 0, metadata !15, null}
!15 = metadata !{metadata !"0xb\002\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [/tmp/test.c]
!16 = metadata !{i32 3, i32 0, metadata !15, null}
!17 = metadata !{i32 4, i32 0, metadata !4, null}
!18 = metadata !{i32 5, i32 0, metadata !4, null}
