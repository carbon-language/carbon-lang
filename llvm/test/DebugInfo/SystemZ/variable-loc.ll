; RUN: llc -mtriple=s390x-linux-gnu -disable-fp-elim < %s | FileCheck %s
; RUN: llc -mtriple=s390x-linux-gnu -disable-fp-elim -filetype=obj < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=DEBUG %s
;
; This is a regression test making sure the location of variables is correct in
; debugging information, even if they're addressed via the frame pointer.
; Originally a copy of the AArch64 test, commandeered for SystemZ.
;
; First make sure main_arr is where we expect it: %r11 + 164
;
; CHECK: main:
; CHECK: aghi    %r15, -568
; CHECK: la      %r2, 164(%r11)
; CHECK: brasl   %r14, populate_array@PLT

; DEBUG: DW_TAG_variable
; Rather hard-coded, but 0x91 => DW_OP_fbreg and 0xa401 is SLEB128 encoded 164.
; DEBUG-NOT: DW_TAG
; DEBUG: DW_AT_location {{.*}}(<0x3> 91 a4 01 )
; DEBUG-NOT: DW_TAG
; DEBUG: DW_AT_name {{.*}} "main_arr"


@.str = private unnamed_addr constant [13 x i8] c"Total is %d\0A\00", align 2

declare void @populate_array(i32*, i32) nounwind

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @sum_array(i32*, i32) nounwind

define i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  %main_arr = alloca [100 x i32], align 4
  %val = alloca i32, align 4
  store volatile i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{[100 x i32]* %main_arr}, metadata !17, metadata !{metadata !"0x102"}), !dbg !22
  call void @llvm.dbg.declare(metadata !{i32* %val}, metadata !23, metadata !{metadata !"0x102"}), !dbg !24
  %arraydecay = getelementptr inbounds [100 x i32]* %main_arr, i32 0, i32 0, !dbg !25
  call void @populate_array(i32* %arraydecay, i32 100), !dbg !25
  %arraydecay1 = getelementptr inbounds [100 x i32]* %main_arr, i32 0, i32 0, !dbg !26
  %call = call i32 @sum_array(i32* %arraydecay1, i32 100), !dbg !26
  store i32 %call, i32* %val, align 4, !dbg !26
  %0 = load i32* %val, align 4, !dbg !27
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), i32 %0), !dbg !27
  ret i32 0, !dbg !28
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!30}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.2 \000\00\000\00\000", metadata !29, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ] [/home/timnor01/a64-trunk/build/simple.c] [DW_LANG_C99]
!1 = metadata !{}
!3 = metadata !{metadata !5, metadata !11, metadata !14}
!5 = metadata !{metadata !"0x2e\00populate_array\00populate_array\00\004\000\001\000\006\00256\000\004", metadata !29, metadata !6, metadata !7, null, void (i32*, i32)* @populate_array, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 4] [def] [populate_array]
!6 = metadata !{metadata !"0x29", metadata !29} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !10}
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{metadata !"0x2e\00sum_array\00sum_array\00\009\000\001\000\006\00256\000\009", metadata !29, metadata !6, metadata !12, null, i32 (i32*, i32)* @sum_array, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 9] [def] [sum_array]
!12 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = metadata !{metadata !10, metadata !9, metadata !10}
!14 = metadata !{metadata !"0x2e\00main\00main\00\0018\000\001\000\006\00256\000\0018", metadata !29, metadata !6, metadata !15, null, i32 ()* @main, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 18] [def] [main]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !10}
!17 = metadata !{metadata !"0x100\00main_arr\0019\000", metadata !18, metadata !6, metadata !19} ; [ DW_TAG_auto_variable ] [main_arr] [line 19]
!18 = metadata !{metadata !"0xb\0018\0016\004", metadata !29, metadata !14} ; [ DW_TAG_lexical_block ] [/home/timnor01/a64-trunk/build/simple.c]
!19 = metadata !{metadata !"0x1\00\000\003200\0032\000\000", null, null, metadata !10, metadata !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 3200, align 32, offset 0] [from int]
!20 = metadata !{metadata !"0x21\000\0099"}       ; [ DW_TAG_subrange_type ] [0, 99]
!22 = metadata !{i32 19, i32 7, metadata !18, null}
!23 = metadata !{metadata !"0x100\00val\0020\000", metadata !18, metadata !6, metadata !10} ; [ DW_TAG_auto_variable ] [val] [line 20]
!24 = metadata !{i32 20, i32 7, metadata !18, null}
!25 = metadata !{i32 22, i32 3, metadata !18, null}
!26 = metadata !{i32 23, i32 9, metadata !18, null}
!27 = metadata !{i32 24, i32 3, metadata !18, null}
!28 = metadata !{i32 26, i32 3, metadata !18, null}
!29 = metadata !{metadata !"simple.c", metadata !"/home/timnor01/a64-trunk/build"}
!30 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
