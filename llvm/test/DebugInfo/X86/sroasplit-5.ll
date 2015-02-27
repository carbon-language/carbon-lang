; RUN: opt %s -sroa -verify -S -o - | FileCheck %s
; From:
; struct prog_src_register {
;   unsigned : 4;       
;   int Index : 12 + 1; 
;   unsigned : 12;      
;   unsigned : 4;       
;   int : 12 + 1        
; } src_reg_for_float() {
;   struct prog_src_register a;
;   memset(&a, 0, sizeof(a));
;   int local = a.Index;
;   return a;
; }
; ModuleID = 'pr22495.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; When SROA is creating new smaller allocas, it may add padding.
;
; There should be no debug info for the padding.
; CHECK-NOT: DW_OP_bit_piece offset=56
; CHECK: [ DW_TAG_expression ] [DW_OP_bit_piece offset=32, size=24]
; CHECK-NOT: DW_OP_bit_piece offset=56
; CHECK: [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=32]
; CHECK-NOT: DW_OP_bit_piece offset=56
%struct.prog_src_register = type { i32, i24 }

; Function Attrs: nounwind
define i64 @src_reg_for_float() #0 {
entry:
  %retval = alloca %struct.prog_src_register, align 4
  %a = alloca %struct.prog_src_register, align 4
  %local = alloca i32, align 4
  call void @llvm.dbg.declare(metadata %struct.prog_src_register* %a, metadata !16, metadata !17), !dbg !18
  %0 = bitcast %struct.prog_src_register* %a to i8*, !dbg !19
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 8, i32 4, i1 false), !dbg !19
  call void @llvm.dbg.declare(metadata i32* %local, metadata !20, metadata !17), !dbg !21
  %1 = bitcast %struct.prog_src_register* %a to i32*, !dbg !21
  %bf.load = load i32, i32* %1, align 4, !dbg !21
  %bf.shl = shl i32 %bf.load, 15, !dbg !21
  %bf.ashr = ashr i32 %bf.shl, 19, !dbg !21
  store i32 %bf.ashr, i32* %local, align 4, !dbg !21
  %2 = bitcast %struct.prog_src_register* %retval to i8*, !dbg !22
  %3 = bitcast %struct.prog_src_register* %a to i8*, !dbg !22
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %3, i64 8, i32 4, i1 false), !dbg !22
  %4 = bitcast %struct.prog_src_register* %retval to i64*, !dbg !22
  %5 = load i64, i64* %4, align 1, !dbg !22
  ret i64 %5, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #2

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = !{!"0x11\0012\00clang version 3.7.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/<stdin>] [DW_LANG_C99]
!1 = !{!"<stdin>", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00src_reg_for_float\00src_reg_for_float\00\007\000\001\000\000\000\000\007", !5, !6, !7, null, i64 ()* @src_reg_for_float, null, null, !2} ; [ DW_TAG_subprogram ] [line 7] [def] [src_reg_for_float]
!5 = !{!"pr22495.c", !""}
!6 = !{!"0x29", !5}                               ; [ DW_TAG_file_type ] [/pr22495.c]
!7 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x13\00prog_src_register\001\0064\0032\000\000\000", !5, null, null, !10, null, null, null} ; [ DW_TAG_structure_type ] [prog_src_register] [line 1, size 64, align 32, offset 0] [def] [from ]
!10 = !{!11}
!11 = !{!"0xd\00Index\003\0013\0032\004\000", !5, !9, !12} ; [ DW_TAG_member ] [Index] [line 3, size 13, align 32, offset 4] [from int]
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 2}
!15 = !{!"clang version 3.7.0 "}
!16 = !{!"0x100\00a\008\000", !4, !6, !9}         ; [ DW_TAG_auto_variable ] [a] [line 8]
!17 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!18 = !MDLocation(line: 8, scope: !4)
!19 = !MDLocation(line: 9, scope: !4)
!20 = !{!"0x100\00local\0010\000", !4, !6, !12}   ; [ DW_TAG_auto_variable ] [local] [line 10]
!21 = !MDLocation(line: 10, scope: !4)
!22 = !MDLocation(line: 11, scope: !4)
