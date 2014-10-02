; RUN: llvm-link %s %p/DbgDeclare2.ll -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s
; Test if metadata in dbg.declare is mapped properly or not.

; rdar://13089880
; CHECK: define i32 @main(i32 %argc, i8** %argv)
; CHECK: call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: define void @test(i32 %argc, i8** %argv)
; CHECK: call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !{{[0-9]+}}, metadata {{.*}})
; CHECK: call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !{{[0-9]+}}, metadata {{.*}})

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define i32 @main(i32 %argc, i8** %argv) uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !14, metadata !{metadata !"0x102"}), !dbg !15
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !16, metadata !{metadata !"0x102"}), !dbg !15
  %0 = load i32* %argc.addr, align 4, !dbg !17
  %1 = load i8*** %argv.addr, align 8, !dbg !17
  call void @test(i32 %0, i8** %1), !dbg !17
  ret i32 0, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @test(i32, i8**)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{metadata !"0x11\004\00clang version 3.3 (trunk 173515)\001\00\000\00\000", metadata !20, metadata !2, metadata !2, metadata !3, metadata !2, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00main\00main\00\003\000\001\000\006\00256\000\004", metadata !20, null, metadata !7, null, i32 (i32, i8**)* @main, null, null, metadata !1} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9, metadata !10}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !12} ; [ DW_TAG_pointer_type ]
!12 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !13} ; [ DW_TAG_const_type ]
!13 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!14 = metadata !{metadata !"0x101\00argc\0016777219\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 3, i32 0, metadata !5, null}
!16 = metadata !{metadata !"0x101\00argv\0033554435\000", metadata !5, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 5, i32 0, metadata !18, null}
!18 = metadata !{metadata !"0xb\004\000\000", metadata !20, metadata !5} ; [ DW_TAG_lexical_block ]
!19 = metadata !{i32 6, i32 0, metadata !18, null}
!20 = metadata !{metadata !"main.cpp", metadata !"/private/tmp"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
