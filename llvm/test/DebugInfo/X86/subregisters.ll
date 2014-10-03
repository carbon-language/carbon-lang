; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t.o -filetype=obj -O0
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; Test that on x86_64, the 32-bit subregister esi is emitted as
; DW_OP_piece 32 of the 64-bit rsi.
;
; rdar://problem/16015314
;
; CHECK:  DW_AT_location [DW_FORM_block1]       (<0x03> 54 93 04 )
; CHECK:  DW_AT_name [DW_FORM_strp]{{.*}} "a"
;
; struct bar {
;   int a;
;   int b;
; };
;
; void doSomething() __attribute__ ((noinline));
;
; void doSomething(struct bar *b)
; {
;   int a = b->a;
;   printf("%d\n", a); // set breakpoint here
; }
;
; int main()
; {
;   struct bar myBar = { 3, 4 };
;   doSomething(&myBar);
;   return 0;
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.bar = type { i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@main.myBar = private unnamed_addr constant %struct.bar { i32 3, i32 4 }, align 4

; Function Attrs: noinline nounwind ssp uwtable
define void @doSomething(%struct.bar* nocapture readonly %b) #0 {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.bar* %b}, i64 0, metadata !15, metadata !{metadata !"0x102"}), !dbg !25
  %a1 = getelementptr inbounds %struct.bar* %b, i64 0, i32 0, !dbg !26
  %0 = load i32* %a1, align 4, !dbg !26, !tbaa !27
  tail call void @llvm.dbg.value(metadata !{i32 %0}, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !26
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32 %0) #4, !dbg !32
  ret void, !dbg !33
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #2

; Function Attrs: nounwind ssp uwtable
define i32 @main() #3 {
entry:
  %myBar = alloca i64, align 8, !dbg !34
  %tmpcast = bitcast i64* %myBar to %struct.bar*, !dbg !34
  tail call void @llvm.dbg.declare(metadata !{%struct.bar* %tmpcast}, metadata !21, metadata !{metadata !"0x102"}), !dbg !34
  store i64 17179869187, i64* %myBar, align 8, !dbg !34
  call void @doSomething(%struct.bar* %tmpcast), !dbg !35
  ret i32 0, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { noinline nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { nounwind ssp uwtable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5 \001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [subregisters.c] [DW_LANG_C99]
!1 = metadata !{metadata !"subregisters.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !17}
!4 = metadata !{metadata !"0x2e\00doSomething\00doSomething\00\0010\000\001\000\006\00256\001\0011", metadata !1, metadata !5, metadata !6, null, void (%struct.bar*)* @doSomething, null, null, metadata !14} ; [ DW_TAG_subprogram ] [line 10] [def] [scope 11] [doSomething]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [subregisters.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from bar]
!9 = metadata !{metadata !"0x13\00bar\003\0064\0032\000\000\000", metadata !1, null, null, metadata !10, null, null, null} ; [ DW_TAG_structure_type ] [bar] [line 3, size 64, align 32, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !13}
!11 = metadata !{metadata !"0xd\00a\004\0032\0032\000\000", metadata !1, metadata !9, metadata !12} ; [ DW_TAG_member ] [a] [line 4, size 32, align 32, offset 0] [from int]
!12 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{metadata !"0xd\00b\005\0032\0032\0032\000", metadata !1, metadata !9, metadata !12} ; [ DW_TAG_member ] [b] [line 5, size 32, align 32, offset 32] [from int]
!14 = metadata !{metadata !15, metadata !16}
!15 = metadata !{metadata !"0x101\00b\0016777226\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [b] [line 10]
!16 = metadata !{metadata !"0x100\00a\0012\000", metadata !4, metadata !5, metadata !12} ; [ DW_TAG_auto_variable ] [a] [line 12]
!17 = metadata !{metadata !"0x2e\00main\00main\00\0016\000\001\000\006\000\001\0017", metadata !1, metadata !5, metadata !18, null, i32 ()* @main, null, null, metadata !20} ; [ DW_TAG_subprogram ] [line 16] [def] [scope 17] [main]
!18 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !19, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = metadata !{metadata !12}
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"0x100\00myBar\0018\000", metadata !17, metadata !5, metadata !9} ; [ DW_TAG_auto_variable ] [myBar] [line 18]
!22 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!23 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!24 = metadata !{metadata !"clang version 3.5 "}
!25 = metadata !{i32 10, i32 0, metadata !4, null}
!26 = metadata !{i32 12, i32 0, metadata !4, null}
!27 = metadata !{metadata !28, metadata !29, i64 0}
!28 = metadata !{metadata !"bar", metadata !29, i64 0, metadata !29, i64 4}
!29 = metadata !{metadata !"int", metadata !30, i64 0}
!30 = metadata !{metadata !"omnipotent char", metadata !31, i64 0}
!31 = metadata !{metadata !"Simple C/C++ TBAA"}
!32 = metadata !{i32 13, i32 0, metadata !4, null}
!33 = metadata !{i32 14, i32 0, metadata !4, null}
!34 = metadata !{i32 18, i32 0, metadata !17, null}
!35 = metadata !{i32 19, i32 0, metadata !17, null}
!36 = metadata !{i32 20, i32 0, metadata !17, null}
