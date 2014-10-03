; RUN: llc < %s
; RUN: llc < %s -march=x86-64
; PR3538
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
define signext i8 @foo(i8* %s1) nounwind ssp {
entry:
  %s1_addr = alloca i8*                           ; <i8**> [#uses=2]
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %saved_stack.1 = alloca i8*                     ; <i8**> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %str.0 = alloca [0 x i8]*                       ; <[0 x i8]**> [#uses=3]
  %1 = alloca i64                                 ; <i64*> [#uses=2]
  %2 = alloca i64                                 ; <i64*> [#uses=1]
  %3 = alloca i64                                 ; <i64*> [#uses=6]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{i8** %s1_addr}, metadata !0, metadata !{metadata !"0x102"}), !dbg !7
  store i8* %s1, i8** %s1_addr
  call void @llvm.dbg.declare(metadata !{[0 x i8]** %str.0}, metadata !8, metadata !{metadata !"0x102"}), !dbg !7
  %4 = call i8* @llvm.stacksave(), !dbg !7        ; <i8*> [#uses=1]
  store i8* %4, i8** %saved_stack.1, align 8, !dbg !7
  %5 = load i8** %s1_addr, align 8, !dbg !13      ; <i8*> [#uses=1]
  %6 = call i64 @strlen(i8* %5) nounwind readonly, !dbg !13 ; <i64> [#uses=1]
  %7 = add i64 %6, 1, !dbg !13                    ; <i64> [#uses=1]
  store i64 %7, i64* %3, align 8, !dbg !13
  %8 = load i64* %3, align 8, !dbg !13            ; <i64> [#uses=1]
  %9 = sub nsw i64 %8, 1, !dbg !13                ; <i64> [#uses=0]
  %10 = load i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  %11 = mul i64 %10, 8, !dbg !13                  ; <i64> [#uses=0]
  %12 = load i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  store i64 %12, i64* %2, align 8, !dbg !13
  %13 = load i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  %14 = mul i64 %13, 8, !dbg !13                  ; <i64> [#uses=0]
  %15 = load i64* %3, align 8, !dbg !13           ; <i64> [#uses=1]
  store i64 %15, i64* %1, align 8, !dbg !13
  %16 = load i64* %1, align 8, !dbg !13           ; <i64> [#uses=1]
  %17 = trunc i64 %16 to i32, !dbg !13            ; <i32> [#uses=1]
  %18 = alloca i8, i32 %17, !dbg !13              ; <i8*> [#uses=1]
  %19 = bitcast i8* %18 to [0 x i8]*, !dbg !13    ; <[0 x i8]*> [#uses=1]
  store [0 x i8]* %19, [0 x i8]** %str.0, align 8, !dbg !13
  %20 = load [0 x i8]** %str.0, align 8, !dbg !15 ; <[0 x i8]*> [#uses=1]
  %21 = getelementptr inbounds [0 x i8]* %20, i64 0, i64 0, !dbg !15 ; <i8*> [#uses=1]
  store i8 0, i8* %21, align 1, !dbg !15
  %22 = load [0 x i8]** %str.0, align 8, !dbg !16 ; <[0 x i8]*> [#uses=1]
  %23 = getelementptr inbounds [0 x i8]* %22, i64 0, i64 0, !dbg !16 ; <i8*> [#uses=1]
  %24 = load i8* %23, align 1, !dbg !16           ; <i8> [#uses=1]
  %25 = sext i8 %24 to i32, !dbg !16              ; <i32> [#uses=1]
  store i32 %25, i32* %0, align 4, !dbg !16
  %26 = load i8** %saved_stack.1, align 8, !dbg !16 ; <i8*> [#uses=1]
  call void @llvm.stackrestore(i8* %26), !dbg !16
  %27 = load i32* %0, align 4, !dbg !16           ; <i32> [#uses=1]
  store i32 %27, i32* %retval, align 4, !dbg !16
  br label %return, !dbg !16

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !16          ; <i32> [#uses=1]
  %retval12 = trunc i32 %retval1 to i8, !dbg !16  ; <i8> [#uses=1]
  ret i8 %retval12, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare i64 @strlen(i8*) nounwind readonly

declare void @llvm.stackrestore(i8*) nounwind

!0 = metadata !{metadata !"0x101\00s1\002\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00foo\002\000\001\000\006\000\000\000", i32 0, metadata !2, metadata !3, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !17, metadata !18, metadata !18, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5, metadata !6}
!5 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, metadata !2, metadata !5} ; [ DW_TAG_pointer_type ]
!7 = metadata !{i32 2, i32 0, metadata !1, null}
!8 = metadata !{metadata !"0x100\00str.0\003\000", metadata !1, metadata !2, metadata !9} ; [ DW_TAG_auto_variable ]
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", null, metadata !2, metadata !10} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x1\00\000\008\008\000\000", null, metadata !2, metadata !5, metadata !11, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 8, align 8, offset 0] [from char]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0x21\000\001"}        ; [ DW_TAG_subrange_type ]
!13 = metadata !{i32 3, i32 0, metadata !14, null}
!14 = metadata !{metadata !"0xb\000\000\000", metadata !17, metadata !1} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 4, i32 0, metadata !14, null}
!16 = metadata !{i32 5, i32 0, metadata !14, null}
!17 = metadata !{metadata !"vla.c", metadata !"/tmp/"}
!18 = metadata !{i32 0}
