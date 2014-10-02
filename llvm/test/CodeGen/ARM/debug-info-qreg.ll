; RUN: llc < %s - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

;CHECK: sub-register
;CHECK-NEXT: DW_OP_regx
;CHECK-NEXT: ascii
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: byte   8
;CHECK-NEXT: sub-register
;CHECK-NEXT: DW_OP_regx
;CHECK-NEXT: ascii
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: byte   8

@.str = external constant [13 x i8]

declare <4 x float> @test0001(float) nounwind readnone ssp

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %entry
  %add19 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  br i1 undef, label %for.end54, label %for.body9, !dbg !44

for.end54:                                        ; preds = %for.body9
  tail call void @llvm.dbg.value(metadata !{<4 x float> %add19}, i64 0, metadata !27, metadata !{metadata !"0x102"}), !dbg !39
  %tmp115 = extractelement <4 x float> %add19, i32 1
  %conv6.i75 = fpext float %tmp115 to double, !dbg !45
  %call.i82 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i75, double undef, double undef) nounwind, !dbg !45
  ret i32 0, !dbg !49
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!56}

!0 = metadata !{metadata !"0x2e\00test0001\00test0001\00\003\000\001\000\006\00256\001\003", metadata !54, metadata !1, metadata !3, null, <4 x float> (float)* @test0001, null, null, metadata !51} ; [ DW_TAG_subprogram ] [line 3] [def] [test0001]
!1 = metadata !{metadata !"0x29", metadata !54} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 129915)\001\00\000\00\001", metadata !54, metadata !17, metadata !17, metadata !50, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !54, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x16\00v4f32\0014\000\000\000\000", metadata !54, metadata !2, metadata !6} ; [ DW_TAG_typedef ]
!6 = metadata !{metadata !"0x1\00\000\00128\00128\000\000", metadata !2, null, metadata !7, metadata !8, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [from float]
!7 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, metadata !2} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x21\000\004"}         ; [ DW_TAG_subrange_type ]
!10 = metadata !{metadata !"0x2e\00main\00main\00\0059\000\001\000\006\00256\001\0059", metadata !54, metadata !1, metadata !11, null, i32 (i32, i8**)* @main, null, null, metadata !52} ; [ DW_TAG_subprogram ] [line 59] [def] [main]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !54, metadata !1, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!14 = metadata !{metadata !"0x2e\00printFV\00printFV\00\0041\001\001\000\006\00256\001\0041", metadata !55, metadata !15, metadata !16, null, null, null, null, metadata !53} ; [ DW_TAG_subprogram ] [line 41] [local] [def] [printFV]
!15 = metadata !{metadata !"0x29", metadata !55} ; [ DW_TAG_file_type ]
!16 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !55, metadata !15, null, metadata !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null}
!18 = metadata !{metadata !"0x101\00a\0016777219\000", metadata !0, metadata !1, metadata !7} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0x101\00argc\0016777275\000", metadata !10, metadata !1, metadata !13} ; [ DW_TAG_arg_variable ]
!20 = metadata !{metadata !"0x101\00argv\0033554491\000", metadata !10, metadata !1, metadata !21} ; [ DW_TAG_arg_variable ]
!21 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, metadata !22} ; [ DW_TAG_pointer_type ]
!22 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, metadata !23} ; [ DW_TAG_pointer_type ]
!23 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, metadata !2} ; [ DW_TAG_base_type ]
!24 = metadata !{metadata !"0x100\00i\0060\000", metadata !25, metadata !1, metadata !13} ; [ DW_TAG_auto_variable ]
!25 = metadata !{metadata !"0xb\0059\0033\0014", metadata !54, metadata !10} ; [ DW_TAG_lexical_block ]
!26 = metadata !{metadata !"0x100\00j\0060\000", metadata !25, metadata !1, metadata !13} ; [ DW_TAG_auto_variable ]
!27 = metadata !{metadata !"0x100\00x\0061\000", metadata !25, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!28 = metadata !{metadata !"0x100\00y\0062\000", metadata !25, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!29 = metadata !{metadata !"0x100\00z\0063\000", metadata !25, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!30 = metadata !{metadata !"0x101\00F\0016777257\000", metadata !14, metadata !15, metadata !31} ; [ DW_TAG_arg_variable ]
!31 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, metadata !32} ; [ DW_TAG_pointer_type ]
!32 = metadata !{metadata !"0x16\00FV\0025\000\000\000\000", metadata !55, metadata !2, metadata !33} ; [ DW_TAG_typedef ]
!33 = metadata !{metadata !"0x17\00\0022\00128\00128\000\000\000", metadata !55, metadata !2, i32 0, metadata !34, null} ; [ DW_TAG_union_type ]
!34 = metadata !{metadata !35, metadata !37}
!35 = metadata !{metadata !"0xd\00V\0023\00128\00128\000\000", metadata !55, metadata !15, metadata !36} ; [ DW_TAG_member ]
!36 = metadata !{metadata !"0x16\00v4sf\003\000\000\000\000", metadata !55, metadata !2, metadata !6} ; [ DW_TAG_typedef ]
!37 = metadata !{metadata !"0xd\00A\0024\00128\0032\000\000", metadata !55, metadata !15, metadata !38} ; [ DW_TAG_member ]
!38 = metadata !{metadata !"0x1\00\000\00128\0032\000\000", null, metadata !2, metadata !7, metadata !8, i32 0, i32 0} ; [ DW_TAG_array_type ]
!39 = metadata !{i32 79, i32 7, metadata !40, null}
!40 = metadata !{metadata !"0xb\0075\0035\0018", metadata !54, metadata !41} ; [ DW_TAG_lexical_block ]
!41 = metadata !{metadata !"0xb\0075\005\0017", metadata !54, metadata !42} ; [ DW_TAG_lexical_block ]
!42 = metadata !{metadata !"0xb\0071\0032\0016", metadata !54, metadata !43} ; [ DW_TAG_lexical_block ]
!43 = metadata !{metadata !"0xb\0071\003\0015", metadata !54, metadata !25} ; [ DW_TAG_lexical_block ]
!44 = metadata !{i32 75, i32 5, metadata !42, null}
!45 = metadata !{i32 42, i32 2, metadata !46, metadata !48}
!46 = metadata !{metadata !"0xb\0042\002\0020", metadata !55, metadata !47} ; [ DW_TAG_lexical_block ]
!47 = metadata !{metadata !"0xb\0041\0028\0019", metadata !55, metadata !14} ; [ DW_TAG_lexical_block ]
!48 = metadata !{i32 95, i32 3, metadata !25, null}
!49 = metadata !{i32 99, i32 3, metadata !25, null}
!50 = metadata !{metadata !0, metadata !10, metadata !14}
!51 = metadata !{metadata !18}
!52 = metadata !{metadata !19, metadata !20, metadata !24, metadata !26, metadata !27, metadata !28, metadata !29}
!53 = metadata !{metadata !30}
!54 = metadata !{metadata !"build2.c", metadata !"/private/tmp"}
!55 = metadata !{metadata !"/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", metadata !"/private/tmp"}
!56 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
