
;RUN: llc < %s -o /dev/null
;Radar 7937109

%struct.anon = type { i64, i32, i32, i32, [1 x i32] }
%struct.gpm_t = type { i32, i8*, [16 x i8], i32, i64, i64, i64, i64, i64, i64, i32, i16, i16, [8 x %struct.gpmr_t] }
%struct.gpmr_t = type { [48 x i8], [48 x i8], [16 x i8], i64, i64, i64, i64, i16 }
%struct.gpt_t = type { [8 x i8], i32, i32, i32, i32, i64, i64, i64, i64, [16 x i8], %struct.anon }

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.gpm_t*, %struct.gpt_t*)* @gpt2gpm to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define fastcc void @gpt2gpm(%struct.gpm_t* %gpm, %struct.gpt_t* %gpt) nounwind optsize ssp {
entry:
  %data_addr.i18 = alloca i64, align 8            ; <i64*> [#uses=1]
  %data_addr.i17 = alloca i64, align 8            ; <i64*> [#uses=2]
  %data_addr.i16 = alloca i64, align 8            ; <i64*> [#uses=0]
  %data_addr.i15 = alloca i32, align 4            ; <i32*> [#uses=0]
  %data_addr.i = alloca i64, align 8              ; <i64*> [#uses=0]
  %0 = getelementptr inbounds %struct.gpm_t* %gpm, i32 0, i32 2, i32 0 ; <i8*> [#uses=1]
  %1 = getelementptr inbounds %struct.gpt_t* %gpt, i32 0, i32 9, i32 0 ; <i8*> [#uses=1]
  call void @uuid_LtoB(i8* %0, i8* %1) nounwind, !dbg !0
  %a9 = load volatile i64* %data_addr.i18, align 8 ; <i64> [#uses=1]
  %a10 = call i64 @llvm.bswap.i64(i64 %a9) nounwind ; <i64> [#uses=1]
  %a11 = getelementptr inbounds %struct.gpt_t* %gpt, i32 0, i32 8, !dbg !7 ; <i64*> [#uses=1]
  %a12 = load i64* %a11, align 4, !dbg !7         ; <i64> [#uses=1]
  call void @llvm.dbg.declare(metadata !{i64* %data_addr.i17}, metadata !8, metadata !{metadata !"0x102"}) nounwind, !dbg !14
  store i64 %a12, i64* %data_addr.i17, align 8
  call void @llvm.dbg.value(metadata !6, i64 0, metadata !15, metadata !{metadata !"0x102"}) nounwind
  call void @llvm.dbg.value(metadata !18, i64 0, metadata !19, metadata !{metadata !"0x102"}) nounwind
  call void @llvm.dbg.declare(metadata !6, metadata !23, metadata !{metadata !"0x102"}) nounwind
  call void @llvm.dbg.value(metadata !{i64* %data_addr.i17}, i64 0, metadata !34, metadata !{metadata !"0x102"}) nounwind
  %a13 = load volatile i64* %data_addr.i17, align 8 ; <i64> [#uses=1]
  %a14 = call i64 @llvm.bswap.i64(i64 %a13) nounwind ; <i64> [#uses=2]
  %a15 = add i64 %a10, %a14, !dbg !7              ; <i64> [#uses=1]
  %a16 = sub i64 %a15, %a14                       ; <i64> [#uses=1]
  %a17 = getelementptr inbounds %struct.gpm_t* %gpm, i32 0, i32 5, !dbg !7 ; <i64*> [#uses=1]
  store i64 %a16, i64* %a17, align 4, !dbg !7
  ret void, !dbg !7
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare i32 @llvm.bswap.i32(i32) nounwind readnone

declare i64 @llvm.bswap.i64(i64) nounwind readnone

declare void @uuid_LtoB(i8*, i8*)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!41}
!0 = metadata !{i32 808, i32 0, metadata !1, null}
!1 = metadata !{metadata !"0xb\00807\000\000", metadata !39, metadata !2} ; [ DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0x2e\00gpt2gpm\00gpt2gpm\00gpt2gpm\00807\001\001\000\006\000\000\000", metadata !39, null, metadata !5, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!3 = metadata !{metadata !"0x29", metadata !39} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !"0x11\001\00llvm-gcc\001\00\000\00\000", metadata !39, metadata !18, metadata !18, metadata !40, null, null} ; [ DW_TAG_compile_unit ]
!5 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !39, metadata !3, null, metadata !6, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{null}
!7 = metadata !{i32 810, i32 0, metadata !1, null}
!8 = metadata !{metadata !"0x101\00data\00201\000", metadata !9, metadata !10, metadata !11} ; [ DW_TAG_arg_variable ]
!9 = metadata !{metadata !"0x2e\00_OSSwapInt64\00_OSSwapInt64\00_OSSwapInt64\00202\001\001\000\006\000\000\000", metadata !10, null, metadata !5, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!10 = metadata !{metadata !"0x29", metadata !"OSByteOrder.h", metadata !"/usr/include/libkern/ppc", metadata !4} ; [ DW_TAG_file_type ]
!11 = metadata !{metadata !"0x16\00uint64_t\0059\000\000\000\000", metadata !36, metadata !3, metadata !13} ; [ DW_TAG_typedef ]
!12 = metadata !{metadata !"0x29", metadata !"stdint.h", metadata !"/usr/4.2.1/include", metadata !4} ; [ DW_TAG_file_type ]
!13 = metadata !{metadata !"0x24\00long long unsigned int\000\0064\0064\000\000\007", metadata !39, metadata !3} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 202, i32 0, metadata !9, metadata !7}
!15 = metadata !{metadata !"0x101\00base\0092\000", metadata !16, metadata !10, metadata !17} ; [ DW_TAG_arg_variable ]
!16 = metadata !{metadata !"0x2e\00OSReadSwapInt64\00OSReadSwapInt64\00OSReadSwapInt64\0095\001\001\000\006\000\000\000", metadata !38, null, metadata !5, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!17 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !39, metadata !3, null} ; [ DW_TAG_pointer_type ]
!18 = metadata !{i32 0}
!19 = metadata !{metadata !"0x101\00byteOffset\0094\000", metadata !16, metadata !10, metadata !20} ; [ DW_TAG_arg_variable ]
!20 = metadata !{metadata !"0x16\00uintptr_t\00114\000\000\000\000", metadata !37, metadata !3, metadata !22} ; [ DW_TAG_typedef ]
!21 = metadata !{metadata !"0x29", metadata !"types.h", metadata !"/usr/include/ppc", metadata !4} ; [ DW_TAG_file_type ]
!22 = metadata !{metadata !"0x24\00long unsigned int\000\0032\0032\000\000\007", metadata !39, metadata !3} ; [ DW_TAG_base_type ]
!23 = metadata !{metadata !"0x100\00u\00100\000", metadata !24, metadata !10, metadata !25} ; [ DW_TAG_auto_variable ]
!24 = metadata !{metadata !"0xb\0095\000\000", metadata !38, metadata !16} ; [ DW_TAG_lexical_block ]
!25 = metadata !{metadata !"0x17\00\0097\0064\0064\000\000\000", metadata !38, metadata !16, null, metadata !26, null, null, null} ; [ DW_TAG_union_type ] [line 97, size 64, align 64, offset 0] [def] [from ]
!26 = metadata !{metadata !27, metadata !28}
!27 = metadata !{metadata !"0xd\00u64\0098\0064\0064\000\000", metadata !38, metadata !25, metadata !11} ; [ DW_TAG_member ]
!28 = metadata !{metadata !"0xd\00u32\0099\0064\0032\000\000", metadata !38, metadata !25, metadata !29} ; [ DW_TAG_member ]
!29 = metadata !{metadata !"0x1\00\000\0064\0032\000\000", metadata !39, metadata !3, metadata !30, metadata !32, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 64, align 32, offset 0] [from uint32_t]
!30 = metadata !{metadata !"0x16\00uint32_t\0055\000\000\000\000", metadata !36, metadata !3, metadata !31} ; [ DW_TAG_typedef ]
!31 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", metadata !39, metadata !3} ; [ DW_TAG_base_type ]
!32 = metadata !{metadata !33}
!33 = metadata !{metadata !"0x21\000\002"}        ; [ DW_TAG_subrange_type ]
!34 = metadata !{metadata !"0x100\00addr\0096\000", metadata !24, metadata !10, metadata !35} ; [ DW_TAG_auto_variable ]
!35 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !39, metadata !3, metadata !11} ; [ DW_TAG_pointer_type ]
!36 = metadata !{metadata !"stdint.h", metadata !"/usr/4.2.1/include"}
!37 = metadata !{metadata !"types.h", metadata !"/usr/include/ppc"}
!38 = metadata !{metadata !"OSByteOrder.h", metadata !"/usr/include/libkern/ppc"}
!39 = metadata !{metadata !"G.c", metadata !"/tmp"}
!40 = metadata !{metadata !2, metadata !9, metadata !16}
!41 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
