; RUN: llc < %s -O2 -mtriple=aarch64-none-linux-gnu 

; Bug 20598


define void @test() #0 {
entry:
  br label %for.body, !dbg !39

for.body:                                         ; preds = %for.body, %entry
  %arrayidx5 = getelementptr inbounds i32* null, i64 1, !dbg !43
  %0 = load i32* null, align 4, !dbg !45, !tbaa !46
  %s1 = sub nsw i32 0, %0, !dbg !50
  %n1 = sext i32 %s1 to i64, !dbg !50
  %arrayidx21 = getelementptr inbounds i32* null, i64 3, !dbg !51
  %add53 = add nsw i64 %n1, 0, !dbg !52
  %add55 = add nsw i64 %n1, 0, !dbg !53
  %mul63 = mul nsw i64 %add53, -20995, !dbg !54
  tail call void @llvm.dbg.value(metadata !{i64 %mul63}, i64 0, metadata !30, metadata !{metadata !"0x102"}), !dbg !55
  %mul65 = mul nsw i64 %add55, -3196, !dbg !56
  %add67 = add nsw i64 0, %mul65, !dbg !57
  %add80 = add i64 0, 1024, !dbg !58
  %add81 = add i64 %add80, %mul63, !dbg !58
  %add82 = add i64 %add81, 0, !dbg !58
  %shr83351 = lshr i64 %add82, 11, !dbg !58
  %conv84 = trunc i64 %shr83351 to i32, !dbg !58
  store i32 %conv84, i32* %arrayidx21, align 4, !dbg !58, !tbaa !46
  %add86 = add i64 0, 1024, !dbg !59
  %add87 = add i64 %add86, 0, !dbg !59
  %add88 = add i64 %add87, %add67, !dbg !59
  %shr89352 = lshr i64 %add88, 11, !dbg !59
  %n2 = trunc i64 %shr89352 to i32, !dbg !59
  store i32 %n2, i32* %arrayidx5, align 4, !dbg !59, !tbaa !46
  br label %for.body, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}
!llvm.ident = !{!38}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.6.0 \001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [] [] []
!1 = metadata !{metadata !"test.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00\00\00\00140\000\001\000\006\00256\001\00141", metadata !1, metadata !5, metadata !6, null, void ()* @test, null, null, metadata !12} ; [ DW_TAG_subprogram ] [] [] [def] [scope 141] []
!5 = metadata !{metadata !"0x29", metadata !1} ; [ DW_TAG_file_type ] [] []
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [] [] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [] [] []
!9 = metadata !{metadata !"0x16\00\0030\000\000\000\000", metadata !10, null, metadata !11} ; [ DW_TAG_typedef ] [] [] [] [from int]
!10 = metadata !{metadata !"", metadata !""}
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [] [int] []
!12 = metadata !{metadata !13, metadata !14, metadata !18, metadata !19, metadata !20, metadata !21, metadata !22, metadata !23, metadata !24, metadata !25, metadata !26, metadata !27, metadata !28, metadata !29, metadata !30, metadata !31, metadata !32, metadata !33, metadata !34, metadata !35}
!13 = metadata !{metadata !"0x101\00\0016777356\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [] [data] []
!14 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!15 = metadata !{metadata !"0x16\00\00183\000\000\000\000", metadata !16, null, metadata !17} ; [ DW_TAG_typedef ] [] [INT32] [] [from long int]
!16 = metadata !{metadata !"", metadata !""}
!17 = metadata !{metadata !"0x24\00\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [] [long int] []
!18 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!19 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!20 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!21 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!22 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!23 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [] [] []
!24 = metadata !{metadata !"0x100\00\00142\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!25 = metadata !{metadata !"0x100\00\00143\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!26 = metadata !{metadata !"0x100\00\00143\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!27 = metadata !{metadata !"0x100\00\00143\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!28 = metadata !{metadata !"0x100\00\00143\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!29 = metadata !{metadata !"0x100\00\00144\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!30 = metadata !{metadata !"0x100\00\00144\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!31 = metadata !{metadata !"0x100\00\00144\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!32 = metadata !{metadata !"0x100\00\00144\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [ ] [] []
!33 = metadata !{metadata !"0x100\00\00144\000", metadata !4, metadata !5, metadata !15} ; [ DW_TAG_auto_variable ] [  ] [] []
!34 = metadata !{metadata !"0x100\00\00145\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_auto_variable ] [  ] [] []
!35 = metadata !{metadata !"0x100\00\00146\000", metadata !4, metadata !5, metadata !11} ; [ DW_TAG_auto_variable ] [  ] [] []
!36 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!37 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!38 = metadata !{metadata !"clang version 3.6.0 "}
!39 = metadata !{i32 154, i32 8, metadata !40, null}
!40 = metadata !{metadata !"0xb\00154\008\002", metadata !1, metadata !41} ; [ DW_TAG_lexical_block ] [  ] []
!41 = metadata !{metadata !"0xb\00154\008\001", metadata !1, metadata !42} ; [ DW_TAG_lexical_block ] [  ] []
!42 = metadata !{metadata !"0xb\00154\003\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [  ] []
!43 = metadata !{i32 157, i32 5, metadata !44, null}
!44 = metadata !{metadata !"0xb\00154\0042\000", metadata !1, metadata !42} ; [ DW_TAG_lexical_block ] [  ] []
!45 = metadata !{i32 159, i32 5, metadata !44, null}
!46 = metadata !{metadata !47, metadata !47, i64 0}
!47 = metadata !{metadata !"int", metadata !48, i64 0}
!48 = metadata !{metadata !"omnipotent char", metadata !49, i64 0}
!49 = metadata !{metadata !"Simple C/C++ TBAA"}
!50 = metadata !{i32 160, i32 5, metadata !44, null}
!51 = metadata !{i32 161, i32 5, metadata !44, null}
!52 = metadata !{i32 188, i32 5, metadata !44, null}
!53 = metadata !{i32 190, i32 5, metadata !44, null}
!54 = metadata !{i32 198, i32 5, metadata !44, null}
!55 = metadata !{i32 144, i32 13, metadata !4, null}
!56 = metadata !{i32 200, i32 5, metadata !44, null}
!57 = metadata !{i32 203, i32 5, metadata !44, null}
!58 = metadata !{i32 207, i32 5, metadata !44, null}
!59 = metadata !{i32 208, i32 5, metadata !44, null}
