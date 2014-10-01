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
  tail call void @llvm.dbg.value(metadata !{i64 %mul63}, i64 0, metadata !30), !dbg !55
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
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}
!llvm.ident = !{!38}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [] [] []
!1 = metadata !{metadata !"test.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"", metadata !"", metadata !"", i32 140, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @test, null, null, metadata !12, i32 141} ; [] [] [def] [scope 141] []
!5 = metadata !{i32 786473, metadata !1}          ; [] []
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [] [] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [] [] []
!9 = metadata !{i32 786454, metadata !10, null, metadata !"", i32 30, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [] [] [] [from int]
!10 = metadata !{metadata !"", metadata !""}
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [] [int] []
!12 = metadata !{metadata !13, metadata !14, metadata !18, metadata !19, metadata !20, metadata !21, metadata !22, metadata !23, metadata !24, metadata !25, metadata !26, metadata !27, metadata !28, metadata !29, metadata !30, metadata !31, metadata !32, metadata !33, metadata !34, metadata !35}
!13 = metadata !{i32 786689, metadata !4, metadata !"", metadata !5, i32 16777356, metadata !8, i32 0, i32 0} ; [] [data] []
!14 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!15 = metadata !{i32 786454, metadata !16, null, metadata !"", i32 183, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [] [INT32] [] [from long int]
!16 = metadata !{metadata !"", metadata !""}
!17 = metadata !{i32 786468, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [] [long int] []
!18 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!19 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!20 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!21 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!22 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!23 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [] [] []
!24 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 142, metadata !15, i32 0, i32 0} ; [  ] [] []
!25 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 143, metadata !15, i32 0, i32 0} ; [  ] [] []
!26 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 143, metadata !15, i32 0, i32 0} ; [  ] [] []
!27 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 143, metadata !15, i32 0, i32 0} ; [  ] [] []
!28 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 143, metadata !15, i32 0, i32 0} ; [  ] [] []
!29 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 144, metadata !15, i32 0, i32 0} ; [  ] [] []
!30 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 144, metadata !15, i32 0, i32 0} ; [  ] [] []
!31 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 144, metadata !15, i32 0, i32 0} ; [  ] [] []
!32 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 144, metadata !15, i32 0, i32 0} ; [ ] [] []
!33 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 144, metadata !15, i32 0, i32 0} ; [  ] [] []
!34 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 145, metadata !8, i32 0, i32 0} ; [  ] [] []
!35 = metadata !{i32 786688, metadata !4, metadata !"", metadata !5, i32 146, metadata !11, i32 0, i32 0} ; [  ] [] []
!36 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!37 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!38 = metadata !{metadata !"clang version 3.6.0 "}
!39 = metadata !{i32 154, i32 8, metadata !40, null}
!40 = metadata !{i32 786443, metadata !1, metadata !41, i32 154, i32 8, i32 2, i32 5} ; [  ] []
!41 = metadata !{i32 786443, metadata !1, metadata !42, i32 154, i32 8, i32 1, i32 4} ; [  ] []
!42 = metadata !{i32 786443, metadata !1, metadata !4, i32 154, i32 3, i32 0, i32 0} ; [  ] []
!43 = metadata !{i32 157, i32 5, metadata !44, null}
!44 = metadata !{i32 786443, metadata !1, metadata !42, i32 154, i32 42, i32 0, i32 1} ; [  ] []
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
