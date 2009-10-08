; RUN: llc < %s

%struct.TConstantDictionary = type { %struct.__CFDictionary* }
%struct.TSharedGlobalSet_AS = type { [52 x i32], [20 x i32], [22 x i32], [8 x i32], [20 x i32], [146 x i32] }
%struct.__CFDictionary = type opaque

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ()* @func to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define void @func() ssp {
entry:
  tail call void @llvm.dbg.func.start(metadata !13)
  tail call void @llvm.dbg.stoppoint(i32 1001, i32 0, metadata !1)
  %0 = tail call %struct.TSharedGlobalSet_AS* @g1() nounwind ; <%struct.TSharedGlobalSet_AS*> [#uses=1]
  %1 = getelementptr inbounds %struct.TSharedGlobalSet_AS* %0, i32 0, i32 4, i32 4 ; <i32*> [#uses=1]
  %2 = bitcast i32* %1 to %struct.TConstantDictionary* ; <%struct.TConstantDictionary*> [#uses=1]
  tail call void @g2(%struct.TConstantDictionary* %2) ssp
  tail call void @llvm.dbg.stoppoint(i32 1002, i32 0, metadata !1)
  %3 = tail call %struct.TSharedGlobalSet_AS* @g1() nounwind ; <%struct.TSharedGlobalSet_AS*> [#uses=1]
  %4 = getelementptr inbounds %struct.TSharedGlobalSet_AS* %3, i32 0, i32 4, i32 3 ; <i32*> [#uses=1]
  %5 = bitcast i32* %4 to %struct.TConstantDictionary* ; <%struct.TConstantDictionary*> [#uses=1]
  tail call void @g4(%struct.TConstantDictionary* %5) ssp
  tail call void @llvm.dbg.stoppoint(i32 1003, i32 0, metadata !1)
  %6 = tail call %struct.TSharedGlobalSet_AS* @g1() nounwind ; <%struct.TSharedGlobalSet_AS*> [#uses=1]
  %7 = getelementptr inbounds %struct.TSharedGlobalSet_AS* %6, i32 0, i32 4, i32 2 ; <i32*> [#uses=1]
  %8 = bitcast i32* %7 to %struct.TConstantDictionary* ; <%struct.TConstantDictionary*> [#uses=1]
  tail call void @g3(%struct.TConstantDictionary* %8) ssp
  tail call void @llvm.dbg.stoppoint(i32 1004, i32 0, metadata !1)
  %9 = tail call %struct.TSharedGlobalSet_AS* @g1() nounwind ; <%struct.TSharedGlobalSet_AS*> [#uses=1]
  %10 = getelementptr inbounds %struct.TSharedGlobalSet_AS* %9, i32 0, i32 4, i32 1 ; <i32*> [#uses=1]
  %11 = bitcast i32* %10 to %struct.TConstantDictionary* ; <%struct.TConstantDictionary*> [#uses=1]
  tail call void @g4(%struct.TConstantDictionary* %11) ssp
  tail call void @llvm.dbg.stoppoint(i32 1005, i32 0, metadata !1)
  tail call void @g5()
  tail call void @llvm.dbg.stoppoint(i32 1006, i32 0, metadata !1)
  tail call void @llvm.dbg.region.end(metadata !13)
  ret void
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

declare %struct.TSharedGlobalSet_AS* @g1() nounwind readonly ssp

declare void @g2(%struct.TConstantDictionary* nocapture) ssp align 2

declare void @g3(%struct.TConstantDictionary* nocapture) ssp align 2

declare void @g4(%struct.TConstantDictionary* nocapture) ssp align 2

declare void @g5()

!llvm.dbg.gv = !{!0, !9, !10, !11, !12}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"_ZZ7UASInitmmmmmmmmmE5C.408", metadata !"C.408", metadata !"_ZZ7UASInitmmmmmmmmmE5C.408", metadata !1, i32 874, metadata !2, i1 false, i1 true, null}; [DW_TAG_variable ]
!1 = metadata !{i32 458769, i32 0, i32 4, metadata !"func.cp", metadata !"/tmp/func", metadata !"4.2.1 (Based on Apple Inc. build 5653) (LLVM build 2311)", i1 false, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!2 = metadata !{i32 458753, metadata !3, metadata !"", metadata !3, i32 0, i64 16, i64 16, i64 0, i32 0, metadata !4, metadata !7, i32 0}; [DW_TAG_array_type ]
!3 = metadata !{i32 458769, i32 0, i32 4, metadata !"testcase.ii", metadata !"/tmp/", metadata !"4.2.1 (Based on Apple Inc. build 5653) (LLVM build 2311)", i1 true, i1 false, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!4 = metadata !{i32 458774, metadata !3, metadata !"UniChar", metadata !5, i32 417, i64 0, i64 0, i64 0, i32 0, metadata !6}; [DW_TAG_typedef ]
!5 = metadata !{i32 458769, i32 0, i32 4, metadata !"MacTypes.h", metadata !"/System/Library/Frameworks/CoreServices.framework/Headers/../Frameworks/CarbonCore.framework/Headers", metadata !"4.2.1 (Based on Apple Inc. build 5653) (LLVM build 2311)", i1 false, i1 true, metadata !"", i32 0}; [DW_TAG_compile_unit ]
!6 = metadata !{i32 458788, metadata !3, metadata !"short unsigned int", metadata !3, i32 0, i64 16, i64 16, i64 0, i32 0, i32 7}; [DW_TAG_base_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 458785, i64 0, i64 0}; [DW_TAG_subrange_type ]
!9 = metadata !{i32 458804, i32 0, metadata !1, metadata !"_ZZ7UASInitmmmmmmmmmE5C.409", metadata !"C.409", metadata !"_ZZ7UASInitmmmmmmmmmE5C.409", metadata !1, i32 877, metadata !2, i1 false, i1 true, null}; [DW_TAG_variable ]
!10 = metadata !{i32 458804, i32 0, metadata !1, metadata !"_ZZ7UASInitmmmmmmmmmE5C.410", metadata !"C.410", metadata !"_ZZ7UASInitmmmmmmmmmE5C.410", metadata !1, i32 880, metadata !2, i1 false, i1 true, null}; [DW_TAG_variable ]
!11 = metadata !{i32 458804, i32 0, metadata !1, metadata !"_ZZ7UASInitmmmmmmmmmE5C.411", metadata !"C.411", metadata !"_ZZ7UASInitmmmmmmmmmE5C.411", metadata !1, i32 924, metadata !2, i1 false, i1 true, null}; [DW_TAG_variable ]
!12 = metadata !{i32 458804, i32 0, metadata !1, metadata !"_ZZ7UASInitmmmmmmmmmE5C.412", metadata !"C.412", metadata !"_ZZ7UASInitmmmmmmmmmE5C.412", metadata !1, i32 928, metadata !2, i1 false, i1 true, null}; [DW_TAG_variable ]
!13 = metadata !{i32 458798, i32 0, metadata !3, metadata !"UASShutdown", metadata !"UASShutdown", metadata !"_Z11UASShutdownv", metadata !1, i32 999, metadata !14, i1 false, i1 true}; [DW_TAG_subprogram ]
!14 = metadata !{i32 458773, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0}; [DW_TAG_subroutine_type ]
!15 = metadata !{null}
