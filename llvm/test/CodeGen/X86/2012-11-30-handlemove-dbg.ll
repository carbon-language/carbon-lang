; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test LiveInterval update handling of DBG_VALUE.
; rdar://12777252.
;
; CHECK: %entry
; CHECK: DEBUG_VALUE: hg
; CHECK: j

%struct.node.0.27 = type { i16, double, [3 x double], i32, i32 }
%struct.hgstruct.2.29 = type { %struct.bnode.1.28*, [3 x double], double, [3 x double] }
%struct.bnode.1.28 = type { i16, double, [3 x double], i32, i32, [3 x double], [3 x double], [3 x double], double, %struct.bnode.1.28*, %struct.bnode.1.28* }

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define signext i16 @subdivp(%struct.node.0.27* nocapture %p, double %dsq, double %tolsq, %struct.hgstruct.2.29* nocapture byval align 8 %hg) nounwind uwtable readonly ssp {
entry:
  call void @llvm.dbg.declare(metadata %struct.hgstruct.2.29* %hg, metadata !4, metadata !{!"0x102"})
  %type = getelementptr inbounds %struct.node.0.27, %struct.node.0.27* %p, i64 0, i32 0
  %0 = load i16, i16* %type, align 2
  %cmp = icmp eq i16 %0, 1
  br i1 %cmp, label %return, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  %arrayidx6.1 = getelementptr inbounds %struct.hgstruct.2.29, %struct.hgstruct.2.29* %hg, i64 0, i32 1, i64 1
  %cmp22 = fcmp olt double 0.000000e+00, %dsq
  %conv24 = zext i1 %cmp22 to i16
  br label %return

return:                                           ; preds = %for.cond.preheader, %entry
  %retval.0 = phi i16 [ %conv24, %for.cond.preheader ], [ 0, %entry ]
  ret i16 %retval.0
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = !{!"0x11\0012\00clang version 3.3 (trunk 168918) (llvm/trunk 168920)\001\00\000\00\000", !11, !2, !2, !13, !2, null} ; [ DW_TAG_compile_unit ] [MultiSource/Benchmarks/Olden/bh/newbh.c] [DW_LANG_C99]
!2 = !{}
!4 = !{!"0x101\00hg\0067109589\000", null, !5, !6} ; [ DW_TAG_arg_variable ] [hg] [line 725]
!5 = !{!"0x29", !11} ; [ DW_TAG_file_type ]
!6 = !{!"0x16\00hgstruct\00492\000\000\000\000", !11, null, !7} ; [ DW_TAG_typedef ] [hgstruct] [line 492, size 0, align 0, offset 0] [from ]
!7 = !{!"0x13\00\00487\00512\0064\000\000\000", !11, null, null, null, null, i32 0, null} ; [ DW_TAG_structure_type ] [line 487, size 512, align 64, offset 0] [def] [from ]
!11 = !{!"MultiSource/Benchmarks/Olden/bh/newbh.c", !"MultiSource/Benchmarks/Olden/bh"}
!12 = !{i32 1, !"Debug Info Version", i32 2}
!13 = !{!14}
!14 = !{!"0x2e\00subdivp\00subdivp\00\000\000\001\000\006\00256\001\001", !11, !5, !15, null, i16 (%struct.node.0.27*, double, double, %struct.hgstruct.2.29* )* @subdivp, null, null, null} ; [ DW_TAG_subprogram ] [def] [subdivp]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{null}
