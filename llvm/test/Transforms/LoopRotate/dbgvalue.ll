; RUN: opt -S -loop-rotate < %s | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @tak(i32 %x, i32 %y, i32 %z) nounwind ssp {
; CHECK-LABEL: define i32 @tak(
; CHECK: entry
; CHECK-NEXT: call void @llvm.dbg.value(metadata !{i32 %x}

entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %if.then, %entry
  %x.tr = phi i32 [ %x, %entry ], [ %call, %if.then ]
  %y.tr = phi i32 [ %y, %entry ], [ %call9, %if.then ]
  %z.tr = phi i32 [ %z, %entry ], [ %call14, %if.then ]
  tail call void @llvm.dbg.value(metadata !{i32 %x.tr}, i64 0, metadata !6, metadata !{}), !dbg !7
  tail call void @llvm.dbg.value(metadata !{i32 %y.tr}, i64 0, metadata !8, metadata !{}), !dbg !9
  tail call void @llvm.dbg.value(metadata !{i32 %z.tr}, i64 0, metadata !10, metadata !{}), !dbg !11
  %cmp = icmp slt i32 %y.tr, %x.tr, !dbg !12
  br i1 %cmp, label %if.then, label %if.end, !dbg !12

if.then:                                          ; preds = %tailrecurse
  %sub = sub nsw i32 %x.tr, 1, !dbg !14
  %call = tail call i32 @tak(i32 %sub, i32 %y.tr, i32 %z.tr), !dbg !14
  %sub6 = sub nsw i32 %y.tr, 1, !dbg !14
  %call9 = tail call i32 @tak(i32 %sub6, i32 %z.tr, i32 %x.tr), !dbg !14
  %sub11 = sub nsw i32 %z.tr, 1, !dbg !14
  %call14 = tail call i32 @tak(i32 %sub11, i32 %x.tr, i32 %y.tr), !dbg !14
  br label %tailrecurse

if.end:                                           ; preds = %tailrecurse
  br label %return, !dbg !16

return:                                           ; preds = %if.end
  ret i32 %z.tr, !dbg !17
}

@channelColumns = external global i64
@horzPlane = external global i8*, align 8

define void @FindFreeHorzSeg(i64 %startCol, i64 %row, i64* %rowStart) {
; Ensure that the loop increment basic block is rotated into the tail of the
; body, even though it contains a debug intrinsic call.
; CHECK-LABEL: define void @FindFreeHorzSeg(
; CHECK: %dec = add
; CHECK-NEXT: tail call void @llvm.dbg.value
; CHECK: %cmp = icmp
; CHECK: br i1 %cmp
; CHECK: phi i64 [ %{{[^,]*}}, %{{[^,]*}} ]
; CHECK-NEXT: br label %for.end


entry:
  br label %for.cond

for.cond:
  %i.0 = phi i64 [ %startCol, %entry ], [ %dec, %for.inc ]
  %cmp = icmp eq i64 %i.0, 0
  br i1 %cmp, label %for.end, label %for.body

for.body:
  %0 = load i64* @channelColumns, align 8
  %mul = mul i64 %0, %row
  %add = add i64 %mul, %i.0
  %1 = load i8** @horzPlane, align 8
  %arrayidx = getelementptr inbounds i8* %1, i64 %add
  %2 = load i8* %arrayidx, align 1
  %tobool = icmp eq i8 %2, 0
  br i1 %tobool, label %for.inc, label %for.end

for.inc:
  %dec = add i64 %i.0, -1
  tail call void @llvm.dbg.value(metadata !{i64 %dec}, i64 0, metadata !{metadata !"undef"}, metadata !{})
  br label %for.cond

for.end:
  %add1 = add i64 %i.0, 1
  store i64 %add1, i64* %rowStart, align 8
  ret void
}

!llvm.module.flags = !{!20}
!llvm.dbg.sp = !{!0}

!0 = metadata !{metadata !"0x2e\00tak\00tak\00\0032\000\001\000\006\00256\000\000", metadata !18, metadata !1, metadata !3, null, i32 (i32, i32, i32)* @tak, null, null, null} ; [ DW_TAG_subprogram ] [line 32] [def] [scope 0] [tak]
!1 = metadata !{metadata !"0x29", metadata !18} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 125492)\001\00\000\00\000", metadata !18, metadata !19, metadata !19, null, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !18, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x101\00x\0032\000", metadata !0, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!7 = metadata !{i32 32, i32 13, metadata !0, null}
!8 = metadata !{metadata !"0x101\00y\0032\000", metadata !0, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 32, i32 20, metadata !0, null}
!10 = metadata !{metadata !"0x101\00z\0032\000", metadata !0, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 32, i32 27, metadata !0, null}
!12 = metadata !{i32 33, i32 3, metadata !13, null}
!13 = metadata !{metadata !"0xb\0032\0030\006", metadata !18, metadata !0} ; [ DW_TAG_lexical_block ]
!14 = metadata !{i32 34, i32 5, metadata !15, null}
!15 = metadata !{metadata !"0xb\0033\0014\007", metadata !18, metadata !13} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 36, i32 3, metadata !13, null}
!17 = metadata !{i32 37, i32 1, metadata !13, null}
!18 = metadata !{metadata !"/Volumes/Lalgate/cj/llvm/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame/recursive.c", metadata !"/Volumes/Lalgate/cj/D/projects/llvm-test/SingleSource/Benchmarks/BenchmarkGame"}
!19 = metadata !{i32 0}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
