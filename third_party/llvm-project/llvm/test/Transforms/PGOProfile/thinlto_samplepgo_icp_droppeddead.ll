; REQUIRES: x86-registered-target

; RUN: opt -module-summary %s -o %t.bc

; Tests that with dead stripping in the thin link enabled (default), we do not
; promote to target of the dropped dead symbol _ZL3foov. This can happen with a
; sample profile collected for one binary used to  optimize for another binary.
; RUN: llvm-lto2 run -save-temps -o %t2 %t.bc -r %t.bc,fptr,plx \
; RUN:		-r %t.bc,main,plx -r %t.bc,_ZL3foov,l
; RUN: llvm-dis < %t2.1.4.opt.bc | FileCheck %s --check-prefix=OPT
; RUN: llvm-lto2 run -save-temps -o %t2 %t.bc -r %t.bc,fptr,plx \
; RUN: 		-r %t.bc,main,plx -r %t.bc,_ZL3foov,l -compute-dead=false
; RUN: llvm-dis < %t2.1.4.opt.bc | FileCheck %s --check-prefix=OPT-NODEAD

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptr = local_unnamed_addr global void ()* null, align 8

define void @_ZL3foov() #1 {
entry:
  ret void
}

define i32 @main() local_unnamed_addr #0 !prof !34 {
entry:
  %0 = load void ()*, void ()** @fptr, align 8
; OPT-NOT: label %if.false.orig_indirect
; OPT-NODEAD: br i1 %{{[0-9]+}}, label %if.end.icp, label %if.false.orig_indirect
  tail call void %0(), !prof !40
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3,!4}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 297016)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "main.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"ProfileSummary", !5}
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13}
!6 = !{!"ProfileFormat", !"SampleProfile"}
!7 = !{!"TotalCount", i64 3003}
!8 = !{!"MaxCount", i64 3000}
!9 = !{!"MaxInternalCount", i64 0}
!10 = !{!"MaxFunctionCount", i64 0}
!11 = !{!"NumCounts", i64 3}
!12 = !{!"NumFunctions", i64 1}
!13 = !{!"DetailedSummary", !14}
!14 = !{!15, !16, !17, !18, !19, !20, !20, !21, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30}
!15 = !{i32 10000, i64 3000, i32 1}
!16 = !{i32 100000, i64 3000, i32 1}
!17 = !{i32 200000, i64 3000, i32 1}
!18 = !{i32 300000, i64 3000, i32 1}
!19 = !{i32 400000, i64 3000, i32 1}
!20 = !{i32 500000, i64 3000, i32 1}
!21 = !{i32 600000, i64 3000, i32 1}
!22 = !{i32 700000, i64 3000, i32 1}
!23 = !{i32 800000, i64 3000, i32 1}
!24 = !{i32 900000, i64 3000, i32 1}
!25 = !{i32 950000, i64 3000, i32 1}
!26 = !{i32 990000, i64 3000, i32 1}
!27 = !{i32 999000, i64 3000, i32 1}
!28 = !{i32 999900, i64 2, i32 2}
!29 = !{i32 999990, i64 2, i32 2}
!30 = !{i32 999999, i64 2, i32 2}
!31 = !{!"clang version 5.0.0 (trunk 297016)"}
!34 = !{!"function_entry_count", i64 1}
!40 = !{!"VP", i32 0, i64 3000, i64 -8789629626369651636, i64 3000}
