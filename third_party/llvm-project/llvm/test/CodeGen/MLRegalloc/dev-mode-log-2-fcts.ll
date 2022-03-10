; REQUIRES: have_tf_api
; REQUIRES: x86_64-linux
;
; Check that we can log more than 1 function.
;
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t1 -tfutils-text-log < %s
; RUN: sed -i 's/ \+/ /g' %t1
; RUN: sed -i 's/\\n key:/\n key:/g' %t1
; RUN: sed -i 's/\\n feature/\n feature/g' %t1
; RUN: sed -i 's/\\n/ /g' %t1
; RUN: FileCheck --input-file %t1 %s

; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-eviction-test-model.py %t
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t2 -tfutils-text-log -regalloc-model=%t < %s
; RUN: sed -i 's/ \+/ /g' %t2
; RUN: sed -i 's/\\n key:/\n key:/g' %t2
; RUN: sed -i 's/\\n feature/\n feature/g' %t2
; RUN: sed -i 's/\\n/ /g' %t2
; RUN: FileCheck --input-file %t2 %s

declare void @f();

define void @f1(i64 %lhs, i64 %rhs, i64* %addr) !prof !15 {
  %sum = add i64 %lhs, %rhs
  call void @f();
  store i64 %sum, i64* %addr
  ret void
}

define void @f2(i64 %lhs, i64 %rhs, i64* %addr) !prof !16 {
  %sum = add i64 %lhs, %rhs
  store i64 %sum, i64* %addr
  ret void
}

; CHECK:  key: "f1"
; CHECK:  key: "f2"

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"function_entry_count", i64 1000}
