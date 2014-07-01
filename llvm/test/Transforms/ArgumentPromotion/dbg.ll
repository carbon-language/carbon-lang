; RUN: opt < %s -argpromotion -S | FileCheck %s
; CHECK: call void @test(), !dbg [[DBG_LOC:![0-9]]]
; CHECK: [[TEST_FN:.*]] = {{.*}} void ()* @test
; CHECK: [[DBG_LOC]] = metadata !{i32 8, i32 0, metadata [[TEST_FN]], null}

define internal void @test(i32* %X) {
  ret void
}

define void @caller() {
  call void @test(i32* null), !dbg !1
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!3}

!0 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!1 = metadata !{i32 8, i32 0, metadata !2, null}
!2 = metadata !{i32 786478, null, null, metadata !"test", metadata !"test", metadata !"", i32 3, null, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32*)* @test, null, null, null, i32 3}
!3 = metadata !{i32 786449, null, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, null, null, metadata !4, null, null, metadata !"", i32 2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/pr20038/reduce/<stdin>] [DW_LANG_C_plus_plus]
!4 = metadata !{metadata !2}
