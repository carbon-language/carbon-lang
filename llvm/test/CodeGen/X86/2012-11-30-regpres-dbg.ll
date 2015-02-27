; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test RegisterPressure handling of DBG_VALUE.
;
; CHECK: %entry
; CHECK: DEBUG_VALUE: callback
; CHECK: ret

%struct.btCompoundLeafCallback = type { i32, i32 }

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define void @test() unnamed_addr uwtable ssp align 2 {
entry:
  %callback = alloca %struct.btCompoundLeafCallback, align 8
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata %struct.btCompoundLeafCallback* %callback, metadata !3, metadata !{!"0x102"})
  %m = getelementptr inbounds %struct.btCompoundLeafCallback, %struct.btCompoundLeafCallback* %callback, i64 0, i32 1
  store i32 0, i32* undef, align 8
  %cmp12447 = icmp sgt i32 undef, 0
  br i1 %cmp12447, label %for.body.lr.ph, label %invoke.cont44

for.body.lr.ph:                                   ; preds = %if.end
  unreachable

invoke.cont44:                                    ; preds = %if.end
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 168984) (llvm/trunk 168983)\001\00\000\00\000", !6, null, null, !1, null, null} ; [ DW_TAG_compile_unit ] [MultiSource/Benchmarks/Bullet/MultiSource/Benchmarks/Bullet/btCompoundCollisionAlgorithm.cpp] [DW_LANG_C_plus_plus]
!1 = !{!2}
!2 = !{!"0x2e\00test\00test\00\000\000\001\000\006\00256\001\001", !6, !5, !7, null, void ()* @test, null, null, null} ; [ DW_TAG_subprogram ] [def] [test]
!3 = !{!"0x100\00callback\00214\000", null, null, !4} ; [ DW_TAG_auto_variable ] [callback] [line 214]
!4 = !{!"0x13\00btCompoundLeafCallback\0090\00512\0064\000\000\000", !6, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [btCompoundLeafCallback] [line 90, size 512, align 64, offset 0] [def] [from ]
!5 = !{!"0x29", !6} ; [ DW_TAG_file_type ]
!6 = !{!"MultiSource/Benchmarks/Bullet/btCompoundCollisionAlgorithm.cpp", !"MultiSource/Benchmarks/Bullet"}
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{i32 1, !"Debug Info Version", i32 2}
!9 = !{null}
