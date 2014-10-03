; RUN: llvm-as < %s | llvm-dis | grep " !dbg " | count 4
define i32 @foo() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  call void @llvm.dbg.func.start(metadata !0)
  store i32 42, i32* %retval, !dbg !3
  br label %0, !dbg !3

; <label>:0                                       ; preds = %entry
  call void @llvm.dbg.region.end(metadata !0)
  %1 = load i32* %retval, !dbg !3                  ; <i32> [#uses=1]
  ret i32 %1, !dbg !3
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!llvm.module.flags = !{!6}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00foo\001\000\001\000\006\000\000\000", i32 0, metadata !1, metadata !2, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x11\0012\00clang 1.0\001\00\000\00\000", metadata !4, metadata !5, metadata !5, metadata !4, null, null} ; [ DW_TAG_compile_unit ]
!2 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !1} ; [ DW_TAG_base_type ]
!3 = metadata !{i32 1, i32 13, metadata !1, metadata !1}
!4 = metadata !{metadata !"foo.c", metadata !"/tmp"}
!5 = metadata !{i32 0}
!6 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
