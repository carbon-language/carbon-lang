; RUN: llc %s -O0 -o /dev/null -mtriple=x86_64-apple-darwin
; PR 5197
; There is not any llvm instruction assocated with !5. The code generator
; should be able to handle this.

define void @bar() nounwind ssp {
entry:
  %count_ = alloca i32, align 4                   ; <i32*> [#uses=2]
  br label %do.body, !dbg !0

do.body:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata !{i32* %count_}, metadata !4, metadata !{metadata !"0x102"})
  %conv = ptrtoint i32* %count_ to i32, !dbg !0   ; <i32> [#uses=1]
  %call = call i32 @foo(i32 %conv) ssp, !dbg !0   ; <i32> [#uses=0]
  br label %do.end, !dbg !0

do.end:                                           ; preds = %do.body
  ret void, !dbg !7
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @foo(i32) ssp

!0 = metadata !{i32 5, i32 2, metadata !1, null}
!1 = metadata !{metadata !"0xb\001\001\000", null, metadata !2}; [DW_TAG_lexical_block ]
!2 = metadata !{metadata !"0x2e\00bar\00bar\00bar\004\000\001\000\006\000\000\000", i32 0, metadata !3, null, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!3 = metadata !{metadata !"0x11\0012\00clang 1.1\001\00\000\00\000", metadata !8, null, metadata !9, null, null, null}; [DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x100\00count_\005\000", metadata !5, metadata !3, metadata !6}; [ DW_TAG_auto_variable ]
!5 = metadata !{metadata !"0xb\001\001\000", null, metadata !1}; [DW_TAG_lexical_block ]
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !3}; [DW_TAG_base_type ]
!7 = metadata !{i32 6, i32 1, metadata !2, null}
!8 = metadata !{metadata !"genmodes.i", metadata !"/Users/yash/Downloads"}
!9 = metadata !{i32 0}
