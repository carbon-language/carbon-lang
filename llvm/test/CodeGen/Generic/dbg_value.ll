; RUN: llc < %s
; rdar://7759395

%0 = type { i32, i32 }

define void @t(%0*, i32, i32, i32, i32) nounwind {
  tail call void @llvm.dbg.value(metadata !{%0* %0}, i64 0, metadata !0, metadata !{metadata !"0x102"})
  unreachable
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

; !0 should conform to the format of DIVariable.
!0 = metadata !{metadata !"0x101\00a\000\000", null, null, null} ; [ DW_TAG_arg_variable ]
