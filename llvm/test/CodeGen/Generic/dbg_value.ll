; RUN: llc < %s
; rdar://7759395

%0 = type { i32, i32 }

define void @t(%0*, i32, i32, i32, i32) nounwind {
  tail call void @llvm.dbg.value(metadata !{%0* %0}, i64 0, metadata !0)
  unreachable
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

; !0 should conform to the format of DIVariable.
!0 = metadata !{i32 786689, null, metadata !"a", null, i32 0, null, i32 0, i32 0} ;
