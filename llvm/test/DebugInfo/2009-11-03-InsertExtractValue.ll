; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!0 = metadata !{i32 42}

define <{i32, i32}> @f1() {
; CHECK: !dbg !0
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbg !0
; CHECK: !dbg !0
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbg !0
  ret <{ i32, i32 }> %r
}
