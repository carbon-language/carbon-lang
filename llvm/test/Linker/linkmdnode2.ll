; This file is used by linkmdnode.ll, so it doesn't actually do anything itself
;
; RUN: true

!22 = metadata !{i32 42, metadata !"foobar"}

declare i8 @llvm.something(metadata %a)
define void @foo1() {
  ;; Intrinsic using MDNode and MDString
  %x = call i8 @llvm.something(metadata !22)
  ret void
}
