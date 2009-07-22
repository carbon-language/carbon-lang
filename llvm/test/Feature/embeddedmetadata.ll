; RUN: llvm-as < %s | llvm-dis | not grep undef

declare i8 @llvm.something(metadata %a, i32 %b, metadata %c)

;; Simple MDNode
!21 = metadata !{i17 123, null, metadata !"foobar"}

define void @foo() {
  ;; Intrinsic using MDNode and MDString
  %x = call i8 @llvm.something(metadata !21, i32 42, metadata !"bar")
  ret void
}

