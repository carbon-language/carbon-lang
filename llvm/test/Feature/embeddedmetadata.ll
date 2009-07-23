; RUN: llvm-as < %s | llvm-dis | not grep undef

declare i8 @llvm.something(metadata %a, i32 %b, metadata %c)

;; Simple MDNode
!21 = metadata !{i17 123, null, metadata !"foobar"}

define void @foo() {
  ;; Intrinsic using MDNode and MDString
  %x = call i8 @llvm.something(metadata !21, i32 42, metadata !"bar")
  ret void
}

;; Test forward reference
declare i8 @llvm.f2(metadata %a)
define void @f2() {
  %x = call i8 @llvm.f2(metadata !2)
  ret void
}
!2 = metadata !{i32 420}
