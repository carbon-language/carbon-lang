; RUN: llvm-as < %s | llvm-dis | not grep undef

declare i8 @llvm.something(metadata %a)

@llvm.foo = internal constant metadata !{i17 123, null, metadata !"foobar"}

define void @foo() {
  %x = call i8 @llvm.something(metadata !{metadata !"f\00oa", i42 123})
  ret void
}

