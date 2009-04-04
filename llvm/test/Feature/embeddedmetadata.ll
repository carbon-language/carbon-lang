; RUN: llvm-as < %s | llvm-dis | not grep undef

declare i8 @llvm.something({ } %a)

@llvm.foo = internal constant { } !{i17 123, { } !"foobar"}

define void @foo() {
  %x = call i8 @llvm.something({ } !{{ } !"f\00oa", i42 123})
  ret void
}

