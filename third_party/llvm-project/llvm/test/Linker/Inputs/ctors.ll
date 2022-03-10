@v = weak global i8 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @f, i8* @v}]

define weak void @f() {
  ret void
}
