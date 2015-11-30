$foo = comdat any
@foo = global i8 1, comdat
define void @zed() {
  call void @bar()
  ret void
}
define internal void @bar() comdat($foo) {
  ret void
}
