$foo = comdat any
define linkonce_odr i32 @f1() comdat($foo) {
  ret i32 1
}
define void @f2() comdat($foo) {
  ret void
}
