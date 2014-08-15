$c = comdat any
@a = alias void ()* @f
define internal void @f() comdat $c {
  ret void
}
