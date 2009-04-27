; RUN: llvm-as < %s | llc
; PR2671

define void @g(<2 x i16>* %p, <2 x i8>* %q) {
  %t = load <2 x i16>* %p
  %r = trunc <2 x i16> %t to <2 x i8>
  store <2 x i8> %r, <2 x i8>* %q
  ret void
}
