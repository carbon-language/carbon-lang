; RUN: llc -march=mipsel -mattr=+dsp < %s

@g1 = common global <2 x i8> zeroinitializer, align 2
@g0 = common global <2 x i8> zeroinitializer, align 2

define void @extend_load_trunc_store_v2i8() {
entry:
  %0 = load <2 x i8>, <2 x i8>* @g1, align 2
  store <2 x i8> %0, <2 x i8>* @g0, align 2
  ret void
}
