; RUN: llc < %s -march=alpha

define i1 @a(float %x) {
  %r = fcmp ult float %x, 1.0
  ret i1 %r
}
