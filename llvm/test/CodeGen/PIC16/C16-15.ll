; RUN: llc < %s -march=pic16 | grep "extern" | grep "@.lib.unordered.f32" | count 3

@pc = global i8* inttoptr (i64 160 to i8*), align 1 ; <i8**> [#uses=2]
@aa = common global i16 0, align 1                ; <i16*> [#uses=0]
@c6214.auto.d = internal global float 0.000000e+00, align 4 ; <float*> [#uses=1]
@c6214.auto.l = internal global float 0.000000e+00, align 4 ; <float*> [#uses=1]

define float @dvalue(float %f) nounwind {
entry:
  ret float %f
}

define void @_assert(i16 %line, i16 %result) nounwind {
entry:
  %add = add i16 %line, %result                   ; <i16> [#uses=1]
  %conv = trunc i16 %add to i8                    ; <i8> [#uses=1]
  %tmp2 = load i8** @pc                           ; <i8*> [#uses=1]
  store i8 %conv, i8* %tmp2
  ret void
}

define i16 @main() nounwind {
entry:
  %retval = alloca i16, align 1                   ; <i16*> [#uses=2]
  store i16 0, i16* %retval
  call void @c6214()
  %0 = load i16* %retval                          ; <i16> [#uses=1]
  ret i16 %0
}

define internal void @c6214() nounwind {
entry:
  %call = call float @dvalue(float 0x3FF3C0CA40000000) ; <float> [#uses=3]
  store float %call, float* @c6214.auto.d
  store float %call, float* @c6214.auto.l
  %cmp = fcmp ord float %call, 0.000000e+00       ; <i1> [#uses=1]
  %conv = zext i1 %cmp to i16                     ; <i16> [#uses=1]
  call void @_assert(i16 10, i16 %conv)
  %tmp3 = load i8** @pc                           ; <i8*> [#uses=2]
  %tmp4 = load i8* %tmp3                          ; <i8> [#uses=1]
  %sub = add i8 %tmp4, -10                        ; <i8> [#uses=1]
  store i8 %sub, i8* %tmp3
  ret void
}
