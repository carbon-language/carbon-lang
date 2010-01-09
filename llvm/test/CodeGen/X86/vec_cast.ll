; RUN: llc < %s -march=x86-64 
; RUN: llc < %s -march=x86-64 -disable-mmx

define <8 x i32> @a(<8 x i16> %a) nounwind {
  %c = sext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %c
}

define <3 x i32> @b(<3 x i16> %a) nounwind {
  %c = sext <3 x i16> %a to <3 x i32>
  ret <3 x i32> %c
}

define <1 x i32> @c(<1 x i16> %a) nounwind {
  %c = sext <1 x i16> %a to <1 x i32>
  ret <1 x i32> %c
}

define <8 x i32> @d(<8 x i16> %a) nounwind {
  %c = zext <8 x i16> %a to <8 x i32>
  ret <8 x i32> %c
}

define <3 x i32> @e(<3 x i16> %a) nounwind {
  %c = zext <3 x i16> %a to <3 x i32>
  ret <3 x i32> %c
}

define <1 x i32> @f(<1 x i16> %a) nounwind {
  %c = zext <1 x i16> %a to <1 x i32>
  ret <1 x i32> %c
}

; TODO: Legalize doesn't yet handle this.
;define <8 x i16> @g(<8 x i32> %a) nounwind {
;  %c = trunc <8 x i32> %a to <8 x i16>
;  ret <8 x i16> %c
;}

define <3 x i16> @h(<3 x i32> %a) nounwind {
  %c = trunc <3 x i32> %a to <3 x i16>
  ret <3 x i16> %c
}

define <1 x i16> @i(<1 x i32> %a) nounwind {
  %c = trunc <1 x i32> %a to <1 x i16>
  ret <1 x i16> %c
}
