$c2 = comdat any

@v1 = weak_odr global i32 41, comdat $c2
define weak_odr i32 @f1() comdat $c2 {
bb20:
  br label %bb21
bb21:
  ret i32 41
}

@r21 = global i32* @v1
@r22 = global i32()* @f1

@a21 = alias i32* @v1
@a22 = alias bitcast (i32* @v1 to i16*)

@a23 = alias i32()* @f1
@a24 = alias bitcast (i32()* @f1 to i16*)
@a25 = alias i16* @a24
