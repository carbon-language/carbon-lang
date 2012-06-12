; RUN: opt -reassociate -disable-output %s


; rdar://7507855
define fastcc i32 @test1() nounwind {
entry:
  %cond = select i1 undef, i32 1, i32 -1          ; <i32> [#uses=2]
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %sub889 = sub i32 undef, undef                  ; <i32> [#uses=1]
  %sub891 = sub i32 %sub889, %cond                ; <i32> [#uses=0]
  %add896 = sub i32 0, %cond                      ; <i32> [#uses=0]
  ret i32 undef
}

; PR5981
define i32 @test2() nounwind ssp {
entry:
  %0 = load i32* undef, align 4
  %1 = mul nsw i32 undef, %0
  %2 = mul nsw i32 undef, %0
  %3 = add nsw i32 undef, %1
  %4 = add nsw i32 %3, %2
  %5 = add nsw i32 %4, 4
  %6 = shl i32 %0, 3
  %7 = add nsw i32 %5, %6
  br label %bb4.i9

bb4.i9:
  %8 = add nsw i32 undef, %1
  ret i32 0
}


define i32 @test3(i32 %Arg, i32 %x1, i32 %x2, i32 %x3) {
 %A = mul i32 %x1, %Arg
 %B = mul i32 %Arg, %x2 ;; Part of add operation being factored, also used by C
 %C = mul i32 %x3, %B

 %D = add i32 %A, %B
 %E = add i32 %D, %C
  ret i32 %E
}


; rdar://9096268
define void @x66303361ae3f602889d1b7d0f86e5455(i8* %arg) nounwind {
_:
  br label %_33

_33:                                              ; preds = %_33, %_
  %tmp348 = load i8* %arg, align 1
  %tmp349 = lshr i8 %tmp348, 7
  %tmp350 = or i8 %tmp349, 42
  %tmp351 = add i8 %tmp350, -42
  %tmp352 = zext i8 %tmp351 to i32
  %tmp358 = add i32 %tmp352, -501049439
  %tmp359 = mul i32 %tmp358, %tmp358
  %tmp360 = mul i32 %tmp352, %tmp352
  %tmp361 = sub i32 %tmp359, %tmp360
  %tmp362 = mul i32 %tmp361, -920056735
  %tmp363 = add i32 %tmp362, 501049439
  %tmp364 = add i32 %tmp362, -2000262972
  %tmp365 = sub i32 %tmp363, %tmp364
  %tmp366 = sub i32 -501049439, %tmp362
  %tmp367 = add i32 %tmp365, %tmp366
  br label %_33
}

define void @test(i32 %a, i32 %b, i32 %c, i32 %d) {
  %tmp.2 = xor i32 %a, %b		; <i32> [#uses=1]
  %tmp.5 = xor i32 %c, %d		; <i32> [#uses=1]
  %tmp.6 = xor i32 %tmp.2, %tmp.5		; <i32> [#uses=1]
  %tmp.9 = xor i32 %c, %a		; <i32> [#uses=1]
  %tmp.12 = xor i32 %b, %d		; <i32> [#uses=1]
  %tmp.13 = xor i32 %tmp.9, %tmp.12		; <i32> [#uses=1]
  %tmp.16 = xor i32 %tmp.6, %tmp.13		; <i32> [#uses=0]
  ret void
}

define i128 @foo() {
  %mul = mul i128 0, 0
  ret i128 %mul
}
