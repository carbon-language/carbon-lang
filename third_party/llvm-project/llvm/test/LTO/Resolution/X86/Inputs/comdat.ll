target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$c2 = comdat any
$c1 = comdat any

; This is only present in this file. The linker will keep $c1 from the first
; file and this will be undefined.
@will_be_undefined = global i32 1, comdat($c1)

@v1 = weak_odr global i32 41, comdat($c2)
define weak_odr protected i32 @f1(i8* %this) comdat($c2) {
bb20:
  store i8* %this, i8** null
  br label %bb21
bb21:
  ret i32 41
}

@r21 = global i32* @v1
@r22 = global i32(i8*)* @f1

@a21 = alias i32, i32* @v1
@a22 = alias i16, bitcast (i32* @v1 to i16*)

@a23 = alias i32(i8*), i32(i8*)* @f1
@a24 = alias i16, bitcast (i32(i8*)* @f1 to i16*)
@a25 = alias i16, i16* @a24
