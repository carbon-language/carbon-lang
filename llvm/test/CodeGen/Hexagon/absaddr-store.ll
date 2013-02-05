; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we generate load instructions with absolute addressing mode.

@a = external global i32
@b = external global i8
@c = external global i16
@d = external global i64

define zeroext i8 @absStoreByte() nounwind {
; CHECK: memb(##b){{ *}}={{ *}}r{{[0-9]+}}
entry:
  %0 = load i8* @b, align 1
  %conv = zext i8 %0 to i32
  %mul = mul nsw i32 100, %conv
  %conv1 = trunc i32 %mul to i8
  store i8 %conv1, i8* @b, align 1
  ret i8 %conv1
}

define signext i16 @absStoreHalf() nounwind {
; CHECK: memh(##c){{ *}}={{ *}}r{{[0-9]+}}
entry:
  %0 = load i16* @c, align 2
  %conv = sext i16 %0 to i32
  %mul = mul nsw i32 100, %conv
  %conv1 = trunc i32 %mul to i16
  store i16 %conv1, i16* @c, align 2
  ret i16 %conv1
}

define i32 @absStoreWord() nounwind {
; CHECK: memw(##a){{ *}}={{ *}}r{{[0-9]+}}
entry:
  %0 = load i32* @a, align 4
  %mul = mul nsw i32 100, %0
  store i32 %mul, i32* @a, align 4
  ret i32 %mul
}

define void @absStoreDouble() nounwind {
; CHECK: memd(##d){{ *}}={{ *}}r{{[0-9]+}}:{{[0-9]+}}
entry:
  store i64 100, i64* @d, align 8
  ret void
}

