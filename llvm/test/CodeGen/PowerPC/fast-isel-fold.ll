; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

@a = global i8 1, align 1
@b = global i16 2, align 2
@c = global i32 4, align 4

define void @t1() nounwind uwtable ssp {
; ELF64: t1
  %1 = load i8* @a, align 1
  call void @foo1(i8 zeroext %1)
; ELF64: lbz
; ELF64-NOT: rldicl
; ELF64-NOT: rlwinm
  ret void
}

define void @t2() nounwind uwtable ssp {
; ELF64: t2
  %1 = load i16* @b, align 2
  call void @foo2(i16 zeroext %1)
; ELF64: lhz
; ELF64-NOT: rldicl
; ELF64-NOT: rlwinm
  ret void
}

define void @t2a() nounwind uwtable ssp {
; ELF64: t2a
  %1 = load i32* @c, align 4
  call void @foo3(i32 zeroext %1)
; ELF64: lwz
; ELF64-NOT: rldicl
; ELF64-NOT: rlwinm
  ret void
}

declare void @foo1(i8 zeroext)
declare void @foo2(i16 zeroext)
declare void @foo3(i32 zeroext)

define i32 @t3() nounwind uwtable ssp {
; ELF64: t3
  %1 = load i8* @a, align 1
  %2 = zext i8 %1 to i32
; ELF64: lbz
; ELF64-NOT: rlwinm
  ret i32 %2
}

define i32 @t4() nounwind uwtable ssp {
; ELF64: t4
  %1 = load i16* @b, align 2
  %2 = zext i16 %1 to i32
; ELF64: lhz
; ELF64-NOT: rlwinm
  ret i32 %2
}

define i32 @t5() nounwind uwtable ssp {
; ELF64: t5
  %1 = load i16* @b, align 2
  %2 = sext i16 %1 to i32
; ELF64: lha
; ELF64-NOT: rlwinm
  ret i32 %2
}

define i32 @t6() nounwind uwtable ssp {
; ELF64: t6
  %1 = load i8* @a, align 2
  %2 = sext i8 %1 to i32
; ELF64: lbz
; ELF64-NOT: rlwinm
  ret i32 %2
}

define i64 @t7() nounwind uwtable ssp {
; ELF64: t7
  %1 = load i8* @a, align 1
  %2 = zext i8 %1 to i64
; ELF64: lbz
; ELF64-NOT: rldicl
  ret i64 %2
}

define i64 @t8() nounwind uwtable ssp {
; ELF64: t8
  %1 = load i16* @b, align 2
  %2 = zext i16 %1 to i64
; ELF64: lhz
; ELF64-NOT: rldicl
  ret i64 %2
}

define i64 @t9() nounwind uwtable ssp {
; ELF64: t9
  %1 = load i16* @b, align 2
  %2 = sext i16 %1 to i64
; ELF64: lha
; ELF64-NOT: extsh
  ret i64 %2
}

define i64 @t10() nounwind uwtable ssp {
; ELF64: t10
  %1 = load i8* @a, align 2
  %2 = sext i8 %1 to i64
; ELF64: lbz
; ELF64: extsb
  ret i64 %2
}

define i64 @t11() nounwind uwtable ssp {
; ELF64: t11
  %1 = load i32* @c, align 4
  %2 = zext i32 %1 to i64
; ELF64: lwz
; ELF64-NOT: rldicl
  ret i64 %2
}

define i64 @t12() nounwind uwtable ssp {
; ELF64: t12
  %1 = load i32* @c, align 4
  %2 = sext i32 %1 to i64
; ELF64: lwa
; ELF64-NOT: extsw
  ret i64 %2
}
