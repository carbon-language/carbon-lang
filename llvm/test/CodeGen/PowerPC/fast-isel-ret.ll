; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

define signext i8 @ret2(i8 signext %a) nounwind uwtable ssp {
entry:
; ELF64: ret2
; ELF64: extsb
; ELF64: blr
  ret i8 %a
}

define zeroext i8 @ret3(i8 signext %a) nounwind uwtable ssp {
entry:
; ELF64: ret3
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 56
; ELF64: blr
  ret i8 %a
}

define signext i16 @ret4(i16 signext %a) nounwind uwtable ssp {
entry:
; ELF64: ret4
; ELF64: extsh
; ELF64: blr
  ret i16 %a
}

define zeroext i16 @ret5(i16 signext %a) nounwind uwtable ssp {
entry:
; ELF64: ret5
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 48
; ELF64: blr
  ret i16 %a
}

define i16 @ret6(i16 %a) nounwind uwtable ssp {
entry:
; ELF64: ret6
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 48
; ELF64: blr
  ret i16 %a
}

define signext i32 @ret7(i32 signext %a) nounwind uwtable ssp {
entry:
; ELF64: ret7
; ELF64: extsw
; ELF64: blr
  ret i32 %a
}

define zeroext i32 @ret8(i32 signext %a) nounwind uwtable ssp {
entry:
; ELF64: ret8
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 32
; ELF64: blr
  ret i32 %a
}

define i32 @ret9(i32 %a) nounwind uwtable ssp {
entry:
; ELF64: ret9
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 32
; ELF64: blr
  ret i32 %a
}

define i64 @ret10(i64 %a) nounwind uwtable ssp {
entry:
; ELF64: ret10
; ELF64-NOT: exts
; ELF64-NOT: rldicl
; ELF64: blr
  ret i64 %a
}

define float @ret11(float %a) nounwind uwtable ssp {
entry:
; ELF64: ret11
; ELF64: blr
  ret float %a
}

define double @ret12(double %a) nounwind uwtable ssp {
entry:
; ELF64: ret12
; ELF64: blr
  ret double %a
}

define i8 @ret13() nounwind uwtable ssp {
entry:
; ELF64: ret13
; ELF64: li
; ELF64: blr
  ret i8 15;
}

define i16 @ret14() nounwind uwtable ssp {
entry:
; ELF64: ret14
; ELF64: li
; ELF64: blr
  ret i16 -225;
}

define i32 @ret15() nounwind uwtable ssp {
entry:
; ELF64: ret15
; ELF64: lis
; ELF64: ori
; ELF64: blr
  ret i32 278135;
}

define i64 @ret16() nounwind uwtable ssp {
entry:
; ELF64: ret16
; ELF64: li
; ELF64: sldi
; ELF64: oris
; ELF64: ori
; ELF64: blr
  ret i64 27813515225;
}

define float @ret17() nounwind uwtable ssp {
entry:
; ELF64: ret17
; ELF64: addis
; ELF64: lfs
; ELF64: blr
  ret float 2.5;
}

define double @ret18() nounwind uwtable ssp {
entry:
; ELF64: ret18
; ELF64: addis
; ELF64: lfd
; ELF64: blr
  ret double 2.5e-33;
}
