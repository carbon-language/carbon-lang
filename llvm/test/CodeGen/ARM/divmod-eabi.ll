; RUN: llc -mtriple armv7-none-eabi %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-none-eabihf %s -o - | FileCheck %s --check-prefix=EABI
; Both "none-eabi" and "androideabi" must lower SREM/UREM to __aeabi_{u,i}divmod
; RUN: llc -mtriple armv7-linux-androideabi %s -o - | FileCheck %s --check-prefix=EABI
; RUN: llc -mtriple armv7-linux-gnueabi %s -o - | FileCheck %s --check-prefix=GNU
; RUN: llc -mtriple armv7-apple-darwin %s -o - | FileCheck %s --check-prefix=DARWIN
; FIXME: long-term, we will use "-apple-macho" and won't need this exception:
; RUN: llc -mtriple armv7-apple-darwin-eabi %s -o - | FileCheck %s --check-prefix=DARWIN

define signext i16 @f16(i16 signext %a, i16 signext %b) {
; EABI-LABEL: f16:
; GNU-LABEL: f16:
; DARWIN-LABEL: f16:
entry:
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %div = sdiv i32 %conv, %conv1
  %rem = srem i32 %conv, %conv1
; EABI: __aeabi_idivmod
; EABI: mov [[div:r[0-9]+]], r0
; EABI: mov [[rem:r[0-9]+]], r1
; GNU: __aeabi_idiv
; GNU: mov [[sum:r[0-9]+]], r0
; GNU: __modsi3
; GNU: add [[sum]]{{.*}}r0
; DARWIN: ___divsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; DARWIN: __modsi3
; DARWIN: add [[sum]]{{.*}}r0
  %rem8 = srem i32 %conv1, %conv
; EABI: __aeabi_idivmod
; GNU: __modsi3
; DARWIN: __modsi3
  %add = add nsw i32 %rem, %div
  %add13 = add nsw i32 %add, %rem8
  %conv14 = trunc i32 %add13 to i16
; EABI: add r0{{.*}}r1
; EABI: sxth r0, r0
; GNU: add r0{{.*}}[[sum]]
; GNU: sxth r0, r0
; DARWIN: add r0{{.*}}[[sum]]
; DARWIN: sxth r0, r0
  ret i16 %conv14
}

define i32 @f32(i32 %a, i32 %b) {
; EABI-LABEL: f32:
; GNU-LABEL: f32:
; DARWIN-LABEL: f32:
entry:
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; EABI: mov [[div:r[0-9]+]], r0
; EABI: mov [[rem:r[0-9]+]], r1
; GNU: __aeabi_idiv
; GNU: mov [[sum:r[0-9]+]], r0
; GNU: __modsi3
; GNU: add [[sum]]{{.*}}r0
; DARWIN: ___divsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; DARWIN: __modsi3
; DARWIN: add [[sum]]{{.*}}r0
  %rem1 = srem i32 %b, %a
; EABI: __aeabi_idivmod
; GNU: __modsi3
; DARWIN: __modsi3
  %add = add nsw i32 %rem, %div
  %add2 = add nsw i32 %add, %rem1
; EABI: add r0{{.*}}r1
; GNU: add r0{{.*}}[[sum]]
; DARWIN: add r0{{.*}}[[sum]]
  ret i32 %add2
}

define i32 @uf(i32 %a, i32 %b) {
; EABI-LABEL: uf:
; GNU-LABEL: uf:
; DARWIN-LABEL: uf:
entry:
  %div = udiv i32 %a, %b
  %rem = urem i32 %a, %b
; EABI: __aeabi_uidivmod
; GNU: __aeabi_uidiv
; GNU: mov [[sum:r[0-9]+]], r0
; GNU: __umodsi3
; GNU: add [[sum]]{{.*}}r0
; DARWIN: ___udivsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; DARWIN: __umodsi3
; DARWIN: add [[sum]]{{.*}}r0
  %rem1 = urem i32 %b, %a
; EABI: __aeabi_uidivmod
; GNU: __umodsi3
; DARWIN: __umodsi3
  %add = add nuw i32 %rem, %div
  %add2 = add nuw i32 %add, %rem1
; EABI: add r0{{.*}}r1
; GNU: add r0{{.*}}[[sum]]
; DARWIN: add r0{{.*}}[[sum]]
  ret i32 %add2
}

; FIXME: AEABI is not lowering long u/srem into u/ldivmod
define i64 @longf(i64 %a, i64 %b) {
; EABI-LABEL: longf:
; GNU-LABEL: longf:
; DARWIN-LABEL: longf:
entry:
  %div = sdiv i64 %a, %b
  %rem = srem i64 %a, %b
; EABI: __aeabi_ldivmod
; GNU: __aeabi_ldivmod
; GNU: mov [[div1:r[0-9]+]], r0
; GNU: mov [[div2:r[0-9]+]], r1
; DARWIN: ___divdi3
; DARWIN: mov [[div1:r[0-9]+]], r0
; DARWIN: mov [[div2:r[0-9]+]], r1
; DARWIN: __moddi3
  %add = add nsw i64 %rem, %div
; GNU: adds r0{{.*}}[[div1]]
; GNU: adc r1{{.*}}[[div2]]
; DARWIN: adds r0{{.*}}[[div1]]
; DARWIN: adc r1{{.*}}[[div2]]
  ret i64 %add
}

define i32 @g1(i32 %a, i32 %b) {
; EABI-LABEL: g1:
; GNU-LABEL: g1:
; DARWIN-LABEL: g1:
entry:
  %div = sdiv i32 %a, %b
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; GNU: __aeabi_idiv
; GNU: mov [[sum:r[0-9]+]], r0
; GNU: __modsi3
; DARWIN: ___divsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
; DARWIN: __modsi3
  %add = add nsw i32 %rem, %div
; EABI:	add	r0{{.*}}r1
; GNU: add r0{{.*}}[[sum]]
; DARWIN: add r0{{.*}}[[sum]]
  ret i32 %add
}

; On both Darwin and Gnu, this is just a call to __modsi3
define i32 @g2(i32 %a, i32 %b) {
; EABI-LABEL: g2:
; GNU-LABEL: g2:
; DARWIN-LABEL: g2:
entry:
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; GNU: __modsi3
; DARWIN: __modsi3
  ret i32 %rem
; EABI:	mov	r0, r1
}

define i32 @g3(i32 %a, i32 %b) {
; EABI-LABEL: g3:
; GNU-LABEL: g3:
; DARWIN-LABEL: g3:
entry:
  %rem = srem i32 %a, %b
; EABI: __aeabi_idivmod
; EABI: mov [[mod:r[0-9]+]], r1
; GNU: __modsi3
; GNU: mov [[sum:r[0-9]+]], r0
; DARWIN: __modsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
  %rem1 = srem i32 %b, %rem
; EABI: __aeabi_idivmod
; GNU: __modsi3
; DARWIN: __modsi3
  %add = add nsw i32 %rem1, %rem
; EABI: add r0, r1, [[mod]]
; GNU: add r0{{.*}}[[sum]]
; DARWIN: add r0{{.*}}[[sum]]
  ret i32 %add
}

define i32 @g4(i32 %a, i32 %b) {
; EABI-LABEL: g4:
; GNU-LABEL: g4:
; DARWIN-LABEL: g4:
entry:
  %div = sdiv i32 %a, %b
; EABI: __aeabi_idiv{{$}}
; EABI: mov [[div:r[0-9]+]], r0
; GNU: __aeabi_idiv
; GNU: mov [[sum:r[0-9]+]], r0
; DARWIN: ___divsi3
; DARWIN: mov [[sum:r[0-9]+]], r0
  %rem = srem i32 %b, %div
; EABI: __aeabi_idivmod
; GNU: __modsi3
; DARWIN: __modsi3
  %add = add nsw i32 %rem, %div
; EABI: add r0, r1, [[div]]
; GNU: add r0{{.*}}[[sum]]
; DARWIN: add r0{{.*}}[[sum]]
  ret i32 %add
}
