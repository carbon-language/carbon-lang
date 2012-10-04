; RUN: opt < %s -sroa -S | FileCheck %s
; RUN: opt < %s -sroa -force-ssa-updater -S | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

define i8 @test1() {
; We fully promote these to the i24 load or store size, resulting in just masks
; and other operations that instcombine will fold, but no alloca. Note this is
; the same as test12 in basictest.ll, but here we assert big-endian byte
; ordering.
;
; CHECK: @test1

entry:
  %a = alloca [3 x i8]
  %b = alloca [3 x i8]
; CHECK-NOT: alloca

  %a0ptr = getelementptr [3 x i8]* %a, i64 0, i32 0
  store i8 0, i8* %a0ptr
  %a1ptr = getelementptr [3 x i8]* %a, i64 0, i32 1
  store i8 0, i8* %a1ptr
  %a2ptr = getelementptr [3 x i8]* %a, i64 0, i32 2
  store i8 0, i8* %a2ptr
  %aiptr = bitcast [3 x i8]* %a to i24*
  %ai = load i24* %aiptr
; CHCEK-NOT: store
; CHCEK-NOT: load
; CHECK:      %[[mask0:.*]] = and i24 undef, 65535
; CHECK-NEXT: %[[mask1:.*]] = and i24 %[[mask0]], -65281
; CHECK-NEXT: %[[mask2:.*]] = and i24 %[[mask1]], -256

  %biptr = bitcast [3 x i8]* %b to i24*
  store i24 %ai, i24* %biptr
  %b0ptr = getelementptr [3 x i8]* %b, i64 0, i32 0
  %b0 = load i8* %b0ptr
  %b1ptr = getelementptr [3 x i8]* %b, i64 0, i32 1
  %b1 = load i8* %b1ptr
  %b2ptr = getelementptr [3 x i8]* %b, i64 0, i32 2
  %b2 = load i8* %b2ptr
; CHCEK-NOT: store
; CHCEK-NOT: load
; CHECK:      %[[shift0:.*]] = lshr i24 %[[mask2]], 16
; CHECK-NEXT: %[[trunc0:.*]] = trunc i24 %[[shift0]] to i8
; CHECK-NEXT: %[[shift1:.*]] = lshr i24 %[[mask2]], 8
; CHECK-NEXT: %[[trunc1:.*]] = trunc i24 %[[shift1]] to i8
; CHECK-NEXT: %[[trunc2:.*]] = trunc i24 %[[mask2]] to i8

  %bsum0 = add i8 %b0, %b1
  %bsum1 = add i8 %bsum0, %b2
  ret i8 %bsum1
; CHECK:      %[[sum0:.*]] = add i8 %[[trunc0]], %[[trunc1]]
; CHECK-NEXT: %[[sum1:.*]] = add i8 %[[sum0]], %[[trunc2]]
; CHECK-NEXT: ret i8 %[[sum1]]
}

define i64 @test2() {
; Test for various mixed sizes of integer loads and stores all getting
; promoted.
;
; CHECK: @test2

entry:
  %a = alloca [7 x i8]
; CHECK-NOT: alloca

  %a0ptr = getelementptr [7 x i8]* %a, i64 0, i32 0
  %a1ptr = getelementptr [7 x i8]* %a, i64 0, i32 1
  %a2ptr = getelementptr [7 x i8]* %a, i64 0, i32 2
  %a3ptr = getelementptr [7 x i8]* %a, i64 0, i32 3

; CHCEK-NOT: store
; CHCEK-NOT: load

  %a0i16ptr = bitcast i8* %a0ptr to i16*
  store i16 1, i16* %a0i16ptr
; CHECK:      %[[mask:.*]] = and i56 undef, 1099511627775
; CHECK-NEXT: %[[or:.*]] = or i56 %[[mask]], 1099511627776

  %a1i4ptr = bitcast i8* %a1ptr to i4*
  store i4 1, i4* %a1i4ptr
; CHECK:      %[[mask:.*]] = and i56 %[[or]], -16492674416641
; CHECK-NEXT: %[[or:.*]] = or i56 %[[mask]], 1099511627776

  store i8 1, i8* %a2ptr
; CHECK-NEXT: %[[mask:.*]] = and i56 %[[or]], -1095216660481
; CHECK-NEXT: %[[or:.*]] = or i56 %[[mask]], 4294967296

  %a3i24ptr = bitcast i8* %a3ptr to i24*
  store i24 1, i24* %a3i24ptr
; CHECK-NEXT: %[[mask:.*]] = and i56 %[[or]], -4294967041
; CHECK-NEXT: %[[or:.*]] = or i56 %[[mask]], 256

  %a2i40ptr = bitcast i8* %a2ptr to i40*
  store i40 1, i40* %a2i40ptr
; CHECK-NEXT: %[[mask:.*]] = and i56 %[[or]], -1099511627776
; CHECK-NEXT: %[[or:.*]] = or i56 %[[mask]], 1

; CHCEK-NOT: store
; CHCEK-NOT: load

  %aiptr = bitcast [7 x i8]* %a to i56*
  %ai = load i56* %aiptr
  %ret = zext i56 %ai to i64
  ret i64 %ret
; CHECK:      %[[ret:.*]] = zext i56 %[[or]] to i64
; CHECK-NEXT: ret i64 %[[ret]]
}
