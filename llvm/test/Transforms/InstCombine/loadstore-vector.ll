; RUN: opt -instcombine -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Basic test for turning element extraction from integer loads and element
; insertion into integer stores into extraction and insertion with vectors.
define void @test1({ float, float }* %x, float %a, float %b, { float, float }* %out) {
; CHECK-LABEL: @test1(
entry:
  %x.cast = bitcast { float, float }* %x to i64*
  %x.load = load i64* %x.cast, align 4
; CHECK-NOT: load i64*
; CHECK: %[[LOAD:.*]] = load <2 x float>*

  %lo.trunc = trunc i64 %x.load to i32
  %hi.shift = lshr i64 %x.load, 32
  %hi.trunc = trunc i64 %hi.shift to i32
  %hi.cast = bitcast i32 %hi.trunc to float
  %lo.cast = bitcast i32 %lo.trunc to float
; CHECK-NOT: trunc
; CHECK-NOT: lshr
; CHECK: %[[HI:.*]] = extractelement <2 x float> %[[LOAD]], i32 1
; CHECK: %[[LO:.*]] = extractelement <2 x float> %[[LOAD]], i32 0

  %add.i.i = fadd float %lo.cast, %a
  %add5.i.i = fadd float %hi.cast, %b
; CHECK: %[[LO_SUM:.*]] = fadd float %[[LO]], %a
; CHECK: %[[HI_SUM:.*]] = fadd float %[[HI]], %b

  %add.lo.cast = bitcast float %add.i.i to i32
  %add.hi.cast = bitcast float %add5.i.i to i32
  %add.hi.ext = zext i32 %add.hi.cast to i64
  %add.hi.shift = shl nuw i64 %add.hi.ext, 32
  %add.lo.ext = zext i32 %add.lo.cast to i64
  %add.lo.or = or i64 %add.hi.shift, %add.lo.ext
; CHECK-NOT: zext i32
; CHECK-NOT: shl {{.*}} i64
; CHECK-NOT: or i64
; CHECK: %[[INSERT1:.*]] = insertelement <2 x float> undef, float %[[LO_SUM]], i32 0
; CHECK: %[[INSERT2:.*]] = insertelement <2 x float> %[[INSERT1]], float %[[HI_SUM]], i32 1

  %out.cast = bitcast { float, float }* %out to i64*
  store i64 %add.lo.or, i64* %out.cast, align 4
; CHECK-NOT: store i64
; CHECK: store <2 x float> %[[INSERT2]]

  ret void
}

define void @test2({ float, float }* %x, float %a, float %b, { float, float }* %out1, { float, float }* %out2) {
; CHECK-LABEL: @test2(
entry:
  %x.cast = bitcast { float, float }* %x to i64*
  %x.load = load i64* %x.cast, align 4
; CHECK-NOT: load i64*
; CHECK: %[[LOAD:.*]] = load <2 x float>*

  %lo.trunc = trunc i64 %x.load to i32
  %hi.shift = lshr i64 %x.load, 32
  %hi.trunc = trunc i64 %hi.shift to i32
  %hi.cast = bitcast i32 %hi.trunc to float
  %lo.cast = bitcast i32 %lo.trunc to float
; CHECK-NOT: trunc
; CHECK-NOT: lshr
; CHECK: %[[HI:.*]] = extractelement <2 x float> %[[LOAD]], i32 1
; CHECK: %[[LO:.*]] = extractelement <2 x float> %[[LOAD]], i32 0

  %add.i.i = fadd float %lo.cast, %a
  %add5.i.i = fadd float %hi.cast, %b
; CHECK: %[[LO_SUM:.*]] = fadd float %[[LO]], %a
; CHECK: %[[HI_SUM:.*]] = fadd float %[[HI]], %b

  %add.lo.cast = bitcast float %add.i.i to i32
  %add.hi.cast = bitcast float %add5.i.i to i32
  %add.hi.ext = zext i32 %add.hi.cast to i64
  %add.hi.shift = shl nuw i64 %add.hi.ext, 32
  %add.lo.ext = zext i32 %add.lo.cast to i64
  %add.lo.or = or i64 %add.hi.shift, %add.lo.ext
; CHECK-NOT: zext i32
; CHECK-NOT: shl {{.*}} i64
; CHECK-NOT: or i64
; CHECK: %[[INSERT1:.*]] = insertelement <2 x float> undef, float %[[LO_SUM]], i32 0
; CHECK: %[[INSERT2:.*]] = insertelement <2 x float> %[[INSERT1]], float %[[HI_SUM]], i32 1

  %out1.cast = bitcast { float, float }* %out1 to i64*
  store i64 %add.lo.or, i64* %out1.cast, align 4
  %out2.cast = bitcast { float, float }* %out2 to i64*
  store i64 %add.lo.or, i64* %out2.cast, align 4
; CHECK-NOT: store i64
; CHECK: store <2 x float> %[[INSERT2]]
; CHECK-NOT: store i64
; CHECK: store <2 x float> %[[INSERT2]]

  ret void
}

; We handle some cases where there is partial CSE but not complete CSE of
; repeated insertion and extraction. Currently, we don't catch the store side
; yet because it would require extreme heroics to match this reliably.
define void @test3({ float, float, float }* %x, float %a, float %b, { float, float, float }* %out1, { float, float, float }* %out2) {
; CHECK-LABEL: @test3(
entry:
  %x.cast = bitcast { float, float, float }* %x to i96*
  %x.load = load i96* %x.cast, align 4
; CHECK-NOT: load i96*
; CHECK: %[[LOAD:.*]] = load <3 x float>*

  %lo.trunc = trunc i96 %x.load to i32
  %lo.cast = bitcast i32 %lo.trunc to float
  %mid.shift = lshr i96 %x.load, 32
  %mid.trunc = trunc i96 %mid.shift to i32
  %mid.cast = bitcast i32 %mid.trunc to float
  %mid.trunc2 = trunc i96 %mid.shift to i32
  %mid.cast2 = bitcast i32 %mid.trunc2 to float
  %hi.shift = lshr i96 %mid.shift, 32
  %hi.trunc = trunc i96 %hi.shift to i32
  %hi.cast = bitcast i32 %hi.trunc to float
; CHECK-NOT: trunc
; CHECK-NOT: lshr
; CHECK: %[[LO:.*]] = extractelement <3 x float> %[[LOAD]], i32 0
; CHECK: %[[MID1:.*]] = extractelement <3 x float> %[[LOAD]], i32 1
; CHECK: %[[MID2:.*]] = extractelement <3 x float> %[[LOAD]], i32 1
; CHECK: %[[HI:.*]] = extractelement <3 x float> %[[LOAD]], i32 2

  %add.lo = fadd float %lo.cast, %a
  %add.mid = fadd float %mid.cast, %b
  %add.hi = fadd float %hi.cast, %mid.cast2
; CHECK: %[[LO_SUM:.*]] = fadd float %[[LO]], %a
; CHECK: %[[MID_SUM:.*]] = fadd float %[[MID1]], %b
; CHECK: %[[HI_SUM:.*]] = fadd float %[[HI]], %[[MID2]]

  %add.lo.cast = bitcast float %add.lo to i32
  %add.mid.cast = bitcast float %add.mid to i32
  %add.hi.cast = bitcast float %add.hi to i32
  %result.hi.ext = zext i32 %add.hi.cast to i96
  %result.hi.shift = shl nuw i96 %result.hi.ext, 32
  %result.mid.ext = zext i32 %add.mid.cast to i96
  %result.mid.or = or i96 %result.hi.shift, %result.mid.ext
  %result.mid.shift = shl nuw i96 %result.mid.or, 32
  %result.lo.ext = zext i32 %add.lo.cast to i96
  %result.lo.or = or i96 %result.mid.shift, %result.lo.ext
; FIXME-NOT: zext i32
; FIXME-NOT: shl {{.*}} i64
; FIXME-NOT: or i64
; FIXME: %[[INSERT1:.*]] = insertelement <3 x float> undef, float %[[HI_SUM]], i32 2
; FIXME: %[[INSERT2:.*]] = insertelement <3 x float> %[[INSERT1]], float %[[MID_SUM]], i32 1
; FIXME: %[[INSERT3:.*]] = insertelement <3 x float> %[[INSERT2]], float %[[LO_SUM]], i32 0

  %out1.cast = bitcast { float, float, float }* %out1 to i96*
  store i96 %result.lo.or, i96* %out1.cast, align 4
; FIXME-NOT: store i96
; FIXME: store <3 x float> %[[INSERT3]]

  %result2.lo.ext = zext i32 %add.lo.cast to i96
  %result2.lo.or = or i96 %result.mid.shift, %result2.lo.ext
; FIXME-NOT: zext i32
; FIXME-NOT: shl {{.*}} i64
; FIXME-NOT: or i64
; FIXME: %[[INSERT4:.*]] = insertelement <3 x float> %[[INSERT2]], float %[[LO_SUM]], i32 0

  %out2.cast = bitcast { float, float, float }* %out2 to i96*
  store i96 %result2.lo.or, i96* %out2.cast, align 4
; FIXME-NOT: store i96
; FIXME: store <3 x float>

  ret void
}

; Basic test that pointers work correctly as the element type.
define void @test4({ i8*, i8* }* %x, i64 %a, i64 %b, { i8*, i8* }* %out) {
; CHECK-LABEL: @test4(
entry:
  %x.cast = bitcast { i8*, i8* }* %x to i128*
  %x.load = load i128* %x.cast, align 4
; CHECK-NOT: load i128*
; CHECK: %[[LOAD:.*]] = load <2 x i8*>* {{.*}}, align 4

  %lo.trunc = trunc i128 %x.load to i64
  %hi.shift = lshr i128 %x.load, 64
  %hi.trunc = trunc i128 %hi.shift to i64
  %hi.cast = inttoptr i64 %hi.trunc to i8*
  %lo.cast = inttoptr i64 %lo.trunc to i8*
; CHECK-NOT: trunc
; CHECK-NOT: lshr
; CHECK: %[[HI:.*]] = extractelement <2 x i8*> %[[LOAD]], i32 1
; CHECK: %[[LO:.*]] = extractelement <2 x i8*> %[[LOAD]], i32 0

  %gep.lo = getelementptr i8* %lo.cast, i64 %a
  %gep.hi = getelementptr i8* %hi.cast, i64 %b
; CHECK: %[[LO_GEP:.*]] = getelementptr i8* %[[LO]], i64 %a
; CHECK: %[[HI_GEP:.*]] = getelementptr i8* %[[HI]], i64 %b

  %gep.lo.cast = ptrtoint i8* %gep.lo to i64
  %gep.hi.cast = ptrtoint i8* %gep.hi to i64
  %gep.hi.ext = zext i64 %gep.hi.cast to i128
  %gep.hi.shift = shl nuw i128 %gep.hi.ext, 64
  %gep.lo.ext = zext i64 %gep.lo.cast to i128
  %gep.lo.or = or i128 %gep.hi.shift, %gep.lo.ext
; CHECK-NOT: zext i32
; CHECK-NOT: shl {{.*}} i64
; CHECK-NOT: or i64
; CHECK: %[[INSERT1:.*]] = insertelement <2 x i8*> undef, i8* %[[LO_GEP]], i32 0
; CHECK: %[[INSERT2:.*]] = insertelement <2 x i8*> %[[INSERT1]], i8* %[[HI_GEP]], i32 1

  %out.cast = bitcast { i8*, i8* }* %out to i128*
  store i128 %gep.lo.or, i128* %out.cast, align 4
; CHECK-NOT: store i128
; CHECK: store <2 x i8*> %[[INSERT2]], <2 x i8*>* {{.*}}, align 4

  ret void
}
