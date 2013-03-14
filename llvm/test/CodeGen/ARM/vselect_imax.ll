; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; Make sure that ARM backend with NEON handles vselect.

define void @vmax_v4i32(<4 x i32>* %m, <4 x i32> %a, <4 x i32> %b) {
; CHECK: vcgt.s32 [[QR:q[0-9]+]], [[Q1:q[0-9]+]], [[Q2:q[0-9]+]]
; CHECK: vbsl [[QR]], [[Q1]], [[Q2]]
    %cmpres = icmp sgt <4 x i32> %a, %b
    %maxres = select <4 x i1> %cmpres, <4 x i32> %a,  <4 x i32> %b
    store <4 x i32> %maxres, <4 x i32>* %m
    ret void
}

; We adjusted the cost model of the following selects. When we improve code
; lowering we also need to adjust the cost.
; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -march=arm -mcpu=cortex-a8 | FileCheck %s --check-prefix=COST
%T0_3 = type <4 x i8>
%T1_3 = type <4 x i1>
; CHECK: func_blend3:
define void @func_blend3(%T0_3* %loadaddr, %T0_3* %loadaddr2,
                           %T1_3* %blend, %T0_3* %storeaddr) {
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: vldr
  %v0 = load %T0_3* %loadaddr
  %v1 = load %T0_3* %loadaddr2
  %c = load %T1_3* %blend
; COST: func_blend3
; COST: cost of 10 {{.*}} select
  %r = select %T1_3 %c, %T0_3 %v0, %T0_3 %v1
  store %T0_3 %r, %T0_3* %storeaddr
  ret void
}
%T0_4 = type <8 x i8>
%T1_4 = type <8 x i1>
; CHECK: func_blend4:
define void @func_blend4(%T0_4* %loadaddr, %T0_4* %loadaddr2,
                           %T1_4* %blend, %T0_4* %storeaddr) {
  %v0 = load %T0_4* %loadaddr
  %v1 = load %T0_4* %loadaddr2
  %c = load %T1_4* %blend
; check: strb
; check: strb
; check: strb
; check: strb
; check: vldr
; COST: func_blend4
; COST: cost of 17 {{.*}} select
  %r = select %T1_4 %c, %T0_4 %v0, %T0_4 %v1
  store %T0_4 %r, %T0_4* %storeaddr
  ret void
}
%T0_5 = type <16 x i8>
%T1_5 = type <16 x i1>
; CHECK: func_blend5:
define void @func_blend5(%T0_5* %loadaddr, %T0_5* %loadaddr2,
                           %T1_5* %blend, %T0_5* %storeaddr) {
  %v0 = load %T0_5* %loadaddr
  %v1 = load %T0_5* %loadaddr2
  %c = load %T1_5* %blend
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: vld
; COST: func_blend5
; COST: cost of 33 {{.*}} select
  %r = select %T1_5 %c, %T0_5 %v0, %T0_5 %v1
  store %T0_5 %r, %T0_5* %storeaddr
  ret void
}
%T0_8 = type <4 x i16>
%T1_8 = type <4 x i1>
; CHECK: func_blend8:
define void @func_blend8(%T0_8* %loadaddr, %T0_8* %loadaddr2,
                           %T1_8* %blend, %T0_8* %storeaddr) {
  %v0 = load %T0_8* %loadaddr
  %v1 = load %T0_8* %loadaddr2
  %c = load %T1_8* %blend
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: vld
; COST: func_blend8
; COST: cost of 9 {{.*}} select
  %r = select %T1_8 %c, %T0_8 %v0, %T0_8 %v1
  store %T0_8 %r, %T0_8* %storeaddr
  ret void
}
%T0_9 = type <8 x i16>
%T1_9 = type <8 x i1>
; CHECK: func_blend9:
define void @func_blend9(%T0_9* %loadaddr, %T0_9* %loadaddr2,
                           %T1_9* %blend, %T0_9* %storeaddr) {
  %v0 = load %T0_9* %loadaddr
  %v1 = load %T0_9* %loadaddr2
  %c = load %T1_9* %blend
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: vld
; COST: func_blend9
; COST: cost of 17 {{.*}} select
  %r = select %T1_9 %c, %T0_9 %v0, %T0_9 %v1
  store %T0_9 %r, %T0_9* %storeaddr
  ret void
}
%T0_10 = type <16 x i16>
%T1_10 = type <16 x i1>
; CHECK: func_blend10:
define void @func_blend10(%T0_10* %loadaddr, %T0_10* %loadaddr2,
                           %T1_10* %blend, %T0_10* %storeaddr) {
  %v0 = load %T0_10* %loadaddr
  %v1 = load %T0_10* %loadaddr2
  %c = load %T1_10* %blend
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: vld
; COST: func_blend10
; COST: cost of 40 {{.*}} select
  %r = select %T1_10 %c, %T0_10 %v0, %T0_10 %v1
  store %T0_10 %r, %T0_10* %storeaddr
  ret void
}
%T0_14 = type <8 x i32>
%T1_14 = type <8 x i1>
; CHECK: func_blend14:
define void @func_blend14(%T0_14* %loadaddr, %T0_14* %loadaddr2,
                           %T1_14* %blend, %T0_14* %storeaddr) {
  %v0 = load %T0_14* %loadaddr
  %v1 = load %T0_14* %loadaddr2
  %c = load %T1_14* %blend
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: strb
; COST: func_blend14
; COST: cost of 41 {{.*}} select
  %r = select %T1_14 %c, %T0_14 %v0, %T0_14 %v1
  store %T0_14 %r, %T0_14* %storeaddr
  ret void
}
%T0_15 = type <16 x i32>
%T1_15 = type <16 x i1>
; CHECK: func_blend15:
define void @func_blend15(%T0_15* %loadaddr, %T0_15* %loadaddr2,
                           %T1_15* %blend, %T0_15* %storeaddr) {
  %v0 = load %T0_15* %loadaddr
  %v1 = load %T0_15* %loadaddr2
  %c = load %T1_15* %blend
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: strb
; COST: func_blend15
; COST: cost of 82 {{.*}} select
  %r = select %T1_15 %c, %T0_15 %v0, %T0_15 %v1
  store %T0_15 %r, %T0_15* %storeaddr
  ret void
}
%T0_18 = type <4 x i64>
%T1_18 = type <4 x i1>
; CHECK: func_blend18:
define void @func_blend18(%T0_18* %loadaddr, %T0_18* %loadaddr2,
                           %T1_18* %blend, %T0_18* %storeaddr) {
  %v0 = load %T0_18* %loadaddr
  %v1 = load %T0_18* %loadaddr2
  %c = load %T1_18* %blend
; CHECK: strh
; CHECK: strh
; CHECK: strh
; CHECK: strh
; COST: func_blend18
; COST: cost of 19 {{.*}} select
  %r = select %T1_18 %c, %T0_18 %v0, %T0_18 %v1
  store %T0_18 %r, %T0_18* %storeaddr
  ret void
}
%T0_19 = type <8 x i64>
%T1_19 = type <8 x i1>
; CHECK: func_blend19:
define void @func_blend19(%T0_19* %loadaddr, %T0_19* %loadaddr2,
                           %T1_19* %blend, %T0_19* %storeaddr) {
  %v0 = load %T0_19* %loadaddr
  %v1 = load %T0_19* %loadaddr2
  %c = load %T1_19* %blend
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: strb
; COST: func_blend19
; COST: cost of 50 {{.*}} select
  %r = select %T1_19 %c, %T0_19 %v0, %T0_19 %v1
  store %T0_19 %r, %T0_19* %storeaddr
  ret void
}
%T0_20 = type <16 x i64>
%T1_20 = type <16 x i1>
; CHECK: func_blend20:
define void @func_blend20(%T0_20* %loadaddr, %T0_20* %loadaddr2,
                           %T1_20* %blend, %T0_20* %storeaddr) {
  %v0 = load %T0_20* %loadaddr
  %v1 = load %T0_20* %loadaddr2
  %c = load %T1_20* %blend
; CHECK: strb
; CHECK: strb
; CHECK: strb
; CHECK: strb
; COST: func_blend20
; COST: cost of 100 {{.*}} select
  %r = select %T1_20 %c, %T0_20 %v0, %T0_20 %v1
  store %T0_20 %r, %T0_20* %storeaddr
  ret void
}
