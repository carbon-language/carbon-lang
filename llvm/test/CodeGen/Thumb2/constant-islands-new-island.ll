; RUN: llc < %s -mtriple=thumbv7-apple-ios7.1.0 -mcpu=cortex-a8 | FileCheck %s

@global = private unnamed_addr constant [16 x float] [float 1.500000e+00, float 2.500000e+00, float 4.000000e+00, float 6.000000e+00, float 1.000000e+01, float 1.600000e+01, float 2.500000e+01, float 3.500000e+01, float 5.000000e+01, float 7.000000e+01, float 9.500000e+01, float 1.200000e+02, float 1.500000e+02, float 1.850000e+02, float 2.400000e+02, float 3.000000e+02], align 4

; Check that new water is created by splitting the basic block right after the
; load instruction. Previously, new water was created before the load
; instruction, which caused the pass to fail to converge.
;
; CHECK:      LBB0_324:                               @ %bb35
; CHECK:        vldr  s6, LCPI0_480
; CHECK-NEXT:   b.w LBB0_461
; CHECK-NEXT:   .align  2
; CHECK-NEXT: @ BB#325:
; CHECK-NEXT:   .data_region
; CHECK-NEXT: LCPI0_480:
; CHECK-NEXT:   .long 1084252750              @ float 5.01200008

define hidden void @wibble(float %arg, float %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, float* nocapture %arg6, i32* nocapture readonly %arg7, float* nocapture %arg8, float* nocapture readnone %arg9, float* nocapture %arg10, i8* nocapture readnone %arg11, float* nocapture %arg12, float* nocapture %arg13, i32* %arg14, float* nocapture %arg15, float* nocapture readnone %arg16, float* nocapture readnone %arg17, float* nocapture %arg18, float* nocapture readnone %arg19, float* nocapture readnone %arg20, float* nocapture readnone %arg21, float* nocapture readnone %arg22, float* nocapture %arg23) {
bb:
  switch i32 %arg4, label %bb50 [
    i32 1, label %bb24
    i32 2, label %bb25
    i32 3, label %bb26
    i32 4, label %bb27
    i32 5, label %bb28
    i32 6, label %bb29
    i32 7, label %bb30
    i32 8, label %bb31
    i32 9, label %bb32
    i32 10, label %bb33
    i32 11, label %bb34
    i32 12, label %bb35
    i32 13, label %bb36
    i32 14, label %bb37
    i32 15, label %bb38
    i32 16, label %bb39
    i32 17, label %bb40
    i32 18, label %bb41
    i32 19, label %bb42
    i32 20, label %bb43
    i32 21, label %bb44
    i32 22, label %bb45
    i32 23, label %bb52
    i32 24, label %bb46
    i32 25, label %bb47
    i32 26, label %bb48
    i32 27, label %bb49
  ]

bb24:                                             ; preds = %bb
  store i32 4, i32* %arg14, align 4
  br label %bb52

bb25:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb26:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb27:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb28:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb29:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb30:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb31:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb32:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb33:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb34:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb35:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb36:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb37:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb38:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb39:                                             ; preds = %bb
  store i32 4, i32* %arg14, align 4
  br label %bb52

bb40:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb41:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb42:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb43:                                             ; preds = %bb
  store i32 5, i32* %arg14, align 4
  br label %bb52

bb44:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb45:                                             ; preds = %bb
  store i32 6, i32* %arg14, align 4
  br label %bb52

bb46:                                             ; preds = %bb
  br label %bb52

bb47:                                             ; preds = %bb
  br label %bb52

bb48:                                             ; preds = %bb
  br label %bb52

bb49:                                             ; preds = %bb
  br label %bb52

bb50:                                             ; preds = %bb
  %tmp = icmp eq i32 %arg4, 28
  br i1 %tmp, label %bb51, label %bb52

bb51:                                             ; preds = %bb50
  store i32 4, i32* %arg14, align 4
  br label %bb52

bb52:                                             ; preds = %bb51, %bb50, %bb49, %bb48, %bb47, %bb46, %bb45, %bb44, %bb43, %bb42, %bb41, %bb40, %bb39, %bb38, %bb37, %bb36, %bb35, %bb34, %bb33, %bb32, %bb31, %bb30, %bb29, %bb28, %bb27, %bb26, %bb25, %bb24, %bb
  %tmp53 = phi double [ 0.000000e+00, %bb24 ], [ 0x4011EB8520000000, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0.000000e+00, %bb35 ], [ 0x400A353F80000000, %bb36 ], [ 0x40103D70A0000000, %bb37 ], [ 0x400D1EB860000000, %bb38 ], [ 0x40147EF9E0000000, %bb39 ], [ 0x4014F1AA00000000, %bb40 ], [ 0.000000e+00, %bb41 ], [ 0x400B53F7C0000000, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0x400BC6A7E0000000, %bb44 ], [ 0x4011958100000000, %bb45 ], [ 0x40150E5600000000, %bb46 ], [ 0x4013B645A0000000, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0x4011CED920000000, %bb49 ], [ 0x40137CEDA0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp54 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0.000000e+00, %bb35 ], [ 0x40105A1CA0000000, %bb36 ], [ 0.000000e+00, %bb37 ], [ 0x4012083120000000, %bb38 ], [ 0x40186A7F00000000, %bb39 ], [ 0x4016BC6A80000000, %bb40 ], [ 0.000000e+00, %bb41 ], [ 0x4010CCCCC0000000, %bb42 ], [ 0x401326E980000000, %bb43 ], [ 0x40113F7CE0000000, %bb44 ], [ 0x40139999A0000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0x40169FBE80000000, %bb47 ], [ 0x4014D4FE00000000, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x4017126EA0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp55 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x4012B43960000000, %bb35 ], [ 0x401224DD20000000, %bb36 ], [ 0.000000e+00, %bb37 ], [ 0x40130A3D80000000, %bb38 ], [ 0x401B1A9FC0000000, %bb39 ], [ 0x4018C08320000000, %bb40 ], [ 0x4017126EA0000000, %bb41 ], [ 0.000000e+00, %bb42 ], [ 0x4015D70A40000000, %bb43 ], [ 0.000000e+00, %bb44 ], [ 0x4016D91680000000, %bb45 ], [ 0x40194FDF40000000, %bb46 ], [ 0x4019333340000000, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0x401547AE20000000, %bb49 ], [ 0x4017F7CEE0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp56 = phi float [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x40140C49C0000000, %bb35 ], [ 0x4012083120000000, %bb36 ], [ 0x4016F5C280000000, %bb37 ], [ 0x40149BA5E0000000, %bb38 ], [ 0x401C1CAC00000000, %bb39 ], [ 0x4019DF3B60000000, %bb40 ], [ 0x40186A7F00000000, %bb41 ], [ 0x4012D0E560000000, %bb42 ], [ 0x40169FBE80000000, %bb43 ], [ 0x40137CEDA0000000, %bb44 ], [ 0x4018A3D700000000, %bb45 ], [ 0x401AA7EFA0000000, %bb46 ], [ 0.000000e+00, %bb47 ], [ 0x40183126E0000000, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x401BC6A7E0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp57 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x4011958100000000, %bb35 ], [ 0x400E76C8C0000000, %bb36 ], [ 0x4014624DE0000000, %bb37 ], [ 0x40115C2900000000, %bb38 ], [ 0x401A353F80000000, %bb39 ], [ 0x40176872C0000000, %bb40 ], [ 0x40172F1AA0000000, %bb41 ], [ 0x40109374C0000000, %bb42 ], [ 0x40149BA5E0000000, %bb43 ], [ 0x401122D0E0000000, %bb44 ], [ 0x40152B0200000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0x4016F5C280000000, %bb47 ], [ 0x401445A1C0000000, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x40194FDF40000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp58 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x400D581060000000, %bb35 ], [ 0x400C72B020000000, %bb36 ], [ 0.000000e+00, %bb37 ], [ 0x400D1EB860000000, %bb38 ], [ 0x40147EF9E0000000, %bb39 ], [ 0x4012ED9160000000, %bb40 ], [ 0.000000e+00, %bb41 ], [ 0.000000e+00, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0x400F22D0E0000000, %bb44 ], [ 0x4010E978E0000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0x40125E3540000000, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x40147EF9E0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp59 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x4004D4FE00000000, %bb35 ], [ 0x4007126EA0000000, %bb36 ], [ 0x4006D91680000000, %bb37 ], [ 0x400547AE20000000, %bb38 ], [ 0x400B1A9FC0000000, %bb39 ], [ 0x40094FDF40000000, %bb40 ], [ 0.000000e+00, %bb41 ], [ 0.000000e+00, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0x40086A7F00000000, %bb44 ], [ 0.000000e+00, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0.000000e+00, %bb47 ], [ 0x40083126E0000000, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x400FCED920000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp60 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0.000000e+00, %bb35 ], [ 0x3FFBC6A7E0000000, %bb36 ], [ 0x3FFAE147A0000000, %bb37 ], [ 0x3FF9FBE760000000, %bb38 ], [ 0x40009374C0000000, %bb39 ], [ 0x400020C4A0000000, %bb40 ], [ 0.000000e+00, %bb41 ], [ 0x3FFA6E9780000000, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0x3FFCAC0840000000, %bb44 ], [ 0x3FFE76C8C0000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0.000000e+00, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x40086A7F00000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp61 = phi float [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0x3FF5810620000000, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0.000000e+00, %bb35 ], [ 0x3FF6666660000000, %bb36 ], [ 0x3FF428F5C0000000, %bb37 ], [ 0.000000e+00, %bb38 ], [ 0x3FF9893740000000, %bb39 ], [ 0x3FF6D91680000000, %bb40 ], [ 0x3FF7BE76C0000000, %bb41 ], [ 0x3FF49BA5E0000000, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0.000000e+00, %bb44 ], [ 0x3FF8A3D700000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0.000000e+00, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x40037CEDA0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp62 = phi float [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0x401076C8C0000000, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0.000000e+00, %bb35 ], [ 0x40086A7F00000000, %bb36 ], [ 0x400C395820000000, %bb37 ], [ 0x40094FDF40000000, %bb38 ], [ 0x401178D500000000, %bb39 ], [ 0x40105A1CA0000000, %bb40 ], [ 0x400F22D0E0000000, %bb41 ], [ 0.000000e+00, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0x4009C28F60000000, %bb44 ], [ 0x400DCAC080000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0x4010E978E0000000, %bb47 ], [ 0x400E3D70A0000000, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0x4013439580000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp63 = phi float [ 0x40445999A0000000, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 4.100000e+01, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0x4044E66660000000, %bb30 ], [ 0x4044D999A0000000, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0.000000e+00, %bb35 ], [ 0x4045A66660000000, %bb36 ], [ 0x4045666660000000, %bb37 ], [ 4.350000e+01, %bb38 ], [ 0x4044D999A0000000, %bb39 ], [ 0x40450CCCC0000000, %bb40 ], [ 0x4044333340000000, %bb41 ], [ 0.000000e+00, %bb42 ], [ 4.300000e+01, %bb43 ], [ 0x4045B33340000000, %bb44 ], [ 0x4045333340000000, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0x40448CCCC0000000, %bb47 ], [ 0x4044B33340000000, %bb48 ], [ 0.000000e+00, %bb49 ], [ 0.000000e+00, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp64 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0x400EE978E0000000, %bb26 ], [ 0x400D916880000000, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0.000000e+00, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0.000000e+00, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x40083126E0000000, %bb35 ], [ 0x40094FDF40000000, %bb36 ], [ 0x400B8D4FE0000000, %bb37 ], [ 0x4008A3D700000000, %bb38 ], [ 0x40110624E0000000, %bb39 ], [ 0x40100418A0000000, %bb40 ], [ 0x400CE56040000000, %bb41 ], [ 0x4007BE76C0000000, %bb42 ], [ 0.000000e+00, %bb43 ], [ 0x4009893740000000, %bb44 ], [ 0.000000e+00, %bb45 ], [ 0x401428F5C0000000, %bb46 ], [ 0x4010B020C0000000, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0x400FCED920000000, %bb49 ], [ 0x40115C2900000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp65 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0.000000e+00, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0.000000e+00, %bb28 ], [ 0x4003B645A0000000, %bb29 ], [ 0x40037CEDA0000000, %bb30 ], [ 0x4003B645A0000000, %bb31 ], [ 0x4003EF9DC0000000, %bb32 ], [ 0.000000e+00, %bb33 ], [ 0.000000e+00, %bb34 ], [ 0x3FFFCED920000000, %bb35 ], [ 0x40013F7CE0000000, %bb36 ], [ 0.000000e+00, %bb37 ], [ 0x40009374C0000000, %bb38 ], [ 0x4005F3B640000000, %bb39 ], [ 0x4005810620000000, %bb40 ], [ 0x4003B645A0000000, %bb41 ], [ 0.000000e+00, %bb42 ], [ 0x40010624E0000000, %bb43 ], [ 0x4001EB8520000000, %bb44 ], [ 0x40025E3540000000, %bb45 ], [ 0x40062D0E60000000, %bb46 ], [ 0x4007F7CEE0000000, %bb47 ], [ 0.000000e+00, %bb48 ], [ 0x4007851EC0000000, %bb49 ], [ 0x400B53F7C0000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0.000000e+00, %bb ]
  %tmp66 = phi double [ 0.000000e+00, %bb24 ], [ 0.000000e+00, %bb25 ], [ 0x3FF9FBE760000000, %bb26 ], [ 0.000000e+00, %bb27 ], [ 0x3FFB53F7C0000000, %bb28 ], [ 0x3FF9893740000000, %bb29 ], [ 0.000000e+00, %bb30 ], [ 0x3FF8A3D700000000, %bb31 ], [ 0.000000e+00, %bb32 ], [ 0x3FF9168720000000, %bb33 ], [ 0x3FF49BA5E0000000, %bb34 ], [ 0x3FF6666660000000, %bb35 ], [ 0x3FF8A3D700000000, %bb36 ], [ 0.000000e+00, %bb37 ], [ 0x3FF6666660000000, %bb38 ], [ 0x3FFC395820000000, %bb39 ], [ 0x3FFB53F7C0000000, %bb40 ], [ 0.000000e+00, %bb41 ], [ 0x3FF7BE76C0000000, %bb42 ], [ 0x3FF6D91680000000, %bb43 ], [ 0x3FF83126E0000000, %bb44 ], [ 0.000000e+00, %bb45 ], [ 0.000000e+00, %bb46 ], [ 0x40005A1CA0000000, %bb47 ], [ 0x3FFD1EB860000000, %bb48 ], [ 0x3FFFCED920000000, %bb49 ], [ 0x4001EB8520000000, %bb51 ], [ 0.000000e+00, %bb50 ], [ 0x3FF50E5600000000, %bb ]
  switch i32 %arg5, label %bb70 [
    i32 0, label %bb105
    i32 12, label %bb67
    i32 7, label %bb69
  ]

bb67:                                             ; preds = %bb52
  %tmp68 = fadd float %tmp63, 1.000000e+01
  store float %tmp68, float* %arg23, align 4
  br label %bb105

bb69:                                             ; preds = %bb52
  br label %bb105

bb70:                                             ; preds = %bb52
  %tmp71 = icmp ult i32 %arg3, 16
  br i1 %tmp71, label %bb72, label %bb105

bb72:                                             ; preds = %bb70
  %tmp73 = sitofp i32 %arg3 to float
  %tmp74 = load float* %arg23, align 4
  %tmp75 = fsub float %tmp73, %tmp74
  %tmp76 = fpext float %tmp75 to double
  %tmp77 = fmul double %tmp76, %tmp76
  %tmp78 = fmul double %tmp77, 1.200000e-04
  %tmp79 = fsub double 1.000000e+00, %tmp78
  %tmp80 = fptrunc double %tmp79 to float
  %tmp81 = fadd float %tmp63, 1.000000e+01
  %tmp82 = fsub float %tmp73, %tmp81
  %tmp83 = fpext float %tmp82 to double
  %tmp84 = fmul double %tmp83, %tmp83
  %tmp85 = fmul double %tmp84, 1.200000e-04
  %tmp86 = fsub double 1.000000e+00, %tmp85
  %tmp87 = fptrunc double %tmp86 to float
  %tmp88 = fadd float %tmp63, -2.000000e+01
  %tmp89 = fsub float %tmp73, %tmp88
  %tmp90 = fpext float %tmp89 to double
  %tmp91 = fmul double %tmp90, %tmp90
  %tmp92 = fmul double %tmp91, 1.200000e-04
  %tmp93 = fsub double 1.000000e+00, %tmp92
  %tmp94 = fptrunc double %tmp93 to float
  %tmp95 = fadd float %tmp63, -1.000000e+01
  %tmp96 = fsub float %tmp73, %tmp95
  %tmp97 = fpext float %tmp96 to double
  %tmp98 = fmul double %tmp97, %tmp97
  %tmp99 = fmul double %tmp98, 1.200000e-04
  %tmp100 = fsub double 1.000000e+00, %tmp99
  %tmp101 = fptrunc double %tmp100 to float
  %tmp102 = fpext float %tmp87 to double
  %tmp103 = fpext float %tmp101 to double
  %tmp104 = fpext float %tmp94 to double
  br label %bb105

bb105:                                            ; preds = %bb72, %bb70, %bb69, %bb67, %bb52
  %tmp106 = phi float [ 0.000000e+00, %bb67 ], [ 0x3FF028F5C0000000, %bb69 ], [ 0.000000e+00, %bb72 ], [ 0.000000e+00, %bb70 ], [ 0.000000e+00, %bb52 ]
  %tmp107 = phi float [ %tmp61, %bb67 ], [ %tmp56, %bb69 ], [ 0.000000e+00, %bb72 ], [ 0.000000e+00, %bb70 ], [ %tmp62, %bb52 ]
  %tmp108 = phi float [ 0.000000e+00, %bb67 ], [ 0.000000e+00, %bb69 ], [ %tmp80, %bb72 ], [ 0.000000e+00, %bb70 ], [ 0.000000e+00, %bb52 ]
  %tmp109 = phi double [ 0.000000e+00, %bb67 ], [ 0.000000e+00, %bb69 ], [ %tmp103, %bb72 ], [ 0.000000e+00, %bb70 ], [ 0.000000e+00, %bb52 ]
  %tmp110 = phi double [ 0.000000e+00, %bb67 ], [ 0.000000e+00, %bb69 ], [ %tmp104, %bb72 ], [ 0.000000e+00, %bb70 ], [ 0.000000e+00, %bb52 ]
  %tmp111 = phi double [ 0.000000e+00, %bb67 ], [ 0.000000e+00, %bb69 ], [ %tmp102, %bb72 ], [ 0.000000e+00, %bb70 ], [ 0.000000e+00, %bb52 ]
  %tmp112 = fmul float %tmp106, %tmp107
  %tmp113 = fmul float %tmp112, %tmp108
  %tmp114 = fmul float %tmp113, 5.000000e+01
  %tmp115 = fmul float %tmp114, 7.500000e-01
  %tmp116 = fdiv float %arg, %tmp115
  %tmp117 = tail call float @widget(float %tmp116)
  %tmp118 = fpext float %tmp117 to double
  %tmp119 = fadd double %tmp118, 1.000000e+00
  %tmp120 = fmul double %tmp119, 5.000000e+01
  %tmp121 = fptrunc double %tmp120 to float
  store float %tmp121, float* %arg6, align 4
  %tmp122 = load i32* %arg14, align 4
  %tmp123 = sitofp i32 %tmp122 to float
  %tmp124 = fmul float %tmp123, %arg
  %tmp125 = fdiv float %tmp124, 3.150000e+02
  %tmp126 = tail call float @widget(float %tmp125)
  %tmp127 = fmul float %tmp126, 5.000000e+01
  store float %tmp127, float* %arg8, align 4
  %tmp128 = fcmp ugt float %tmp127, 2.500000e+02
  %tmp129 = fcmp ogt float %tmp127, 2.500000e+02
  %tmp130 = and i1 %tmp128, %tmp129
  br i1 %tmp130, label %bb131, label %bb140

bb131:                                            ; preds = %bb105
  %tmp132 = load i32* %arg7, align 4
  %tmp133 = icmp eq i32 %tmp132, 48
  br i1 %tmp133, label %bb134, label %bb140

bb134:                                            ; preds = %bb131
  %tmp135 = fadd float %tmp127, 3.500000e+02
  %tmp136 = fmul float %tmp135, 4.000000e+00
  %tmp137 = fadd float %tmp136, 0.000000e+00
  %tmp138 = fmul float %tmp137, 0x3FF2666660000000
  %tmp139 = fpext float %tmp138 to double
  br label %bb140

bb140:                                            ; preds = %bb134, %bb131, %bb105
  %tmp141 = phi double [ %tmp139, %bb134 ], [ 0.000000e+00, %bb131 ], [ 0.000000e+00, %bb105 ]
  %tmp142 = tail call double @wombat(double %tmp141)
  %tmp143 = fptrunc double %tmp142 to float
  store float %tmp143, float* %arg10, align 4
  %tmp144 = fmul double %tmp111, 0x4048066666666667
  %tmp145 = load float* %arg6, align 4
  %tmp146 = fpext float %tmp145 to double
  %tmp147 = fmul double %tmp144, %tmp146
  %tmp148 = fmul double %tmp66, %tmp147
  %tmp149 = fmul double %tmp148, 7.500000e-01
  %tmp150 = fmul double %tmp111, 0x404375C28F5C28F5
  %tmp151 = fmul double %tmp150, %tmp146
  %tmp152 = fmul double %tmp65, %tmp151
  %tmp153 = fmul double %tmp152, 7.500000e-01
  %tmp154 = fadd double %tmp149, %tmp153
  %tmp155 = fmul double %tmp109, 0x4042999999999999
  %tmp156 = fmul double %tmp155, %tmp146
  %tmp157 = fmul double %tmp64, %tmp156
  %tmp158 = fmul double %tmp157, 7.500000e-01
  %tmp159 = fadd double %tmp158, %tmp154
  %tmp160 = fmul double %tmp109, 3.210000e+01
  %tmp161 = fmul double %tmp160, %tmp146
  %tmp162 = fmul double %tmp53, %tmp161
  %tmp163 = fmul double %tmp162, 7.500000e-01
  %tmp164 = fadd double %tmp163, %tmp159
  %tmp165 = fmul double %tmp109, 3.162000e+01
  %tmp166 = fmul double %tmp165, %tmp146
  %tmp167 = fmul double %tmp54, %tmp166
  %tmp168 = fmul double %tmp167, 7.500000e-01
  %tmp169 = fadd double %tmp168, %tmp164
  %tmp170 = fmul double %tmp110, 3.030000e+01
  %tmp171 = fmul double %tmp170, %tmp146
  %tmp172 = fmul double %tmp55, %tmp171
  %tmp173 = fmul double %tmp172, 7.500000e-01
  %tmp174 = fadd double %tmp173, %tmp169
  %tmp175 = fmul double %tmp110, 3.131000e+01
  %tmp176 = fmul double %tmp175, %tmp146
  %tmp177 = fpext float %tmp56 to double
  %tmp178 = fmul double %tmp177, %tmp176
  %tmp179 = fmul double %tmp178, 7.500000e-01
  %tmp180 = fadd double %tmp179, %tmp174
  %tmp181 = fmul double %tmp110, 0x4040466666666667
  %tmp182 = fmul double %tmp181, %tmp146
  %tmp183 = fmul double %tmp57, %tmp182
  %tmp184 = fmul double %tmp183, 7.500000e-01
  %tmp185 = fadd double %tmp184, %tmp180
  %tmp186 = fmul double %tmp109, 3.480000e+01
  %tmp187 = fmul double %tmp186, %tmp146
  %tmp188 = fmul double %tmp58, %tmp187
  %tmp189 = fmul double %tmp188, 7.500000e-01
  %tmp190 = fadd double %tmp189, %tmp185
  %tmp191 = fmul double %tmp109, 4.092000e+01
  %tmp192 = fmul double %tmp191, %tmp146
  %tmp193 = fmul double %tmp59, %tmp192
  %tmp194 = fmul double %tmp193, 7.500000e-01
  %tmp195 = fadd double %tmp194, %tmp190
  %tmp196 = fmul double %tmp111, 4.650000e+01
  %tmp197 = fmul double %tmp196, %tmp146
  %tmp198 = fmul double %tmp60, %tmp197
  %tmp199 = fmul double %tmp198, 7.500000e-01
  %tmp200 = fadd double %tmp199, %tmp195
  %tmp201 = fmul double %tmp111, 4.929000e+01
  %tmp202 = fmul double %tmp201, %tmp146
  %tmp203 = fpext float %tmp61 to double
  %tmp204 = fmul double %tmp203, %tmp202
  %tmp205 = fmul double %tmp204, 7.500000e-01
  %tmp206 = fadd double %tmp205, %tmp200
  %tmp207 = fdiv double %tmp206, 1.000000e+03
  %tmp208 = fptrunc double %tmp207 to float
  store float %tmp208, float* %arg12, align 4
  %tmp209 = load float* %arg10, align 4
  %tmp210 = fmul float %tmp209, 1.000000e+02
  %tmp211 = fmul float %tmp208, 2.000000e+01
  %tmp212 = fdiv float %tmp210, %tmp211
  store float %tmp212, float* %arg13, align 4
  %tmp213 = load float* %arg18, align 4
  %tmp214 = fcmp olt float %tmp213, 1.500000e+00
  br i1 %tmp214, label %bb217, label %bb215

bb215:                                            ; preds = %bb224, %bb140
  %tmp216 = load float* %arg15, align 4
  br label %bb228

bb217:                                            ; preds = %bb224, %bb140
  %tmp218 = phi float [ %tmp225, %bb224 ], [ %tmp213, %bb140 ]
  %tmp219 = phi i32 [ %tmp226, %bb224 ], [ 0, %bb140 ]
  %tmp220 = getelementptr inbounds [16 x float]* @global, i32 0, i32 %tmp219
  %tmp221 = load float* %tmp220, align 4
  %tmp222 = fcmp ogt float %tmp221, %tmp218
  br i1 %tmp222, label %bb223, label %bb224

bb223:                                            ; preds = %bb217
  store float %tmp221, float* %arg18, align 4
  br label %bb224

bb224:                                            ; preds = %bb223, %bb217
  %tmp225 = phi float [ %tmp218, %bb217 ], [ %tmp221, %bb223 ]
  %tmp226 = add nsw i32 %tmp219, 1
  %tmp227 = icmp eq i32 %tmp219, 15
  br i1 %tmp227, label %bb215, label %bb217

bb228:                                            ; preds = %bb235, %bb215
  %tmp229 = phi float [ %tmp216, %bb215 ], [ %tmp236, %bb235 ]
  %tmp230 = phi i32 [ 0, %bb215 ], [ %tmp237, %bb235 ]
  %tmp231 = getelementptr inbounds [16 x float]* @global, i32 0, i32 %tmp230
  %tmp232 = load float* %tmp231, align 4
  %tmp233 = fcmp ogt float %tmp232, %tmp229
  br i1 %tmp233, label %bb234, label %bb235

bb234:                                            ; preds = %bb228
  store float %tmp232, float* %arg15, align 4
  br label %bb235

bb235:                                            ; preds = %bb234, %bb228
  %tmp236 = phi float [ %tmp229, %bb228 ], [ %tmp232, %bb234 ]
  %tmp237 = add nsw i32 %tmp230, 1
  %tmp238 = icmp eq i32 %tmp230, 15
  br i1 %tmp238, label %bb239, label %bb228

bb239:                                            ; preds = %bb235
  ret void
}

declare double @wombat(double)

declare float @widget(float)
