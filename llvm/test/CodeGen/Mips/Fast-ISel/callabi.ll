; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort=1 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s -check-prefix=mips32r2
; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort=1 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s -check-prefix=mips32
; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s -check-prefix=CHECK2
; RUN: llc -march=mipsel -relocation-model=pic -O0 -mips-fast-isel -fast-isel-abort=1 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s -check-prefix=CHECK2


@c1 = global i8 -45, align 1
@uc1 = global i8 27, align 1
@s1 = global i16 -1789, align 2
@us1 = global i16 1256, align 2

; Function Attrs: nounwind
define void @cxi() #0 {
entry:
; CHECK-LABEL:  cxi
  call void @xi(i32 10)
; CHECK-DAG:    addiu   $4, $zero, 10
; CHECK-DAG:    lw      $25, %got(xi)(${{[0-9]+}})
; CHECK:        jalr    $25

  ret void
}

declare void @xi(i32) #1

; Function Attrs: nounwind
define void @cxii() #0 {
entry:
; CHECK-LABEL:  cxii
  call void @xii(i32 746, i32 892)
; CHECK-DAG:    addiu   $4, $zero, 746
; CHECK-DAG:    addiu   $5, $zero, 892
; CHECK-DAG:    lw      $25, %got(xii)(${{[0-9]+}})
; CHECK:        jalr    $25

  ret void
}

declare void @xii(i32, i32) #1

; Function Attrs: nounwind
define void @cxiii() #0 {
entry:
; CHECK-LABEL:  cxiii
  call void @xiii(i32 88, i32 44, i32 11)
; CHECK-DAG:    addiu   $4, $zero, 88
; CHECK-DAG:    addiu   $5, $zero, 44
; CHECK-DAG:    addiu   $6, $zero, 11
; CHECK-DAG:    lw      $25, %got(xiii)(${{[0-9]+}})
; CHECK:        jalr    $25
  ret void
}

declare void @xiii(i32, i32, i32) #1

; Function Attrs: nounwind
define void @cxiiii() #0 {
entry:
; CHECK-LABEL:  cxiiii
  call void @xiiii(i32 167, i32 320, i32 97, i32 14)
; CHECK-DAG:    addiu   $4, $zero, 167
; CHECK-DAG:    addiu   $5, $zero, 320
; CHECK-DAG:    addiu   $6, $zero, 97
; CHECK-DAG:    addiu   $7, $zero, 14
; CHECK-DAG:    lw      $25, %got(xiiii)(${{[0-9]+}})
; CHECK:        jalr    $25

  ret void
}

declare void @xiiii(i32, i32, i32, i32) #1

; Function Attrs: nounwind
define void @cxiiiiconv() #0 {
entry:
; CHECK-LABEL: cxiiiiconv
; mips32r2-LABEL:  cxiiiiconv
; mips32-LABEL:  cxiiiiconv
  %0 = load i8, i8* @c1, align 1
  %conv = sext i8 %0 to i32
  %1 = load i8, i8* @uc1, align 1
  %conv1 = zext i8 %1 to i32
  %2 = load i16, i16* @s1, align 2
  %conv2 = sext i16 %2 to i32
  %3 = load i16, i16* @us1, align 2
  %conv3 = zext i16 %3 to i32
  call void @xiiii(i32 %conv, i32 %conv1, i32 %conv2, i32 %conv3)
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32r2:     addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32:       addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32r2-DAG:         lw      $[[REG_C1_ADDR:[0-9]+]], %got(c1)($[[REG_GP]])
; mips32r2-DAG: lbu     $[[REG_C1:[0-9]+]], 0($[[REG_C1_ADDR]])
; mips32r2-DAG  seb     $3, $[[REG_C1]]
; mips32-DAG:   lw      $[[REG_C1_ADDR:[0-9]+]], %got(c1)($[[REG_GP]])
; mips32-DAG:   lbu     $[[REG_C1:[0-9]+]], 0($[[REG_C1_ADDR]])
; mips32-DAG:   sll     $[[REG_C1_1:[0-9]+]], $[[REG_C1]], 24
; mips32-DAG:   sra     $4, $[[REG_C1_1]], 24
; CHECK-DAG:    lw      $[[REG_UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[REG_UC1:[0-9]+]], 0($[[REG_UC1_ADDR]])
; FIXME andi is superfulous
; CHECK-DAG:    andi    $5, $[[REG_UC1]], 255
; mips32r2-DAG:         lw      $[[REG_S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
; mips32r2-DAG: lhu     $[[REG_S1:[0-9]+]], 0($[[REG_S1_ADDR]])
; mips32r2-DAG: seh     $6, $[[REG_S1]]
; mips32-DAG:   lw      $[[REG_S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
; mips32-DAG:   lhu     $[[REG_S1:[0-9]+]], 0($[[REG_S1_ADDR]])
; mips32-DAG:   sll     $[[REG_S1_1:[0-9]+]], $[[REG_S1]], 16
; mips32-DAG:   sra     $6, $[[REG_S1_1]], 16
; CHECK-DAG:    lw      $[[REG_US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[REG_US1:[0-9]+]], 0($[[REG_US1_ADDR]])
; FIXME andi is superfulous
; CHECK-DAG:    andi    $7, $[[REG_US1]], 65535
; mips32r2:     jalr    $25
; mips32r2:     jalr    $25
; CHECK:        jalr    $25
  ret void
}

; Function Attrs: nounwind
define void @cxf() #0 {
entry:
; CHECK-LABEL:  cxf
  call void @xf(float 0x40BBC85560000000)
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK:        lui     $[[REG_FPCONST_1:[0-9]+]], 17886
; CHECK:        ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 17067
; CHECK: mtc1   $[[REG_FPCONST]], $f12
; CHECK:        lw      $25, %got(xf)($[[REG_GP]])
; CHECK:        jalr    $25
  ret void
}

declare void @xf(float) #1

; Function Attrs: nounwind
define void @cxff() #0 {
entry:
; CHECK-LABEL:  cxff
  call void @xff(float 0x3FF74A6CA0000000, float 0x401A2C0840000000)
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    lui     $[[REG_FPCONST_1:[0-9]+]], 16314
; CHECK-DAG:    ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 21349
; CHECK-DAG: mtc1       $[[REG_FPCONST]], $f12
; CHECK-DAG:    lui     $[[REG_FPCONST_2:[0-9]+]], 16593
; CHECK-DAG:    ori     $[[REG_FPCONST_3:[0-9]+]], $[[REG_FPCONST_2]], 24642
; CHECK-DAG: mtc1       $[[REG_FPCONST_3]], $f14
; CHECK:        lw      $25, %got(xff)($[[REG_GP]])
; CHECK:        jalr    $25
  ret void
}

declare void @xff(float, float) #1

; Function Attrs: nounwind
define void @cxfi() #0 {
entry:
; CHECK-LABEL: cxfi
  call void @xfi(float 0x4013906240000000, i32 102)
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    lui     $[[REG_FPCONST_1:[0-9]+]], 16540
; CHECK-DAG:    ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 33554
; CHECK-DAG: mtc1       $[[REG_FPCONST]], $f12
; CHECK-DAG:    addiu   $5, $zero, 102
; CHECK:        lw      $25, %got(xfi)($[[REG_GP]])
; CHECK:        jalr    $25

  ret void
}

declare void @xfi(float, i32) #1

; Function Attrs: nounwind
define void @cxfii() #0 {
entry:
; CHECK-LABEL: cxfii
  call void @xfii(float 0x405EC7EE00000000, i32 9993, i32 10922)
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    lui     $[[REG_FPCONST_1:[0-9]+]], 17142
; CHECK-DAG:    ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 16240
; CHECK-DAG: mtc1       $[[REG_FPCONST]], $f12
; CHECK-DAG:    addiu   $5, $zero, 9993
; CHECK-DAG:    addiu   $6, $zero, 10922
; CHECK:        lw      $25, %got(xfii)($[[REG_GP]])
; CHECK:        jalr    $25
  ret void
}

declare void @xfii(float, i32, i32) #1

; Function Attrs: nounwind
define void @cxfiii() #0 {
entry:
; CHECK-LABEL: cxfiii
  call void @xfiii(float 0x405C072B20000000, i32 3948, i32 89011, i32 111222)
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    lui     $[[REG_FPCONST_1:[0-9]+]], 17120
; CHECK-DAG:    ori     $[[REG_FPCONST:[0-9]+]], $[[REG_FPCONST_1]], 14681
; CHECK-DAG: mtc1       $[[REG_FPCONST]], $f12
; CHECK-DAG:    addiu   $5, $zero, 3948
; CHECK-DAG:    lui     $[[REG_I_1:[0-9]+]], 1
; CHECK-DAG:    ori     $6, $[[REG_I_1]], 23475
; CHECK-DAG:    lui     $[[REG_I_2:[0-9]+]], 1
; CHECK-DAG:    ori     $7, $[[REG_I_2]], 45686
; CHECK:        lw      $25, %got(xfiii)($[[REG_GP]])
; CHECK:        jalr    $25
  ret void
}

declare void @xfiii(float, i32, i32, i32) #1

; Function Attrs: nounwind
define void @cxd() #0 {
entry:
; mips32r2-LABEL: cxd:
; mips32-LABEL: cxd:
  call void @xd(double 5.994560e+02)
; mips32:       addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32-DAG:   lui     $[[REG_FPCONST_1:[0-9]+]], 16514
; mips32-DAG:   ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 48037
; mips32-DAG:   lui     $[[REG_FPCONST_3:[0-9]+]], 58195
; mips32-DAG:   ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 63439
; mips32-DAG:    mtc1   $[[REG_FPCONST_4]], $f12
; mips32-DAG:       mtc1        $[[REG_FPCONST_2]], $f13
; mips32-DAG:   lw      $25, %got(xd)($[[REG_GP]])
; mips32:       jalr    $25
; mips32r2:     addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32r2-DAG: lui     $[[REG_FPCONST_1:[0-9]+]], 16514
; mips32r2-DAG: ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 48037
; mips32r2-DAG: lui     $[[REG_FPCONST_3:[0-9]+]], 58195
; mips32r2-DAG: ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 63439
; mips32r2-DAG: mtc1    $[[REG_FPCONST_4]], $f12
; mips32r2-DAG: mthc1   $[[REG_FPCONST_2]], $f12
; mips32r2-DAG: lw      $25, %got(xd)($[[REG_GP]])
; mips32r2 :    jalr    $25
  ret void
}

declare void @xd(double) #1

; Function Attrs: nounwind
define void @cxdd() #0 {
; mips32r2-LABEL: cxdd:
; mips32-LABEL: cxdd:
entry:
  call void @xdd(double 1.234980e+03, double 0x40F5B331F7CED917)
; mips32:       addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32-DAG:   lui     $[[REG_FPCONST_1:[0-9]+]], 16531
; mips32-DAG:   ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 19435
; mips32-DAG:   lui     $[[REG_FPCONST_3:[0-9]+]], 34078
; mips32-DAG:   ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 47186
; mips32-DAG:   mtc1    $[[REG_FPCONST_4]], $f12
; mips32-DAG:   mtc1    $[[REG_FPCONST_2]], $f13
; mips32-DAG:   lui     $[[REG_FPCONST_1:[0-9]+]], 16629
; mips32-DAG:   ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 45873
; mips32-DAG:   lui     $[[REG_FPCONST_3:[0-9]+]], 63438
; mips32-DAG:   ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 55575
; mips32-DAG:   mtc1    $[[REG_FPCONST_4]], $f14
; mips32-DAG:   mtc1    $[[REG_FPCONST_2]], $f15
; mips32-DAG:   lw      $25, %got(xdd)($[[REG_GP]])
; mips32:       jalr    $25
; mips32r2:     addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; mips32r2-DAG: lui     $[[REG_FPCONST_1:[0-9]+]], 16531
; mips32r2-DAG: ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 19435
; mips32r2-DAG: lui     $[[REG_FPCONST_3:[0-9]+]], 34078
; mips32r2-DAG: ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 47186
; mips32r2-DAG: mtc1    $[[REG_FPCONST_4]], $f12
; mips32r2-DAG: mthc1   $[[REG_FPCONST_2]], $f12
; mips32r2-DAG: lui     $[[REG_FPCONST_1:[0-9]+]], 16629
; mips32r2-DAG: ori     $[[REG_FPCONST_2:[0-9]+]], $[[REG_FPCONST_1]], 45873
; mips32r2-DAG: lui     $[[REG_FPCONST_3:[0-9]+]], 63438
; mips32r2-DAG: ori     $[[REG_FPCONST_4:[0-9]+]], $[[REG_FPCONST_3]], 55575
; mips32r2-DAG: mtc1    $[[REG_FPCONST_4]], $f14
; mips32r2-DAG: mthc1   $[[REG_FPCONST_2]], $f14
; mips32r2-DAG: lw      $25, %got(xdd)($[[REG_GP]])
; mips32r2 :    jalr    $25
  ret void
}

declare void @xdd(double, double) #1

; Function Attrs: nounwind
define void @cxif() #0 {
entry:
; CHECK-LABEL: cxif:
  call void @xif(i32 345, float 0x407BCE5A20000000)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    addiu   $4, $zero, 345
; CHECK-DAG:    lui     $[[REGF_1:[0-9]+]], 17374
; CHECK-DAG:    ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 29393
; CHECK-DAG:    mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
; CHECK-DAG:    mfc1    $5, $f[[REGF_3]]
; CHECK-DAG:    lw      $25, %got(xif)($[[REG_GP]])
; CHECK:        jalr    $25

  ret void
}

declare void @xif(i32, float) #1

; Function Attrs: nounwind
define void @cxiff() #0 {
entry:
; CHECK-LABEL: cxiff:
; CHECK2-LABEL: cxiff:
  call void @xiff(i32 12239, float 0x408EDB3340000000, float 0x4013FFE5C0000000)
; We need to do the two floating point parameters in a separate
; check because we can't control the ordering of parts of the sequence
;;
; CHECK:        addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK:        addiu   $4, $zero, 12239
; CHECK2:       addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK2:       addiu   $4, $zero, 12239
; CHECK:        lui     $[[REGF_1:[0-9]+]], 17526
; CHECK:        ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 55706
; CHECK:        mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
; CHECK:        mfc1    $5, $f[[REGF_3]]
; CHECK2:       lui     $[[REGF2_1:[0-9]+]], 16543
; CHECK2:       ori     $[[REGF2_2:[0-9]+]], $[[REGF2_1]], 65326
; CHECK2:       mtc1    $[[REGF2_2]], $f[[REGF2_3:[0-9]+]]
; CHECK2:       mfc1    $6, $f[[REGF2_3]]
; CHECK:        lw      $25, %got(xiff)($[[REG_GP]])
; CHECK2:       lw      $25, %got(xiff)($[[REG_GP]])
; CHECK:        jalr    $25
; CHECK2:       jalr    $25
  ret void
}

declare void @xiff(i32, float, float) #1

; Function Attrs: nounwind
define void @cxifi() #0 {
entry:
; CHECK: cxifi:
  call void @xifi(i32 887, float 0x402277CEE0000000, i32 888)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    addiu   $4, $zero, 887
; CHECK-DAG:    lui     $[[REGF_1:[0-9]+]], 16659
; CHECK-DAG:    ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 48759
; CHECK-DAG:    mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
; CHECK-DAG:    mfc1    $5, $f[[REGF_3]]
; CHECk-DAG:    addiu   $6, $zero, 888
; CHECK-DAG:    lw      $25, %got(xifi)($[[REG_GP]])
; CHECK:        jalr    $25

  ret void
}

declare void @xifi(i32, float, i32) #1

; Function Attrs: nounwind
define void @cxifif() #0 {
entry:
; CHECK: cxifif:
; CHECK2: cxifif:
  call void @xifif(i32 67774, float 0x408EE0FBE0000000, i32 9991, float 0x40B15C8CC0000000)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    lui     $[[REGI:[0-9]+]], 1
; CHECK-DAG:    ori     $4, $[[REGI]], 2238
; CHECK-DAG:    lui     $[[REGF_1:[0-9]+]], 17527
; CHECK-DAG:    ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 2015
; CHECK-DAG:    mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
; CHECK-DAG:    mfc1    $5, $f[[REGF_3]]
; CHECk-DAG:    addiu   $6, $zero, 888
; CHECK2:       lui     $[[REGF2_1:[0-9]+]], 17802
; CHECK2:       ori     $[[REGF2_2:[0-9]+]], $[[REGF2_1]], 58470
; CHECK2:       mtc1    $[[REGF2_2]], $f[[REGF2_3:[0-9]+]]
; CHECK2:       mfc1    $7, $f[[REGF2_3]]
; CHECK:        lw      $25, %got(xifif)($[[REG_GP]])
; CHECK2:       lw      $25, %got(xifif)($[[REG_GP]])
; CHECK2:       jalr    $25
; CHECK:        jalr    $25

  ret void
}

declare void @xifif(i32, float, i32, float) #1

; Function Attrs: nounwind
define void @cxiffi() #0 {
entry:
; CHECK-label: cxiffi:
; CHECK2-label: cxiffi:
  call void @xiffi(i32 45, float 0x3FF6666660000000, float 0x408F333340000000, i32 234)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    addiu   $4, $zero, 45
; CHECK2-DAG:   addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK2-DAG:   addiu   $4, $zero, 45
; CHECK-DAG:    lui     $[[REGF_1:[0-9]+]], 16307
; CHECK-DAG:    ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 13107
; CHECK-DAG:    mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
; CHECK-DAG:    mfc1    $5, $f[[REGF_3]]
; CHECK2:       lui     $[[REGF2_1:[0-9]+]], 17529
; CHECK2:       ori     $[[REGF2_2:[0-9]+]], $[[REGF2_1]], 39322
; CHECK2:       mtc1    $[[REGF2_2]], $f[[REGF2_3:[0-9]+]]
; CHECK2:       mfc1    $6, $f[[REGF2_3]]
; CHECK-DAG:    lw      $25, %got(xiffi)($[[REG_GP]])
; CHECK-DAG:    addiu   $7, $zero, 234
; CHECK2-DAG:   lw      $25, %got(xiffi)($[[REG_GP]])
; CHECK:        jalr    $25
; CHECK2:       jalr    $25

  ret void
}

declare void @xiffi(i32, float, float, i32) #1

; Function Attrs: nounwind
define void @cxifii() #0 {
entry:
; CHECK-DAG:    cxifii:
  call void @xifii(i32 12239, float 0x408EDB3340000000, i32 998877, i32 1234)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], ${{[0-9]+}}, ${{[0-9+]}}
; CHECK-DAG:    addiu   $4, $zero, 12239
; CHECK-DAG:    lui     $[[REGF_1:[0-9]+]], 17526
; CHECK-DAG:    ori     $[[REGF_2:[0-9]+]], $[[REGF_1]], 55706
; CHECK-DAG:    mtc1    $[[REGF_2]], $f[[REGF_3:[0-9]+]]
; CHECK-DAG:    mfc1    $5, $f[[REGF_3]]
; CHECK-DAG:    lui     $[[REGI2:[0-9]+]], 15
; CHECK-DAG:    ori     $6, $[[REGI2]], 15837
; CHECk-DAG:    addiu   $7, $zero, 1234
; CHECK-DAG:    lw      $25, %got(xifii)($[[REG_GP]])
; CHECK:        jalr    $25
  ret void
}

declare void @xifii(i32, float, i32, i32) #1

; FIXME: this function will not pass yet. 
; Function Attrs: nounwind
; define void @cxfid() #0 {
;entry:
;  call void @xfid(float 0x4013B851E0000000, i32 811123, double 0x40934BFF487FCB92)
;  ret void
;}

declare void @xfid(float, i32, double) #1

; Function Attrs: nounwind
define void @g() #0 {
entry:
  call void @cxi()
  call void @cxii()
  call void @cxiii()
  call void @cxiiii()
  call void @cxiiiiconv()
  call void @cxf()
  call void @cxff()
  call void @cxd()
  call void @cxfi()
  call void @cxfii()
  call void @cxfiii()
  call void @cxdd()
  call void @cxif()
  call void @cxiff()
  call void @cxifi()
  call void @cxifii()
  call void @cxifif()
  call void @cxiffi()
  ret void
}


attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.6.0 (gitosis@dmz-portal.mips.com:clang 43992fe7b17de5553ac06d323cb80cc6723a9ae3) (gitosis@dmz-portal.mips.com:llvm.git 0834e6839eb170197c81bb02e916258d1527e312)"}
