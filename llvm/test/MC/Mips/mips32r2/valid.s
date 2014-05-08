# Instructions that are valid
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s

        .set noat
        abs.d     $f7,$f25             # CHECK: encoding:
        abs.s     $f9,$f16
        add       $s7,$s2,$a1
        add.d     $f1,$f7,$f29
        add.s     $f8,$f21,$f24
        addi      $13,$9,26322
        addu      $9,$a0,$a2
        and       $s7,$v0,$12
        c.ngl.d   $f29,$f29
        c.ngle.d  $f0,$f16
        c.sf.d    $f30,$f0
        c.sf.s    $f14,$f22
        ceil.w.d  $f11,$f25
        ceil.w.s  $f6,$f20
        cfc1      $s1,$21
        clo       $11,$a1
        clz       $sp,$gp
        ctc1      $a2,$26
        cvt.d.s   $f22,$f28
        cvt.d.w   $f26,$f11
        cvt.l.d   $f24,$f15
        cvt.l.s   $f11,$f29
        cvt.s.d   $f26,$f8
        cvt.s.w   $f22,$f15
        cvt.w.d   $f20,$f14
        cvt.w.s   $f20,$f24
        deret
        di        $s8
        div       $zero,$25,$11
        div.d     $f29,$f20,$f27
        div.s     $f4,$f5,$f15
        divu      $zero,$25,$15
        ehb                            # CHECK: ehb # encoding:  [0x00,0x00,0x00,0xc0]
        ei        $14
        eret
        floor.w.d $f14,$f11
        floor.w.s $f8,$f9
        lb        $24,-14515($10)
        lbu       $8,30195($v1)
        ldc1      $f11,16391($s0)
        ldc2      $8,-21181($at)
        ldxc1     $f8,$s7($15)
        lh        $11,-8556($s5)
        lhu       $s3,-22851($v0)
        li        $at,-29773
        li        $zero,-29889
        ll        $v0,-7321($s2)
        luxc1     $f19,$s6($s5)
        lw        $8,5674($a1)
        lwc1      $f16,10225($k0)
        lwc2      $18,-841($a2)
        lwl       $s4,-4231($15)
        lwr       $zero,-19147($gp)
        lwxc1     $f12,$s1($s8)
        madd      $s6,$13
        madd      $zero,$9
        madd.d    $f18,$f19,$f26,$f20
        madd.s    $f1,$f31,$f19,$f25
        maddu     $s3,$gp
        maddu     $24,$s2
        mfc0      $a2,$14,1
        mfc1      $a3,$f27
        mfhc1     $s8,$f24
        mfhi      $s3
        mfhi      $sp
        mflo      $s1
        mov.d     $f20,$f14
        mov.s     $f2,$f27
        move      $s8,$a0
        move      $25,$a2
        movf      $gp,$8,$fcc7
        movf.d    $f6,$f11,$fcc5
        movf.s    $f23,$f5,$fcc6
        movn      $v1,$s1,$s0
        movn.d    $f27,$f21,$k0
        movn.s    $f12,$f0,$s7
        movt      $zero,$s4,$fcc5
        movt.d    $f0,$f2,$fcc0
        movt.s    $f30,$f2,$fcc1
        movz      $a1,$s6,$9
        movz.d    $f12,$f29,$9
        movz.s    $f25,$f7,$v1
        msub      $s7,$k1
        msub.d    $f10,$f1,$f31,$f18
        msub.s    $f12,$f19,$f10,$f16
        msubu     $15,$a1
        mtc0      $9,$29,3
        mtc1      $s8,$f9
        mthc1     $zero,$f16
        mthi      $s1
        mtlo      $sp
        mtlo      $25
        mul       $s0,$s4,$at
        mul.d     $f20,$f20,$f16
        mul.s     $f30,$f10,$f2
        mult      $sp,$s4
        mult      $sp,$v0
        multu     $gp,$k0
        multu     $9,$s2
        negu      $2                   # CHECK: negu $2, $2            # encoding: [0x00,0x02,0x10,0x23]
        negu      $2,$3                # CHECK: negu $2, $3            # encoding: [0x00,0x03,0x10,0x23]
        neg.d     $f27,$f18
        neg.s     $f1,$f15
        nmadd.d   $f18,$f9,$f14,$f19
        nmadd.s   $f0,$f5,$f25,$f12
        nmsub.d   $f30,$f8,$f16,$f30
        nmsub.s   $f1,$f24,$f19,$f4
        nop
        nor       $a3,$zero,$a3
        or        $12,$s0,$sp
        pause                          # CHECK: pause # encoding:  [0x00,0x00,0x01,0x40]
        rdhwr     $sp,$11              
        rotr      $1,15                # CHECK: rotr $1, $1, 15         # encoding: [0x00,0x21,0x0b,0xc2]
        rotr      $1,$14,15            # CHECK: rotr $1, $14, 15        # encoding: [0x00,0x2e,0x0b,0xc2]
        rotrv     $1,$14,$15           # CHECK: rotrv $1, $14, $15      # encoding: [0x01,0xee,0x08,0x46]
        round.w.d $f6,$f4
        round.w.s $f27,$f28
        sb        $s6,-19857($14)
        sc        $15,18904($s3)
        sdc1      $f31,30574($13)
        sdc2      $20,23157($s2)
        sdxc1     $f11,$10($14)
        seb       $25,$15
        seh       $v1,$12
        sh        $14,-6704($15)
        sll       $a3,18               # CHECK: sll $7, $7, 18         # encoding: [0x00,0x07,0x3c,0x80]
        sll       $a3,$zero,18         # CHECK: sll $7, $zero, 18      # encoding: [0x00,0x00,0x3c,0x80]
        sll       $a3,$zero,$9         # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
        sllv      $a3,$zero,$9         # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
        slt       $s7,$11,$k1          # CHECK: slt $23, $11, $27      # encoding: [0x01,0x7b,0xb8,0x2a]
        slti      $s1,$10,9489         # CHECK: slti $17, $10, 9489    # encoding: [0x29,0x51,0x25,0x11]
        sltiu     $25,$25,-15531       # CHECK: sltiu $25, $25, -15531 # encoding: [0x2f,0x39,0xc3,0x55]
        sltu      $s4,$s5,$11          # CHECK: sltu  $20, $21, $11    # encoding: [0x02,0xab,0xa0,0x2b]
        sltu      $24,$25,-15531       # CHECK: sltiu $24, $25, -15531 # encoding: [0x2f,0x38,0xc3,0x55]
        sqrt.d    $f17,$f22
        sqrt.s    $f0,$f1
        sra       $s1,15               # CHECK: sra $17, $17, 15       # encoding: [0x00,0x11,0x8b,0xc3]
        sra       $s1,$s7,15           # CHECK: sra $17, $23, 15       # encoding: [0x00,0x17,0x8b,0xc3]
        srav      $s1,$s7,$sp          # CHECK: srav $17, $23, $sp     # encoding: [0x03,0xb7,0x88,0x07]
        srl       $2,7                 # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
        srl       $2,$2,7              # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
        srl       $25,$s4,$a0          # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
        srlv      $25,$s4,$a0          # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
        ssnop                          # CHECK: ssnop                  # encoding: [0x00,0x00,0x00,0x40]
        sub       $s6,$s3,$12
        sub.d     $f18,$f3,$f17
        sub.s     $f23,$f22,$f22
        subu      $sp,$s6,$s6
        suxc1     $f12,$k1($13)
        sw        $ra,-10160($sp)
        swc1      $f6,-8465($24)
        swc2      $25,24880($s0)
        swl       $15,13694($s3)
        swr       $s1,-26590($14)
        swxc1     $f19,$12($k0)
        teqi      $s5,-17504
        tgei      $s1,5025
        tgeiu     $sp,-28621
        tlbp                           # CHECK: tlbp                   # encoding: [0x42,0x00,0x00,0x08]
        tlbr                           # CHECK: tlbr                   # encoding: [0x42,0x00,0x00,0x01]
        tlbwi                          # CHECK: tlbwi                  # encoding: [0x42,0x00,0x00,0x02]
        tlbwr                          # CHECK: tlbwr                  # encoding: [0x42,0x00,0x00,0x06]
        tlti      $14,-21059
        tltiu     $ra,-5076
        tnei      $12,-29647
        trunc.w.d $f22,$f15
        trunc.w.s $f28,$f30
        wsbh      $k1,$9
        xor       $s2,$a0,$s8
