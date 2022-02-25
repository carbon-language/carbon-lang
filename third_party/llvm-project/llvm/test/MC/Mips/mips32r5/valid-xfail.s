# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r5 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

        .set noat
        abs.ps          $f22,$f8
        add.ps          $f25,$f27,$f13
        addqh.w         $s7,$s7,$k1
        addqh_r.w       $8,$v1,$zero
        alnv.ps         $f12,$f18,$f30,$12
        c.eq.ps         $fcc5,$f0,$f9
        c.f.ps          $fcc6,$f11,$f11
        c.le.ps         $fcc1,$f7,$f20
        c.lt.ps         $f19,$f5
        c.nge.ps        $f1,$f26
        c.ngl.ps        $f21,$f30
        c.ngle.ps       $fcc7,$f12,$f20
        c.ngt.ps        $fcc5,$f30,$f6
        c.ole.ps        $fcc7,$f21,$f8
        c.olt.ps        $fcc3,$f7,$f16
        c.seq.ps        $fcc6,$f31,$f14
        c.sf.ps         $fcc6,$f4,$f6
        c.ueq.ps        $fcc1,$f5,$f29
        c.ule.ps        $fcc6,$f17,$f3
        c.ult.ps        $fcc7,$f14,$f0
        c.un.ps         $fcc4,$f2,$f26
        ceil.l.d        $f1,$f3
        ceil.l.s        $f18,$f13
        cfcmsa          $s6,$19
        ctcmsa          $31,$s7
        cvt.d.l         $f4,$f16
        cvt.ps.s        $f3,$f18,$f19
        cvt.s.l         $f15,$f30
        cvt.s.pl        $f30,$f1
        cvt.s.pu        $f14,$f25
        dmt $k0
        dvpe            $s6
        emt $8
        evpe            $v0
        floor.l.d       $f26,$f7
        floor.l.s       $f12,$f5
        fork            $s2,$8,$a0
        iret
        lbe             $14,122($9)
        lbue            $11,-108($10)
        lhe             $s6,219($v1)
        lhue            $gp,118($11)
        lle             $gp,-237($ra)
        lwe             $ra,-145($14)
        lwle            $11,-42($11)
        lwre            $sp,-152($24)
        madd.ps         $f22,$f3,$f14,$f3
        mfgc0           $s6,c0_datahi1
        mov.ps          $f22,$f17
        movf.ps         $f10,$f28,$fcc6
        movn.ps         $f31,$f31,$s3
        movt.ps         $f20,$f25,$fcc2
        movz.ps         $f18,$f17,$ra
        msub.ps         $f12,$f14,$f29,$f17
        mtc0            $9,c0_datahi1
        mtgc0           $s4,$21,7
        mul.ps          $f14,$f0,$f16
        neg.ps          $f19,$f13
        nmadd.ps        $f27,$f4,$f9,$f25
        nmsub.ps        $f6,$f12,$f14,$f17
        pll.ps          $f25,$f9,$f30
        plu.ps          $f1,$f26,$f29
        preceq.w.phl    $s8,$gp
        preceq.w.phr    $s5,$15
        pul.ps          $f9,$f30,$f26
        puu.ps          $f24,$f9,$f2
        rdpgpr          $s3,$9
        rorv            $13,$a3,$s5
        round.l.d       $f12,$f1
        round.l.s       $f25,$f5
        sbe             $s7,33($s1)
        sce             $sp,189($10)
        she             $24,105($v0)
        sub.ps          $f5,$f14,$f26
        swe             $24,94($k0)
        swle            $v1,-209($gp)
        swre            $k0,-202($s2)
        tlbginv
        tlbginvf
        tlbgp
        tlbgr
        tlbgwi
        tlbgwr
        tlbinv
        tlbinvf
        trunc.l.d       $f23,$f23
        trunc.l.s       $f28,$f31
        wrpgpr          $zero,$13
        yield           $v1,$s0
