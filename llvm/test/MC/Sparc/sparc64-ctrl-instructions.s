! RUN: llvm-mc %s -triple=sparc64-unknown-linux-gnu -show-encoding | FileCheck %s


        ! CHECK: bne %xcc, .BB0     ! encoding: [0x12,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne %xcc, .BB0

        ! CHECK: be %xcc, .BB0      ! encoding: [0x02,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be %xcc, .BB0

        ! CHECK: bg %xcc, .BB0      ! encoding: [0x14,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg %xcc, .BB0

        ! CHECK: ble %xcc, .BB0      ! encoding: [0x04,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble %xcc, .BB0

        ! CHECK: bge %xcc, .BB0      ! encoding: [0x16,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge %xcc, .BB0

        ! CHECK: bl %xcc, .BB0      ! encoding: [0x06,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl %xcc, .BB0

        ! CHECK: bgu %xcc, .BB0      ! encoding: [0x18,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu %xcc, .BB0

        ! CHECK: bleu %xcc, .BB0      ! encoding: [0x08,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu %xcc, .BB0

        ! CHECK: bcc %xcc, .BB0      ! encoding: [0x1a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc %xcc, .BB0

        ! CHECK: bcs %xcc, .BB0      ! encoding: [0x0a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs %xcc, .BB0

        ! CHECK: bpos %xcc, .BB0      ! encoding: [0x1c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos %xcc, .BB0

        ! CHECK: bneg %xcc, .BB0      ! encoding: [0x0c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg %xcc, .BB0

        ! CHECK: bvc %xcc, .BB0      ! encoding: [0x1e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc %xcc, .BB0

        ! CHECK: bvs %xcc, .BB0      ! encoding: [0x0e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs %xcc, .BB0


        ! CHECK: movne %icc, %g1, %g2            ! encoding: [0x85,0x66,0x40,0x01]
        ! CHECK: move %icc, %g1, %g2             ! encoding: [0x85,0x64,0x40,0x01]
        ! CHECK: movg %icc, %g1, %g2             ! encoding: [0x85,0x66,0x80,0x01]
        ! CHECK: movle %icc, %g1, %g2            ! encoding: [0x85,0x64,0x80,0x01]
        ! CHECK: movge %icc, %g1, %g2            ! encoding: [0x85,0x66,0xc0,0x01]
        ! CHECK: movl %icc, %g1, %g2             ! encoding: [0x85,0x64,0xc0,0x01]
        ! CHECK: movgu %icc, %g1, %g2            ! encoding: [0x85,0x67,0x00,0x01]
        ! CHECK: movleu %icc, %g1, %g2           ! encoding: [0x85,0x65,0x00,0x01]
        ! CHECK: movcc %icc, %g1, %g2            ! encoding: [0x85,0x67,0x40,0x01]
        ! CHECK: movcs %icc, %g1, %g2            ! encoding: [0x85,0x65,0x40,0x01]
        ! CHECK: movpos %icc, %g1, %g2           ! encoding: [0x85,0x67,0x80,0x01]
        ! CHECK: movneg %icc, %g1, %g2           ! encoding: [0x85,0x65,0x80,0x01]
        ! CHECK: movvc %icc, %g1, %g2            ! encoding: [0x85,0x67,0xc0,0x01]
        ! CHECK: movvs %icc, %g1, %g2            ! encoding: [0x85,0x65,0xc0,0x01]
        movne  %icc, %g1, %g2
        move   %icc, %g1, %g2
        movg   %icc, %g1, %g2
        movle  %icc, %g1, %g2
        movge  %icc, %g1, %g2
        movl   %icc, %g1, %g2
        movgu  %icc, %g1, %g2
        movleu %icc, %g1, %g2
        movcc  %icc, %g1, %g2
        movcs  %icc, %g1, %g2
        movpos %icc, %g1, %g2
        movneg %icc, %g1, %g2
        movvc  %icc, %g1, %g2
        movvs  %icc, %g1, %g2

        ! CHECK: movne %xcc, %g1, %g2            ! encoding: [0x85,0x66,0x50,0x01]
        ! CHECK: move %xcc, %g1, %g2             ! encoding: [0x85,0x64,0x50,0x01]
        ! CHECK: movg %xcc, %g1, %g2             ! encoding: [0x85,0x66,0x90,0x01]
        ! CHECK: movle %xcc, %g1, %g2            ! encoding: [0x85,0x64,0x90,0x01]
        ! CHECK: movge %xcc, %g1, %g2            ! encoding: [0x85,0x66,0xd0,0x01]
        ! CHECK: movl %xcc, %g1, %g2             ! encoding: [0x85,0x64,0xd0,0x01]
        ! CHECK: movgu %xcc, %g1, %g2            ! encoding: [0x85,0x67,0x10,0x01]
        ! CHECK: movleu %xcc, %g1, %g2           ! encoding: [0x85,0x65,0x10,0x01]
        ! CHECK: movcc %xcc, %g1, %g2            ! encoding: [0x85,0x67,0x50,0x01]
        ! CHECK: movcs %xcc, %g1, %g2            ! encoding: [0x85,0x65,0x50,0x01]
        ! CHECK: movpos %xcc, %g1, %g2           ! encoding: [0x85,0x67,0x90,0x01]
        ! CHECK: movneg %xcc, %g1, %g2           ! encoding: [0x85,0x65,0x90,0x01]
        ! CHECK: movvc %xcc, %g1, %g2            ! encoding: [0x85,0x67,0xd0,0x01]
        ! CHECK: movvs %xcc, %g1, %g2            ! encoding: [0x85,0x65,0xd0,0x01]
        movne  %xcc, %g1, %g2
        move   %xcc, %g1, %g2
        movg   %xcc, %g1, %g2
        movle  %xcc, %g1, %g2
        movge  %xcc, %g1, %g2
        movl   %xcc, %g1, %g2
        movgu  %xcc, %g1, %g2
        movleu %xcc, %g1, %g2
        movcc  %xcc, %g1, %g2
        movcs  %xcc, %g1, %g2
        movpos %xcc, %g1, %g2
        movneg %xcc, %g1, %g2
        movvc  %xcc, %g1, %g2
        movvs  %xcc, %g1, %g2

        ! CHECK: movu %fcc0, %g1, %g2            ! encoding: [0x85,0x61,0xc0,0x01]
        ! CHECK: movg %fcc0, %g1, %g2            ! encoding: [0x85,0x61,0x80,0x01]
        ! CHECK: movug %fcc0, %g1, %g2           ! encoding: [0x85,0x61,0x40,0x01]
        ! CHECK: movl %fcc0, %g1, %g2            ! encoding: [0x85,0x61,0x00,0x01]
        ! CHECK: movul %fcc0, %g1, %g2           ! encoding: [0x85,0x60,0xc0,0x01]
        ! CHECK: movlg %fcc0, %g1, %g2           ! encoding: [0x85,0x60,0x80,0x01]
        ! CHECK: movne %fcc0, %g1, %g2           ! encoding: [0x85,0x60,0x40,0x01]
        ! CHECK: move %fcc0, %g1, %g2            ! encoding: [0x85,0x62,0x40,0x01]
        ! CHECK: movue %fcc0, %g1, %g2           ! encoding: [0x85,0x62,0x80,0x01]
        ! CHECK: movge %fcc0, %g1, %g2           ! encoding: [0x85,0x62,0xc0,0x01]
        ! CHECK: movuge %fcc0, %g1, %g2          ! encoding: [0x85,0x63,0x00,0x01]
        ! CHECK: movle %fcc0, %g1, %g2           ! encoding: [0x85,0x63,0x40,0x01]
        ! CHECK: movule %fcc0, %g1, %g2          ! encoding: [0x85,0x63,0x80,0x01]
        ! CHECK: movo %fcc0, %g1, %g2            ! encoding: [0x85,0x63,0xc0,0x01]
        movu   %fcc0, %g1, %g2
        movg   %fcc0, %g1, %g2
        movug  %fcc0, %g1, %g2
        movl   %fcc0, %g1, %g2
        movul  %fcc0, %g1, %g2
        movlg  %fcc0, %g1, %g2
        movne  %fcc0, %g1, %g2
        move   %fcc0, %g1, %g2
        movue  %fcc0, %g1, %g2
        movge  %fcc0, %g1, %g2
        movuge %fcc0, %g1, %g2
        movle  %fcc0, %g1, %g2
        movule %fcc0, %g1, %g2
        movo   %fcc0, %g1, %g2


        ! CHECK fmovsne %icc, %f1, %f2          ! encoding: [0x85,0xaa,0x60,0x21]
        ! CHECK fmovse %icc, %f1, %f2           ! encoding: [0x85,0xa8,0x60,0x21]
        ! CHECK fmovsg %icc, %f1, %f2           ! encoding: [0x85,0xaa,0xa0,0x21]
        ! CHECK fmovsle %icc, %f1, %f2          ! encoding: [0x85,0xa8,0xa0,0x21]
        ! CHECK fmovsge %icc, %f1, %f2          ! encoding: [0x85,0xaa,0xe0,0x21]
        ! CHECK fmovsl %icc, %f1, %f2           ! encoding: [0x85,0xa8,0xe0,0x21]
        ! CHECK fmovsgu %icc, %f1, %f2          ! encoding: [0x85,0xab,0x20,0x21]
        ! CHECK fmovsleu %icc, %f1, %f2         ! encoding: [0x85,0xa9,0x20,0x21]
        ! CHECK fmovscc %icc, %f1, %f2          ! encoding: [0x85,0xab,0x60,0x21]
        ! CHECK fmovscs %icc, %f1, %f2          ! encoding: [0x85,0xa9,0x60,0x21]
        ! CHECK fmovspos %icc, %f1, %f2         ! encoding: [0x85,0xab,0xa0,0x21]
        ! CHECK fmovsneg %icc, %f1, %f2         ! encoding: [0x85,0xa9,0xa0,0x21]
        ! CHECK fmovsvc %icc, %f1, %f2          ! encoding: [0x85,0xab,0xe0,0x21]
        ! CHECK fmovsvs %icc, %f1, %f2          ! encoding: [0x85,0xa9,0xe0,0x21]
        fmovsne  %icc, %f1, %f2
        fmovse   %icc, %f1, %f2
        fmovsg   %icc, %f1, %f2
        fmovsle  %icc, %f1, %f2
        fmovsge  %icc, %f1, %f2
        fmovsl   %icc, %f1, %f2
        fmovsgu  %icc, %f1, %f2
        fmovsleu %icc, %f1, %f2
        fmovscc  %icc, %f1, %f2
        fmovscs  %icc, %f1, %f2
        fmovspos %icc, %f1, %f2
        fmovsneg %icc, %f1, %f2
        fmovsvc  %icc, %f1, %f2
        fmovsvs  %icc, %f1, %f2

        ! CHECK fmovsne %xcc, %f1, %f2          ! encoding: [0x85,0xaa,0x70,0x21]
        ! CHECK fmovse %xcc, %f1, %f2           ! encoding: [0x85,0xa8,0x70,0x21]
        ! CHECK fmovsg %xcc, %f1, %f2           ! encoding: [0x85,0xaa,0xb0,0x21]
        ! CHECK fmovsle %xcc, %f1, %f2          ! encoding: [0x85,0xa8,0xb0,0x21]
        ! CHECK fmovsge %xcc, %f1, %f2          ! encoding: [0x85,0xaa,0xf0,0x21]
        ! CHECK fmovsl %xcc, %f1, %f2           ! encoding: [0x85,0xa8,0xf0,0x21]
        ! CHECK fmovsgu %xcc, %f1, %f2          ! encoding: [0x85,0xab,0x30,0x21]
        ! CHECK fmovsleu %xcc, %f1, %f2         ! encoding: [0x85,0xa9,0x30,0x21]
        ! CHECK fmovscc %xcc, %f1, %f2          ! encoding: [0x85,0xab,0x70,0x21]
        ! CHECK fmovscs %xcc, %f1, %f2          ! encoding: [0x85,0xa9,0x70,0x21]
        ! CHECK fmovspos %xcc, %f1, %f2         ! encoding: [0x85,0xab,0xb0,0x21]
        ! CHECK fmovsneg %xcc, %f1, %f2         ! encoding: [0x85,0xa9,0xb0,0x21]
        ! CHECK fmovsvc %xcc, %f1, %f2          ! encoding: [0x85,0xab,0xf0,0x21]
        ! CHECK fmovsvs %xcc, %f1, %f2          ! encoding: [0x85,0xa9,0xf0,0x21]
        fmovsne  %xcc, %f1, %f2
        fmovse   %xcc, %f1, %f2
        fmovsg   %xcc, %f1, %f2
        fmovsle  %xcc, %f1, %f2
        fmovsge  %xcc, %f1, %f2
        fmovsl   %xcc, %f1, %f2
        fmovsgu  %xcc, %f1, %f2
        fmovsleu %xcc, %f1, %f2
        fmovscc  %xcc, %f1, %f2
        fmovscs  %xcc, %f1, %f2
        fmovspos %xcc, %f1, %f2
        fmovsneg %xcc, %f1, %f2
        fmovsvc  %xcc, %f1, %f2
        fmovsvs  %xcc, %f1, %f2

        ! CHECK fmovsu %fcc0, %f1, %f2          ! encoding: [0x85,0xa9,0xc0,0x21]
        ! CHECK fmovsg %fcc0, %f1, %f2          ! encoding: [0x85,0xa9,0x80,0x21]
        ! CHECK fmovsug %fcc0, %f1, %f2         ! encoding: [0x85,0xa9,0x40,0x21]
        ! CHECK fmovsl %fcc0, %f1, %f2          ! encoding: [0x85,0xa9,0x00,0x21]
        ! CHECK fmovsul %fcc0, %f1, %f2         ! encoding: [0x85,0xa8,0xc0,0x21]
        ! CHECK fmovslg %fcc0, %f1, %f2         ! encoding: [0x85,0xa8,0x80,0x21]
        ! CHECK fmovsne %fcc0, %f1, %f2         ! encoding: [0x85,0xa8,0x40,0x21]
        ! CHECK fmovse %fcc0, %f1, %f2          ! encoding: [0x85,0xaa,0x40,0x21]
        ! CHECK fmovsue %fcc0, %f1, %f2         ! encoding: [0x85,0xaa,0x80,0x21]
        ! CHECK fmovsge %fcc0, %f1, %f2         ! encoding: [0x85,0xaa,0xc0,0x21]
        ! CHECK fmovsuge %fcc0, %f1, %f2        ! encoding: [0x85,0xab,0x00,0x21]
        ! CHECK fmovsle %fcc0, %f1, %f2         ! encoding: [0x85,0xab,0x40,0x21]
        ! CHECK fmovsule %fcc0, %f1, %f2        ! encoding: [0x85,0xab,0x80,0x21]
        ! CHECK fmovso %fcc0, %f1, %f2          ! encoding: [0x85,0xab,0xc0,0x21]
        fmovsu   %fcc0, %f1, %f2
        fmovsg   %fcc0, %f1, %f2
        fmovsug  %fcc0, %f1, %f2
        fmovsl   %fcc0, %f1, %f2
        fmovsul  %fcc0, %f1, %f2
        fmovslg  %fcc0, %f1, %f2
        fmovsne  %fcc0, %f1, %f2
        fmovse   %fcc0, %f1, %f2
        fmovsue  %fcc0, %f1, %f2
        fmovsge  %fcc0, %f1, %f2
        fmovsuge %fcc0, %f1, %f2
        fmovsle  %fcc0, %f1, %f2
        fmovsule %fcc0, %f1, %f2
        fmovso   %fcc0, %f1, %f2

