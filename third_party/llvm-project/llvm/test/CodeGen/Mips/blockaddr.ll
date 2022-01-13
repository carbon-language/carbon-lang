; RUN: llc -march=mipsel -relocation-model=pic < %s \
; RUN:   | FileCheck %s -check-prefix=PIC-O32
; RUN: llc -march=mipsel -relocation-model=static < %s \
; RUN:   | FileCheck %s -check-prefix=STATIC-O32
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n32 \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N32
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n32 \
; RUN:      -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N32
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n64 \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-N64
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi n64 \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-N64
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32 -mattr=+mips16 \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-MIPS16

@reg = common global i8* null, align 4

define i8* @dummy(i8* %x) nounwind readnone noinline {
entry:
  ret i8* %x
}

; PIC-O32: lw  $[[R0:[0-9]+]], %got($tmp[[T0:[0-9]+]])
; PIC-O32: addiu ${{[0-9]+}}, $[[R0]], %lo($tmp[[T0]])
; PIC-O32: lw  $[[R1:[0-9]+]], %got($tmp[[T1:[0-9]+]])
; PIC-O32: addiu ${{[0-9]+}}, $[[R1]], %lo($tmp[[T1]])

; STATIC-O32: lui  $[[R2:[0-9]+]], %hi($tmp[[T2:[0-9]+]])
; STATIC-O32: addiu ${{[0-9]+}}, $[[R2]], %lo($tmp[[T2]])
; STATIC-O32: lui   $[[R3:[0-9]+]], %hi($tmp[[T3:[0-9]+]])
; STATIC-O32: addiu ${{[0-9]+}}, $[[R3]], %lo($tmp[[T3]])

; PIC-N32: lw  $[[R0:[0-9]+]], %got_page(.Ltmp[[T0:[0-9]+]])
; PIC-N32: addiu ${{[0-9]+}}, $[[R0]], %got_ofst(.Ltmp[[T0]])
; PIC-N32: lw  $[[R1:[0-9]+]], %got_page(.Ltmp[[T1:[0-9]+]])
; PIC-N32: addiu ${{[0-9]+}}, $[[R1]], %got_ofst(.Ltmp[[T1]])

; STATIC-N32: lui  $[[R2:[0-9]+]], %hi(.Ltmp[[T2:[0-9]+]])
; STATIC-N32: addiu ${{[0-9]+}}, $[[R2]], %lo(.Ltmp[[T2]])
; STATIC-N32: lui   $[[R3:[0-9]+]], %hi(.Ltmp[[T3:[0-9]+]])
; STATIC-N32: addiu ${{[0-9]+}}, $[[R3]], %lo(.Ltmp[[T3]])

; PIC-N64: ld  $[[R0:[0-9]+]], %got_page(.Ltmp[[T0:[0-9]+]])
; PIC-N64: daddiu ${{[0-9]+}}, $[[R0]], %got_ofst(.Ltmp[[T0]])
; PIC-N64: ld  $[[R1:[0-9]+]], %got_page(.Ltmp[[T1:[0-9]+]])
; PIC-N64: daddiu ${{[0-9]+}}, $[[R1]], %got_ofst(.Ltmp[[T1]])

; STATIC-N64: lui $[[R0:[0-9]]], %highest(.Ltmp[[L0:[0-9]]])
; STATIC-N64: daddiu $[[R1:[0-9]]], $[[R0]], %higher(.Ltmp[[L0]])
; STATIC-N64: dsll $[[R2:[0-9]]], $[[R1]], 16
; STATIC-N64: daddiu $[[R3:[0-9]]], $[[R2]], %hi(.Ltmp[[L0]])
; STATIC-N64: dsll $[[R4:[0-9]]], $[[R3]], 16
; STATIC-N64: daddiu $[[R5:[0-9]]], $[[R4]], %lo(.Ltmp[[L0]])

; STATIC-MIPS16: .ent	f
; STATIC-MIPS16: li   $[[R0:[0-9]+]], %hi($tmp[[L0:[0-9]+]])
; STATIC-MIPS16: sll  $[[R1:[0-9]+]], $[[R0]], 16
; STATIC-MIPS16: li   $[[R2:[0-9]+]], %lo($tmp[[L0]])
; STATIC-MIPS16: addu $[[R3:[0-9]+]], $[[R1]], $[[R2]]
; STATIC-MIPS16: jal	dummy

define void @f() nounwind {
entry:
  %call = tail call i8* @dummy(i8* blockaddress(@f, %baz))
  indirectbr i8* %call, [label %baz, label %foo]

foo:                                              ; preds = %foo, %entry
  store i8* blockaddress(@f, %foo), i8** @reg, align 4
  br label %foo

baz:                                              ; preds = %entry
  store i8* null, i8** @reg, align 4
  ret void
}
