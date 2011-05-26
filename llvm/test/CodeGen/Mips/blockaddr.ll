; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s -check-prefix=CHECK-PIC
; RUN: llc -march=mipsel -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC

@reg = common global i8* null, align 4

define i8* @dummy(i8* %x) nounwind readnone noinline {
entry:
  ret i8* %x
}

; CHECK-PIC: lw  $[[R0:[0-9]+]], %got($tmp[[T0:[0-9]+]])($gp)
; CHECK-PIC: addiu ${{[0-9]+}}, $[[R0]], %lo($tmp[[T0]])
; CHECK-PIC: lw  $[[R1:[0-9]+]], %got($tmp[[T1:[0-9]+]])($gp)
; CHECK-PIC: addiu ${{[0-9]+}}, $[[R1]], %lo($tmp[[T1]])
; CHECK-STATIC: lui  $[[R2:[0-9]+]], %hi($tmp[[T0:[0-9]+]])
; CHECK-STATIC: addiu ${{[0-9]+}}, $[[R2]], %lo($tmp[[T0]])
; CHECK-STATIC: lui   $[[R3:[0-9]+]], %hi($tmp[[T1:[0-9]+]])
; CHECK-STATIC: addiu ${{[0-9]+}}, $[[R3]], %lo($tmp[[T1]])
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
