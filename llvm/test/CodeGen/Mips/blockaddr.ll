; RUN: llc -march=mipsel < %s | FileCheck %s

@reg = common global i8* null, align 4

define i8* @dummy(i8* %x) nounwind readnone noinline {
entry:
  ret i8* %x
}

; CHECK: lw  $2, %got($tmp1)($gp)
; CHECK: addiu $4, $2, %lo($tmp1)
; CHECK: lw  $2, %got($tmp2)($gp)
; CHECK: addiu $2, $2, %lo($tmp2)
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
