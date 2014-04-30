; RUN: llc < %s -march=arm64 | FileCheck %s

;
; Get the actual value of the overflow bit.
;
define i1 @saddo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; CHECK-LABEL:  saddo.i32
; CHECK:        adds w8, w0, w1
; CHECK-NEXT:   csinc w0, wzr, wzr, vc
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define i1 @saddo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; CHECK-LABEL:  saddo.i64
; CHECK:        adds x8, x0, x1
; CHECK-NEXT:   csinc w0, wzr, wzr, vc
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define i1 @uaddo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; CHECK-LABEL:  uaddo.i32
; CHECK:        adds w8, w0, w1
; CHECK-NEXT:   csinc w0, wzr, wzr, lo
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define i1 @uaddo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; CHECK-LABEL:  uaddo.i64
; CHECK:        adds x8, x0, x1
; CHECK-NEXT:   csinc w0, wzr, wzr, lo
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define i1 @ssubo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; CHECK-LABEL:  ssubo.i32
; CHECK:        subs w8, w0, w1
; CHECK-NEXT:   csinc w0, wzr, wzr, vc
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define i1 @ssubo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; CHECK-LABEL:  ssubo.i64
; CHECK:        subs x8, x0, x1
; CHECK-NEXT:   csinc w0, wzr, wzr, vc
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define i1 @usubo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; CHECK-LABEL:  usubo.i32
; CHECK:        subs w8, w0, w1
; CHECK-NEXT:   csinc w0, wzr, wzr, hs
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define i1 @usubo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; CHECK-LABEL:  usubo.i64
; CHECK:        subs x8, x0, x1
; CHECK-NEXT:   csinc w0, wzr, wzr, hs
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define i1 @smulo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; CHECK-LABEL:  smulo.i32
; CHECK:        smull x8, w0, w1
; CHECK-NEXT:   lsr x9, x8, #32
; CHECK-NEXT:   cmp w9, w8, asr #31
; CHECK-NEXT:   csinc w0, wzr, wzr, eq
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define i1 @smulo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; CHECK-LABEL:  smulo.i64
; CHECK:        mul x8, x0, x1
; CHECK-NEXT:   smulh x9, x0, x1
; CHECK-NEXT:   cmp x9, x8, asr #63
; CHECK-NEXT:   csinc w0, wzr, wzr, eq
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}

define i1 @umulo.i32(i32 %v1, i32 %v2, i32* %res) {
entry:
; CHECK-LABEL:  umulo.i32
; CHECK:        umull x8, w0, w1
; CHECK-NEXT:   cmp xzr, x8, lsr #32
; CHECK-NEXT:   csinc w0, wzr, wzr, eq
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, i32* %res
  ret i1 %obit
}

define i1 @umulo.i64(i64 %v1, i64 %v2, i64* %res) {
entry:
; CHECK-LABEL:  umulo.i64
; CHECK:        umulh x8, x0, x1
; CHECK-NEXT:   cmp xzr, x8
; CHECK-NEXT:   csinc w8, wzr, wzr, eq
; CHECK-NEXT:   mul x9, x0, x1
  %t = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  store i64 %val, i64* %res
  ret i1 %obit
}


;
; Check the use of the overflow bit in combination with a select instruction.
;
define i32 @saddo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  saddo.select.i32
; CHECK:        cmn w0, w1
; CHECK-NEXT:   csel w0, w0, w1, vs
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @saddo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  saddo.select.i64
; CHECK:        cmn x0, x1
; CHECK-NEXT:   csel x0, x0, x1, vs
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @uaddo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  uaddo.select.i32
; CHECK:        cmn w0, w1
; CHECK-NEXT:   csel w0, w0, w1, hs
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @uaddo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  uaddo.select.i64
; CHECK:        cmn x0, x1
; CHECK-NEXT:   csel x0, x0, x1, hs
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @ssubo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  ssubo.select.i32
; CHECK:        cmp w0, w1
; CHECK-NEXT:   csel w0, w0, w1, vs
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @ssubo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  ssubo.select.i64
; CHECK:        cmp x0, x1
; CHECK-NEXT:   csel x0, x0, x1, vs
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @usubo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  usubo.select.i32
; CHECK:        cmp w0, w1
; CHECK-NEXT:   csel w0, w0, w1, lo
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @usubo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  usubo.select.i64
; CHECK:        cmp x0, x1
; CHECK-NEXT:   csel x0, x0, x1, lo
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @smulo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  smulo.select.i32
; CHECK:        smull    x8, w0, w1
; CHECK-NEXT:   lsr     x9, x8, #32
; CHECK-NEXT:   cmp     w9, w8, asr #31
; CHECK-NEXT:   csel    w0, w0, w1, ne
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @smulo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  smulo.select.i64
; CHECK:        mul      x8, x0, x1
; CHECK-NEXT:   smulh   x9, x0, x1
; CHECK-NEXT:   cmp     x9, x8, asr #63
; CHECK-NEXT:   csel    x0, x0, x1, ne
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}

define i32 @umulo.select.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  umulo.select.i32
; CHECK:        umull    x8, w0, w1
; CHECK-NEXT:   cmp     xzr, x8, lsr #32
; CHECK-NEXT:   csel    w0, w0, w1, ne
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %obit = extractvalue {i32, i1} %t, 1
  %ret = select i1 %obit, i32 %v1, i32 %v2
  ret i32 %ret
}

define i64 @umulo.select.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  umulo.select.i64
; CHECK:        umulh   x8, x0, x1
; CHECK-NEXT:   cmp     xzr, x8
; CHECK-NEXT:   csel    x0, x0, x1, ne
  %t = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %v1, i64 %v2)
  %obit = extractvalue {i64, i1} %t, 1
  %ret = select i1 %obit, i64 %v1, i64 %v2
  ret i64 %ret
}


;
; Check the use of the overflow bit in combination with a branch instruction.
;
define i1 @saddo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  saddo.br.i32
; CHECK:        cmn w0, w1
; CHECK-NEXT:   b.vc
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @saddo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  saddo.br.i64
; CHECK:        cmn x0, x1
; CHECK-NEXT:   b.vc
  %t = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @uaddo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  uaddo.br.i32
; CHECK:        cmn w0, w1
; CHECK-NEXT:   b.lo
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @uaddo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  uaddo.br.i64
; CHECK:        cmn x0, x1
; CHECK-NEXT:   b.lo
  %t = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @ssubo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  ssubo.br.i32
; CHECK:        cmp w0, w1
; CHECK-NEXT:   b.vc
  %t = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @ssubo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  ssubo.br.i64
; CHECK:        cmp x0, x1
; CHECK-NEXT:   b.vc
  %t = call {i64, i1} @llvm.ssub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @usubo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  usubo.br.i32
; CHECK:        cmp w0, w1
; CHECK-NEXT:   b.hs
  %t = call {i32, i1} @llvm.usub.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @usubo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  usubo.br.i64
; CHECK:        cmp x0, x1
; CHECK-NEXT:   b.hs
  %t = call {i64, i1} @llvm.usub.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @smulo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  smulo.br.i32
; CHECK:        smull    x8, w0, w1
; CHECK-NEXT:   lsr     x9, x8, #32
; CHECK-NEXT:   cmp     w9, w8, asr #31
; CHECK-NEXT:   b.eq
  %t = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @smulo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  smulo.br.i64
; CHECK:        mul      x8, x0, x1
; CHECK-NEXT:   smulh   x9, x0, x1
; CHECK-NEXT:   cmp     x9, x8, asr #63
; CHECK-NEXT:   b.eq
  %t = call {i64, i1} @llvm.smul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @umulo.br.i32(i32 %v1, i32 %v2) {
entry:
; CHECK-LABEL:  umulo.br.i32
; CHECK:        umull    x8, w0, w1
; CHECK-NEXT:   cmp     xzr, x8, lsr #32
; CHECK-NEXT:   b.eq
  %t = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %v1, i32 %v2)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

define i1 @umulo.br.i64(i64 %v1, i64 %v2) {
entry:
; CHECK-LABEL:  umulo.br.i64
; CHECK:        umulh   x8, x0, x1
; CHECK-NEXT:   cbz
  %t = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %v1, i64 %v2)
  %val = extractvalue {i64, i1} %t, 0
  %obit = extractvalue {i64, i1} %t, 1
  br i1 %obit, label %overflow, label %continue

overflow:
  ret i1 false

continue:
  ret i1 true
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.sadd.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.usub.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.smul.with.overflow.i64(i64, i64) nounwind readnone
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64) nounwind readnone

