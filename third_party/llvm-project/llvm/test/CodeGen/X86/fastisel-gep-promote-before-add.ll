; fastisel should not fold add with non-pointer bitwidth
; sext(a) + sext(b) != sext(a + b)
; RUN: llc -mtriple=x86_64-apple-darwin %s -O0 -o - | FileCheck %s

define zeroext i8 @gep_promotion(i8* %ptr) nounwind uwtable ssp {
entry:
  %ptr.addr = alloca i8*, align 8
  %add = add i8 64, 64 ; 0x40 + 0x40
  %0 = load i8*, i8** %ptr.addr, align 8

  ; CHECK-LABEL: _gep_promotion:
  ; CHECK: movzbl ({{.*}})
  %arrayidx = getelementptr inbounds i8, i8* %0, i8 %add

  %1 = load i8, i8* %arrayidx, align 1
  ret i8 %1
}

define zeroext i8 @gep_promotion_nonconst(i8 %i, i8* %ptr) nounwind uwtable ssp {
entry:
  %i.addr = alloca i8, align 4
  %ptr.addr = alloca i8*, align 8
  store i8 %i, i8* %i.addr, align 4
  store i8* %ptr, i8** %ptr.addr, align 8
  %0 = load i8, i8* %i.addr, align 4
  ; CHECK-LABEL: _gep_promotion_nonconst:
  ; CHECK: movzbl ({{.*}})
  %xor = xor i8 %0, -128   ; %0   ^ 0x80
  %add = add i8 %xor, -127 ; %xor + 0x81
  %1 = load i8*, i8** %ptr.addr, align 8

  %arrayidx = getelementptr inbounds i8, i8* %1, i8 %add

  %2 = load i8, i8* %arrayidx, align 1
  ret i8 %2
}

