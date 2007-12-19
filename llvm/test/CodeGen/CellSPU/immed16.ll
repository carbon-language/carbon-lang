; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep "ilh" %t1.s | count 5

define i16 @test_1() {
  %x = alloca i16, align 16
  store i16 419, i16* %x	;; ILH via pattern
  ret i16 0
}

define i16 @test_2() {
  %x = alloca i16, align 16
  store i16 1023, i16* %x	;; ILH via pattern
  ret i16 0
}

define i16 @test_3() {
  %x = alloca i16, align 16
  store i16 -1023, i16* %x	;; ILH via pattern
  ret i16 0
}

define i16 @test_4() {
  %x = alloca i16, align 16
  store i16 32767, i16* %x	;; ILH via pattern
  ret i16 0
}

define i16 @test_5() {
  %x = alloca i16, align 16
  store i16 -32768, i16* %x	;; ILH via pattern
  ret i16 0
}

define i16 @test_6() {
  ret i16 0
}


