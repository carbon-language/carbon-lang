; RUN: llc -march=x86 -promote-elements < %s
; RUN: llc -march=x86                   < %s
; RUN: llc -march=x86-64 -promote-elements < %s
; RUN: llc -march=x86-64                   < %s

define <1 x i8> @test_1xi8(<1 x i8> %x, <1 x i8>* %b) {
  %bb = load <1 x i8>* %b
  %tt = xor <1 x i8> %x, %bb
  store <1 x i8> %tt, <1 x i8>* %b
  br label %next

next:
  ret <1 x i8> %tt
}


define <1 x i16> @test_1xi16(<1 x i16> %x, <1 x i16>* %b) {
  %bb = load <1 x i16>* %b
  %tt = xor <1 x i16> %x, %bb
  store <1 x i16> %tt, <1 x i16>* %b
  br label %next

next:
  ret <1 x i16> %tt
}


define <1 x i32> @test_1xi32(<1 x i32> %x, <1 x i32>* %b) {
  %bb = load <1 x i32>* %b
  %tt = xor <1 x i32> %x, %bb
  store <1 x i32> %tt, <1 x i32>* %b
  br label %next

next:
  ret <1 x i32> %tt
}


define <1 x i64> @test_1xi64(<1 x i64> %x, <1 x i64>* %b) {
  %bb = load <1 x i64>* %b
  %tt = xor <1 x i64> %x, %bb
  store <1 x i64> %tt, <1 x i64>* %b
  br label %next

next:
  ret <1 x i64> %tt
}


define <1 x i128> @test_1xi128(<1 x i128> %x, <1 x i128>* %b) {
  %bb = load <1 x i128>* %b
  %tt = xor <1 x i128> %x, %bb
  store <1 x i128> %tt, <1 x i128>* %b
  br label %next

next:
  ret <1 x i128> %tt
}


define <1 x i256> @test_1xi256(<1 x i256> %x, <1 x i256>* %b) {
  %bb = load <1 x i256>* %b
  %tt = xor <1 x i256> %x, %bb
  store <1 x i256> %tt, <1 x i256>* %b
  br label %next

next:
  ret <1 x i256> %tt
}


define <1 x i512> @test_1xi512(<1 x i512> %x, <1 x i512>* %b) {
  %bb = load <1 x i512>* %b
  %tt = xor <1 x i512> %x, %bb
  store <1 x i512> %tt, <1 x i512>* %b
  br label %next

next:
  ret <1 x i512> %tt
}


define <2 x i8> @test_2xi8(<2 x i8> %x, <2 x i8>* %b) {
  %bb = load <2 x i8>* %b
  %tt = xor <2 x i8> %x, %bb
  store <2 x i8> %tt, <2 x i8>* %b
  br label %next

next:
  ret <2 x i8> %tt
}


define <2 x i16> @test_2xi16(<2 x i16> %x, <2 x i16>* %b) {
  %bb = load <2 x i16>* %b
  %tt = xor <2 x i16> %x, %bb
  store <2 x i16> %tt, <2 x i16>* %b
  br label %next

next:
  ret <2 x i16> %tt
}


define <2 x i32> @test_2xi32(<2 x i32> %x, <2 x i32>* %b) {
  %bb = load <2 x i32>* %b
  %tt = xor <2 x i32> %x, %bb
  store <2 x i32> %tt, <2 x i32>* %b
  br label %next

next:
  ret <2 x i32> %tt
}


define <2 x i64> @test_2xi64(<2 x i64> %x, <2 x i64>* %b) {
  %bb = load <2 x i64>* %b
  %tt = xor <2 x i64> %x, %bb
  store <2 x i64> %tt, <2 x i64>* %b
  br label %next

next:
  ret <2 x i64> %tt
}


define <2 x i128> @test_2xi128(<2 x i128> %x, <2 x i128>* %b) {
  %bb = load <2 x i128>* %b
  %tt = xor <2 x i128> %x, %bb
  store <2 x i128> %tt, <2 x i128>* %b
  br label %next

next:
  ret <2 x i128> %tt
}


define <2 x i256> @test_2xi256(<2 x i256> %x, <2 x i256>* %b) {
  %bb = load <2 x i256>* %b
  %tt = xor <2 x i256> %x, %bb
  store <2 x i256> %tt, <2 x i256>* %b
  br label %next

next:
  ret <2 x i256> %tt
}


define <2 x i512> @test_2xi512(<2 x i512> %x, <2 x i512>* %b) {
  %bb = load <2 x i512>* %b
  %tt = xor <2 x i512> %x, %bb
  store <2 x i512> %tt, <2 x i512>* %b
  br label %next

next:
  ret <2 x i512> %tt
}


define <3 x i8> @test_3xi8(<3 x i8> %x, <3 x i8>* %b) {
  %bb = load <3 x i8>* %b
  %tt = xor <3 x i8> %x, %bb
  store <3 x i8> %tt, <3 x i8>* %b
  br label %next

next:
  ret <3 x i8> %tt
}


define <3 x i16> @test_3xi16(<3 x i16> %x, <3 x i16>* %b) {
  %bb = load <3 x i16>* %b
  %tt = xor <3 x i16> %x, %bb
  store <3 x i16> %tt, <3 x i16>* %b
  br label %next

next:
  ret <3 x i16> %tt
}


define <3 x i32> @test_3xi32(<3 x i32> %x, <3 x i32>* %b) {
  %bb = load <3 x i32>* %b
  %tt = xor <3 x i32> %x, %bb
  store <3 x i32> %tt, <3 x i32>* %b
  br label %next

next:
  ret <3 x i32> %tt
}


define <3 x i64> @test_3xi64(<3 x i64> %x, <3 x i64>* %b) {
  %bb = load <3 x i64>* %b
  %tt = xor <3 x i64> %x, %bb
  store <3 x i64> %tt, <3 x i64>* %b
  br label %next

next:
  ret <3 x i64> %tt
}


define <3 x i128> @test_3xi128(<3 x i128> %x, <3 x i128>* %b) {
  %bb = load <3 x i128>* %b
  %tt = xor <3 x i128> %x, %bb
  store <3 x i128> %tt, <3 x i128>* %b
  br label %next

next:
  ret <3 x i128> %tt
}


define <3 x i256> @test_3xi256(<3 x i256> %x, <3 x i256>* %b) {
  %bb = load <3 x i256>* %b
  %tt = xor <3 x i256> %x, %bb
  store <3 x i256> %tt, <3 x i256>* %b
  br label %next

next:
  ret <3 x i256> %tt
}


define <3 x i512> @test_3xi512(<3 x i512> %x, <3 x i512>* %b) {
  %bb = load <3 x i512>* %b
  %tt = xor <3 x i512> %x, %bb
  store <3 x i512> %tt, <3 x i512>* %b
  br label %next

next:
  ret <3 x i512> %tt
}


define <4 x i8> @test_4xi8(<4 x i8> %x, <4 x i8>* %b) {
  %bb = load <4 x i8>* %b
  %tt = xor <4 x i8> %x, %bb
  store <4 x i8> %tt, <4 x i8>* %b
  br label %next

next:
  ret <4 x i8> %tt
}


define <4 x i16> @test_4xi16(<4 x i16> %x, <4 x i16>* %b) {
  %bb = load <4 x i16>* %b
  %tt = xor <4 x i16> %x, %bb
  store <4 x i16> %tt, <4 x i16>* %b
  br label %next

next:
  ret <4 x i16> %tt
}


define <4 x i32> @test_4xi32(<4 x i32> %x, <4 x i32>* %b) {
  %bb = load <4 x i32>* %b
  %tt = xor <4 x i32> %x, %bb
  store <4 x i32> %tt, <4 x i32>* %b
  br label %next

next:
  ret <4 x i32> %tt
}


define <4 x i64> @test_4xi64(<4 x i64> %x, <4 x i64>* %b) {
  %bb = load <4 x i64>* %b
  %tt = xor <4 x i64> %x, %bb
  store <4 x i64> %tt, <4 x i64>* %b
  br label %next

next:
  ret <4 x i64> %tt
}


define <4 x i128> @test_4xi128(<4 x i128> %x, <4 x i128>* %b) {
  %bb = load <4 x i128>* %b
  %tt = xor <4 x i128> %x, %bb
  store <4 x i128> %tt, <4 x i128>* %b
  br label %next

next:
  ret <4 x i128> %tt
}


define <4 x i256> @test_4xi256(<4 x i256> %x, <4 x i256>* %b) {
  %bb = load <4 x i256>* %b
  %tt = xor <4 x i256> %x, %bb
  store <4 x i256> %tt, <4 x i256>* %b
  br label %next

next:
  ret <4 x i256> %tt
}


define <4 x i512> @test_4xi512(<4 x i512> %x, <4 x i512>* %b) {
  %bb = load <4 x i512>* %b
  %tt = xor <4 x i512> %x, %bb
  store <4 x i512> %tt, <4 x i512>* %b
  br label %next

next:
  ret <4 x i512> %tt
}


define <5 x i8> @test_5xi8(<5 x i8> %x, <5 x i8>* %b) {
  %bb = load <5 x i8>* %b
  %tt = xor <5 x i8> %x, %bb
  store <5 x i8> %tt, <5 x i8>* %b
  br label %next

next:
  ret <5 x i8> %tt
}


define <5 x i16> @test_5xi16(<5 x i16> %x, <5 x i16>* %b) {
  %bb = load <5 x i16>* %b
  %tt = xor <5 x i16> %x, %bb
  store <5 x i16> %tt, <5 x i16>* %b
  br label %next

next:
  ret <5 x i16> %tt
}


define <5 x i32> @test_5xi32(<5 x i32> %x, <5 x i32>* %b) {
  %bb = load <5 x i32>* %b
  %tt = xor <5 x i32> %x, %bb
  store <5 x i32> %tt, <5 x i32>* %b
  br label %next

next:
  ret <5 x i32> %tt
}


define <5 x i64> @test_5xi64(<5 x i64> %x, <5 x i64>* %b) {
  %bb = load <5 x i64>* %b
  %tt = xor <5 x i64> %x, %bb
  store <5 x i64> %tt, <5 x i64>* %b
  br label %next

next:
  ret <5 x i64> %tt
}


define <5 x i128> @test_5xi128(<5 x i128> %x, <5 x i128>* %b) {
  %bb = load <5 x i128>* %b
  %tt = xor <5 x i128> %x, %bb
  store <5 x i128> %tt, <5 x i128>* %b
  br label %next

next:
  ret <5 x i128> %tt
}


define <5 x i256> @test_5xi256(<5 x i256> %x, <5 x i256>* %b) {
  %bb = load <5 x i256>* %b
  %tt = xor <5 x i256> %x, %bb
  store <5 x i256> %tt, <5 x i256>* %b
  br label %next

next:
  ret <5 x i256> %tt
}


define <5 x i512> @test_5xi512(<5 x i512> %x, <5 x i512>* %b) {
  %bb = load <5 x i512>* %b
  %tt = xor <5 x i512> %x, %bb
  store <5 x i512> %tt, <5 x i512>* %b
  br label %next

next:
  ret <5 x i512> %tt
}


define <6 x i8> @test_6xi8(<6 x i8> %x, <6 x i8>* %b) {
  %bb = load <6 x i8>* %b
  %tt = xor <6 x i8> %x, %bb
  store <6 x i8> %tt, <6 x i8>* %b
  br label %next

next:
  ret <6 x i8> %tt
}


define <6 x i16> @test_6xi16(<6 x i16> %x, <6 x i16>* %b) {
  %bb = load <6 x i16>* %b
  %tt = xor <6 x i16> %x, %bb
  store <6 x i16> %tt, <6 x i16>* %b
  br label %next

next:
  ret <6 x i16> %tt
}


define <6 x i32> @test_6xi32(<6 x i32> %x, <6 x i32>* %b) {
  %bb = load <6 x i32>* %b
  %tt = xor <6 x i32> %x, %bb
  store <6 x i32> %tt, <6 x i32>* %b
  br label %next

next:
  ret <6 x i32> %tt
}


define <6 x i64> @test_6xi64(<6 x i64> %x, <6 x i64>* %b) {
  %bb = load <6 x i64>* %b
  %tt = xor <6 x i64> %x, %bb
  store <6 x i64> %tt, <6 x i64>* %b
  br label %next

next:
  ret <6 x i64> %tt
}


define <6 x i128> @test_6xi128(<6 x i128> %x, <6 x i128>* %b) {
  %bb = load <6 x i128>* %b
  %tt = xor <6 x i128> %x, %bb
  store <6 x i128> %tt, <6 x i128>* %b
  br label %next

next:
  ret <6 x i128> %tt
}


define <6 x i256> @test_6xi256(<6 x i256> %x, <6 x i256>* %b) {
  %bb = load <6 x i256>* %b
  %tt = xor <6 x i256> %x, %bb
  store <6 x i256> %tt, <6 x i256>* %b
  br label %next

next:
  ret <6 x i256> %tt
}


define <6 x i512> @test_6xi512(<6 x i512> %x, <6 x i512>* %b) {
  %bb = load <6 x i512>* %b
  %tt = xor <6 x i512> %x, %bb
  store <6 x i512> %tt, <6 x i512>* %b
  br label %next

next:
  ret <6 x i512> %tt
}


define <7 x i8> @test_7xi8(<7 x i8> %x, <7 x i8>* %b) {
  %bb = load <7 x i8>* %b
  %tt = xor <7 x i8> %x, %bb
  store <7 x i8> %tt, <7 x i8>* %b
  br label %next

next:
  ret <7 x i8> %tt
}


define <7 x i16> @test_7xi16(<7 x i16> %x, <7 x i16>* %b) {
  %bb = load <7 x i16>* %b
  %tt = xor <7 x i16> %x, %bb
  store <7 x i16> %tt, <7 x i16>* %b
  br label %next

next:
  ret <7 x i16> %tt
}


define <7 x i32> @test_7xi32(<7 x i32> %x, <7 x i32>* %b) {
  %bb = load <7 x i32>* %b
  %tt = xor <7 x i32> %x, %bb
  store <7 x i32> %tt, <7 x i32>* %b
  br label %next

next:
  ret <7 x i32> %tt
}


define <7 x i64> @test_7xi64(<7 x i64> %x, <7 x i64>* %b) {
  %bb = load <7 x i64>* %b
  %tt = xor <7 x i64> %x, %bb
  store <7 x i64> %tt, <7 x i64>* %b
  br label %next

next:
  ret <7 x i64> %tt
}


define <7 x i128> @test_7xi128(<7 x i128> %x, <7 x i128>* %b) {
  %bb = load <7 x i128>* %b
  %tt = xor <7 x i128> %x, %bb
  store <7 x i128> %tt, <7 x i128>* %b
  br label %next

next:
  ret <7 x i128> %tt
}


define <7 x i256> @test_7xi256(<7 x i256> %x, <7 x i256>* %b) {
  %bb = load <7 x i256>* %b
  %tt = xor <7 x i256> %x, %bb
  store <7 x i256> %tt, <7 x i256>* %b
  br label %next

next:
  ret <7 x i256> %tt
}


define <7 x i512> @test_7xi512(<7 x i512> %x, <7 x i512>* %b) {
  %bb = load <7 x i512>* %b
  %tt = xor <7 x i512> %x, %bb
  store <7 x i512> %tt, <7 x i512>* %b
  br label %next

next:
  ret <7 x i512> %tt
}


define <8 x i8> @test_8xi8(<8 x i8> %x, <8 x i8>* %b) {
  %bb = load <8 x i8>* %b
  %tt = xor <8 x i8> %x, %bb
  store <8 x i8> %tt, <8 x i8>* %b
  br label %next

next:
  ret <8 x i8> %tt
}


define <8 x i16> @test_8xi16(<8 x i16> %x, <8 x i16>* %b) {
  %bb = load <8 x i16>* %b
  %tt = xor <8 x i16> %x, %bb
  store <8 x i16> %tt, <8 x i16>* %b
  br label %next

next:
  ret <8 x i16> %tt
}


define <8 x i32> @test_8xi32(<8 x i32> %x, <8 x i32>* %b) {
  %bb = load <8 x i32>* %b
  %tt = xor <8 x i32> %x, %bb
  store <8 x i32> %tt, <8 x i32>* %b
  br label %next

next:
  ret <8 x i32> %tt
}


define <8 x i64> @test_8xi64(<8 x i64> %x, <8 x i64>* %b) {
  %bb = load <8 x i64>* %b
  %tt = xor <8 x i64> %x, %bb
  store <8 x i64> %tt, <8 x i64>* %b
  br label %next

next:
  ret <8 x i64> %tt
}


define <8 x i128> @test_8xi128(<8 x i128> %x, <8 x i128>* %b) {
  %bb = load <8 x i128>* %b
  %tt = xor <8 x i128> %x, %bb
  store <8 x i128> %tt, <8 x i128>* %b
  br label %next

next:
  ret <8 x i128> %tt
}


define <8 x i256> @test_8xi256(<8 x i256> %x, <8 x i256>* %b) {
  %bb = load <8 x i256>* %b
  %tt = xor <8 x i256> %x, %bb
  store <8 x i256> %tt, <8 x i256>* %b
  br label %next

next:
  ret <8 x i256> %tt
}


define <8 x i512> @test_8xi512(<8 x i512> %x, <8 x i512>* %b) {
  %bb = load <8 x i512>* %b
  %tt = xor <8 x i512> %x, %bb
  store <8 x i512> %tt, <8 x i512>* %b
  br label %next

next:
  ret <8 x i512> %tt
}


define <9 x i8> @test_9xi8(<9 x i8> %x, <9 x i8>* %b) {
  %bb = load <9 x i8>* %b
  %tt = xor <9 x i8> %x, %bb
  store <9 x i8> %tt, <9 x i8>* %b
  br label %next

next:
  ret <9 x i8> %tt
}


define <9 x i16> @test_9xi16(<9 x i16> %x, <9 x i16>* %b) {
  %bb = load <9 x i16>* %b
  %tt = xor <9 x i16> %x, %bb
  store <9 x i16> %tt, <9 x i16>* %b
  br label %next

next:
  ret <9 x i16> %tt
}


define <9 x i32> @test_9xi32(<9 x i32> %x, <9 x i32>* %b) {
  %bb = load <9 x i32>* %b
  %tt = xor <9 x i32> %x, %bb
  store <9 x i32> %tt, <9 x i32>* %b
  br label %next

next:
  ret <9 x i32> %tt
}


define <9 x i64> @test_9xi64(<9 x i64> %x, <9 x i64>* %b) {
  %bb = load <9 x i64>* %b
  %tt = xor <9 x i64> %x, %bb
  store <9 x i64> %tt, <9 x i64>* %b
  br label %next

next:
  ret <9 x i64> %tt
}


define <9 x i128> @test_9xi128(<9 x i128> %x, <9 x i128>* %b) {
  %bb = load <9 x i128>* %b
  %tt = xor <9 x i128> %x, %bb
  store <9 x i128> %tt, <9 x i128>* %b
  br label %next

next:
  ret <9 x i128> %tt
}


define <9 x i256> @test_9xi256(<9 x i256> %x, <9 x i256>* %b) {
  %bb = load <9 x i256>* %b
  %tt = xor <9 x i256> %x, %bb
  store <9 x i256> %tt, <9 x i256>* %b
  br label %next

next:
  ret <9 x i256> %tt
}


define <9 x i512> @test_9xi512(<9 x i512> %x, <9 x i512>* %b) {
  %bb = load <9 x i512>* %b
  %tt = xor <9 x i512> %x, %bb
  store <9 x i512> %tt, <9 x i512>* %b
  br label %next

next:
  ret <9 x i512> %tt
}


define <10 x i8> @test_10xi8(<10 x i8> %x, <10 x i8>* %b) {
  %bb = load <10 x i8>* %b
  %tt = xor <10 x i8> %x, %bb
  store <10 x i8> %tt, <10 x i8>* %b
  br label %next

next:
  ret <10 x i8> %tt
}


define <10 x i16> @test_10xi16(<10 x i16> %x, <10 x i16>* %b) {
  %bb = load <10 x i16>* %b
  %tt = xor <10 x i16> %x, %bb
  store <10 x i16> %tt, <10 x i16>* %b
  br label %next

next:
  ret <10 x i16> %tt
}


define <10 x i32> @test_10xi32(<10 x i32> %x, <10 x i32>* %b) {
  %bb = load <10 x i32>* %b
  %tt = xor <10 x i32> %x, %bb
  store <10 x i32> %tt, <10 x i32>* %b
  br label %next

next:
  ret <10 x i32> %tt
}


define <10 x i64> @test_10xi64(<10 x i64> %x, <10 x i64>* %b) {
  %bb = load <10 x i64>* %b
  %tt = xor <10 x i64> %x, %bb
  store <10 x i64> %tt, <10 x i64>* %b
  br label %next

next:
  ret <10 x i64> %tt
}


define <10 x i128> @test_10xi128(<10 x i128> %x, <10 x i128>* %b) {
  %bb = load <10 x i128>* %b
  %tt = xor <10 x i128> %x, %bb
  store <10 x i128> %tt, <10 x i128>* %b
  br label %next

next:
  ret <10 x i128> %tt
}


define <10 x i256> @test_10xi256(<10 x i256> %x, <10 x i256>* %b) {
  %bb = load <10 x i256>* %b
  %tt = xor <10 x i256> %x, %bb
  store <10 x i256> %tt, <10 x i256>* %b
  br label %next

next:
  ret <10 x i256> %tt
}


define <10 x i512> @test_10xi512(<10 x i512> %x, <10 x i512>* %b) {
  %bb = load <10 x i512>* %b
  %tt = xor <10 x i512> %x, %bb
  store <10 x i512> %tt, <10 x i512>* %b
  br label %next

next:
  ret <10 x i512> %tt
}


define <11 x i8> @test_11xi8(<11 x i8> %x, <11 x i8>* %b) {
  %bb = load <11 x i8>* %b
  %tt = xor <11 x i8> %x, %bb
  store <11 x i8> %tt, <11 x i8>* %b
  br label %next

next:
  ret <11 x i8> %tt
}


define <11 x i16> @test_11xi16(<11 x i16> %x, <11 x i16>* %b) {
  %bb = load <11 x i16>* %b
  %tt = xor <11 x i16> %x, %bb
  store <11 x i16> %tt, <11 x i16>* %b
  br label %next

next:
  ret <11 x i16> %tt
}


define <11 x i32> @test_11xi32(<11 x i32> %x, <11 x i32>* %b) {
  %bb = load <11 x i32>* %b
  %tt = xor <11 x i32> %x, %bb
  store <11 x i32> %tt, <11 x i32>* %b
  br label %next

next:
  ret <11 x i32> %tt
}


define <11 x i64> @test_11xi64(<11 x i64> %x, <11 x i64>* %b) {
  %bb = load <11 x i64>* %b
  %tt = xor <11 x i64> %x, %bb
  store <11 x i64> %tt, <11 x i64>* %b
  br label %next

next:
  ret <11 x i64> %tt
}


define <11 x i128> @test_11xi128(<11 x i128> %x, <11 x i128>* %b) {
  %bb = load <11 x i128>* %b
  %tt = xor <11 x i128> %x, %bb
  store <11 x i128> %tt, <11 x i128>* %b
  br label %next

next:
  ret <11 x i128> %tt
}


define <11 x i256> @test_11xi256(<11 x i256> %x, <11 x i256>* %b) {
  %bb = load <11 x i256>* %b
  %tt = xor <11 x i256> %x, %bb
  store <11 x i256> %tt, <11 x i256>* %b
  br label %next

next:
  ret <11 x i256> %tt
}


define <11 x i512> @test_11xi512(<11 x i512> %x, <11 x i512>* %b) {
  %bb = load <11 x i512>* %b
  %tt = xor <11 x i512> %x, %bb
  store <11 x i512> %tt, <11 x i512>* %b
  br label %next

next:
  ret <11 x i512> %tt
}


define <12 x i8> @test_12xi8(<12 x i8> %x, <12 x i8>* %b) {
  %bb = load <12 x i8>* %b
  %tt = xor <12 x i8> %x, %bb
  store <12 x i8> %tt, <12 x i8>* %b
  br label %next

next:
  ret <12 x i8> %tt
}


define <12 x i16> @test_12xi16(<12 x i16> %x, <12 x i16>* %b) {
  %bb = load <12 x i16>* %b
  %tt = xor <12 x i16> %x, %bb
  store <12 x i16> %tt, <12 x i16>* %b
  br label %next

next:
  ret <12 x i16> %tt
}


define <12 x i32> @test_12xi32(<12 x i32> %x, <12 x i32>* %b) {
  %bb = load <12 x i32>* %b
  %tt = xor <12 x i32> %x, %bb
  store <12 x i32> %tt, <12 x i32>* %b
  br label %next

next:
  ret <12 x i32> %tt
}


define <12 x i64> @test_12xi64(<12 x i64> %x, <12 x i64>* %b) {
  %bb = load <12 x i64>* %b
  %tt = xor <12 x i64> %x, %bb
  store <12 x i64> %tt, <12 x i64>* %b
  br label %next

next:
  ret <12 x i64> %tt
}


define <12 x i128> @test_12xi128(<12 x i128> %x, <12 x i128>* %b) {
  %bb = load <12 x i128>* %b
  %tt = xor <12 x i128> %x, %bb
  store <12 x i128> %tt, <12 x i128>* %b
  br label %next

next:
  ret <12 x i128> %tt
}


define <12 x i256> @test_12xi256(<12 x i256> %x, <12 x i256>* %b) {
  %bb = load <12 x i256>* %b
  %tt = xor <12 x i256> %x, %bb
  store <12 x i256> %tt, <12 x i256>* %b
  br label %next

next:
  ret <12 x i256> %tt
}


define <12 x i512> @test_12xi512(<12 x i512> %x, <12 x i512>* %b) {
  %bb = load <12 x i512>* %b
  %tt = xor <12 x i512> %x, %bb
  store <12 x i512> %tt, <12 x i512>* %b
  br label %next

next:
  ret <12 x i512> %tt
}


define <13 x i8> @test_13xi8(<13 x i8> %x, <13 x i8>* %b) {
  %bb = load <13 x i8>* %b
  %tt = xor <13 x i8> %x, %bb
  store <13 x i8> %tt, <13 x i8>* %b
  br label %next

next:
  ret <13 x i8> %tt
}


define <13 x i16> @test_13xi16(<13 x i16> %x, <13 x i16>* %b) {
  %bb = load <13 x i16>* %b
  %tt = xor <13 x i16> %x, %bb
  store <13 x i16> %tt, <13 x i16>* %b
  br label %next

next:
  ret <13 x i16> %tt
}


define <13 x i32> @test_13xi32(<13 x i32> %x, <13 x i32>* %b) {
  %bb = load <13 x i32>* %b
  %tt = xor <13 x i32> %x, %bb
  store <13 x i32> %tt, <13 x i32>* %b
  br label %next

next:
  ret <13 x i32> %tt
}


define <13 x i64> @test_13xi64(<13 x i64> %x, <13 x i64>* %b) {
  %bb = load <13 x i64>* %b
  %tt = xor <13 x i64> %x, %bb
  store <13 x i64> %tt, <13 x i64>* %b
  br label %next

next:
  ret <13 x i64> %tt
}


define <13 x i128> @test_13xi128(<13 x i128> %x, <13 x i128>* %b) {
  %bb = load <13 x i128>* %b
  %tt = xor <13 x i128> %x, %bb
  store <13 x i128> %tt, <13 x i128>* %b
  br label %next

next:
  ret <13 x i128> %tt
}


define <13 x i256> @test_13xi256(<13 x i256> %x, <13 x i256>* %b) {
  %bb = load <13 x i256>* %b
  %tt = xor <13 x i256> %x, %bb
  store <13 x i256> %tt, <13 x i256>* %b
  br label %next

next:
  ret <13 x i256> %tt
}


define <13 x i512> @test_13xi512(<13 x i512> %x, <13 x i512>* %b) {
  %bb = load <13 x i512>* %b
  %tt = xor <13 x i512> %x, %bb
  store <13 x i512> %tt, <13 x i512>* %b
  br label %next

next:
  ret <13 x i512> %tt
}


define <14 x i8> @test_14xi8(<14 x i8> %x, <14 x i8>* %b) {
  %bb = load <14 x i8>* %b
  %tt = xor <14 x i8> %x, %bb
  store <14 x i8> %tt, <14 x i8>* %b
  br label %next

next:
  ret <14 x i8> %tt
}


define <14 x i16> @test_14xi16(<14 x i16> %x, <14 x i16>* %b) {
  %bb = load <14 x i16>* %b
  %tt = xor <14 x i16> %x, %bb
  store <14 x i16> %tt, <14 x i16>* %b
  br label %next

next:
  ret <14 x i16> %tt
}


define <14 x i32> @test_14xi32(<14 x i32> %x, <14 x i32>* %b) {
  %bb = load <14 x i32>* %b
  %tt = xor <14 x i32> %x, %bb
  store <14 x i32> %tt, <14 x i32>* %b
  br label %next

next:
  ret <14 x i32> %tt
}


define <14 x i64> @test_14xi64(<14 x i64> %x, <14 x i64>* %b) {
  %bb = load <14 x i64>* %b
  %tt = xor <14 x i64> %x, %bb
  store <14 x i64> %tt, <14 x i64>* %b
  br label %next

next:
  ret <14 x i64> %tt
}


define <14 x i128> @test_14xi128(<14 x i128> %x, <14 x i128>* %b) {
  %bb = load <14 x i128>* %b
  %tt = xor <14 x i128> %x, %bb
  store <14 x i128> %tt, <14 x i128>* %b
  br label %next

next:
  ret <14 x i128> %tt
}


define <14 x i256> @test_14xi256(<14 x i256> %x, <14 x i256>* %b) {
  %bb = load <14 x i256>* %b
  %tt = xor <14 x i256> %x, %bb
  store <14 x i256> %tt, <14 x i256>* %b
  br label %next

next:
  ret <14 x i256> %tt
}


define <14 x i512> @test_14xi512(<14 x i512> %x, <14 x i512>* %b) {
  %bb = load <14 x i512>* %b
  %tt = xor <14 x i512> %x, %bb
  store <14 x i512> %tt, <14 x i512>* %b
  br label %next

next:
  ret <14 x i512> %tt
}


define <15 x i8> @test_15xi8(<15 x i8> %x, <15 x i8>* %b) {
  %bb = load <15 x i8>* %b
  %tt = xor <15 x i8> %x, %bb
  store <15 x i8> %tt, <15 x i8>* %b
  br label %next

next:
  ret <15 x i8> %tt
}


define <15 x i16> @test_15xi16(<15 x i16> %x, <15 x i16>* %b) {
  %bb = load <15 x i16>* %b
  %tt = xor <15 x i16> %x, %bb
  store <15 x i16> %tt, <15 x i16>* %b
  br label %next

next:
  ret <15 x i16> %tt
}


define <15 x i32> @test_15xi32(<15 x i32> %x, <15 x i32>* %b) {
  %bb = load <15 x i32>* %b
  %tt = xor <15 x i32> %x, %bb
  store <15 x i32> %tt, <15 x i32>* %b
  br label %next

next:
  ret <15 x i32> %tt
}


define <15 x i64> @test_15xi64(<15 x i64> %x, <15 x i64>* %b) {
  %bb = load <15 x i64>* %b
  %tt = xor <15 x i64> %x, %bb
  store <15 x i64> %tt, <15 x i64>* %b
  br label %next

next:
  ret <15 x i64> %tt
}


define <15 x i128> @test_15xi128(<15 x i128> %x, <15 x i128>* %b) {
  %bb = load <15 x i128>* %b
  %tt = xor <15 x i128> %x, %bb
  store <15 x i128> %tt, <15 x i128>* %b
  br label %next

next:
  ret <15 x i128> %tt
}


define <15 x i256> @test_15xi256(<15 x i256> %x, <15 x i256>* %b) {
  %bb = load <15 x i256>* %b
  %tt = xor <15 x i256> %x, %bb
  store <15 x i256> %tt, <15 x i256>* %b
  br label %next

next:
  ret <15 x i256> %tt
}


define <15 x i512> @test_15xi512(<15 x i512> %x, <15 x i512>* %b) {
  %bb = load <15 x i512>* %b
  %tt = xor <15 x i512> %x, %bb
  store <15 x i512> %tt, <15 x i512>* %b
  br label %next

next:
  ret <15 x i512> %tt
}


define <16 x i8> @test_16xi8(<16 x i8> %x, <16 x i8>* %b) {
  %bb = load <16 x i8>* %b
  %tt = xor <16 x i8> %x, %bb
  store <16 x i8> %tt, <16 x i8>* %b
  br label %next

next:
  ret <16 x i8> %tt
}


define <16 x i16> @test_16xi16(<16 x i16> %x, <16 x i16>* %b) {
  %bb = load <16 x i16>* %b
  %tt = xor <16 x i16> %x, %bb
  store <16 x i16> %tt, <16 x i16>* %b
  br label %next

next:
  ret <16 x i16> %tt
}


define <16 x i32> @test_16xi32(<16 x i32> %x, <16 x i32>* %b) {
  %bb = load <16 x i32>* %b
  %tt = xor <16 x i32> %x, %bb
  store <16 x i32> %tt, <16 x i32>* %b
  br label %next

next:
  ret <16 x i32> %tt
}


define <16 x i64> @test_16xi64(<16 x i64> %x, <16 x i64>* %b) {
  %bb = load <16 x i64>* %b
  %tt = xor <16 x i64> %x, %bb
  store <16 x i64> %tt, <16 x i64>* %b
  br label %next

next:
  ret <16 x i64> %tt
}


define <16 x i128> @test_16xi128(<16 x i128> %x, <16 x i128>* %b) {
  %bb = load <16 x i128>* %b
  %tt = xor <16 x i128> %x, %bb
  store <16 x i128> %tt, <16 x i128>* %b
  br label %next

next:
  ret <16 x i128> %tt
}


define <16 x i256> @test_16xi256(<16 x i256> %x, <16 x i256>* %b) {
  %bb = load <16 x i256>* %b
  %tt = xor <16 x i256> %x, %bb
  store <16 x i256> %tt, <16 x i256>* %b
  br label %next

next:
  ret <16 x i256> %tt
}


define <16 x i512> @test_16xi512(<16 x i512> %x, <16 x i512>* %b) {
  %bb = load <16 x i512>* %b
  %tt = xor <16 x i512> %x, %bb
  store <16 x i512> %tt, <16 x i512>* %b
  br label %next

next:
  ret <16 x i512> %tt
}


define <17 x i8> @test_17xi8(<17 x i8> %x, <17 x i8>* %b) {
  %bb = load <17 x i8>* %b
  %tt = xor <17 x i8> %x, %bb
  store <17 x i8> %tt, <17 x i8>* %b
  br label %next

next:
  ret <17 x i8> %tt
}


define <17 x i16> @test_17xi16(<17 x i16> %x, <17 x i16>* %b) {
  %bb = load <17 x i16>* %b
  %tt = xor <17 x i16> %x, %bb
  store <17 x i16> %tt, <17 x i16>* %b
  br label %next

next:
  ret <17 x i16> %tt
}


define <17 x i32> @test_17xi32(<17 x i32> %x, <17 x i32>* %b) {
  %bb = load <17 x i32>* %b
  %tt = xor <17 x i32> %x, %bb
  store <17 x i32> %tt, <17 x i32>* %b
  br label %next

next:
  ret <17 x i32> %tt
}


define <17 x i64> @test_17xi64(<17 x i64> %x, <17 x i64>* %b) {
  %bb = load <17 x i64>* %b
  %tt = xor <17 x i64> %x, %bb
  store <17 x i64> %tt, <17 x i64>* %b
  br label %next

next:
  ret <17 x i64> %tt
}


define <17 x i128> @test_17xi128(<17 x i128> %x, <17 x i128>* %b) {
  %bb = load <17 x i128>* %b
  %tt = xor <17 x i128> %x, %bb
  store <17 x i128> %tt, <17 x i128>* %b
  br label %next

next:
  ret <17 x i128> %tt
}


define <17 x i256> @test_17xi256(<17 x i256> %x, <17 x i256>* %b) {
  %bb = load <17 x i256>* %b
  %tt = xor <17 x i256> %x, %bb
  store <17 x i256> %tt, <17 x i256>* %b
  br label %next

next:
  ret <17 x i256> %tt
}


define <17 x i512> @test_17xi512(<17 x i512> %x, <17 x i512>* %b) {
  %bb = load <17 x i512>* %b
  %tt = xor <17 x i512> %x, %bb
  store <17 x i512> %tt, <17 x i512>* %b
  br label %next

next:
  ret <17 x i512> %tt
}


define <18 x i8> @test_18xi8(<18 x i8> %x, <18 x i8>* %b) {
  %bb = load <18 x i8>* %b
  %tt = xor <18 x i8> %x, %bb
  store <18 x i8> %tt, <18 x i8>* %b
  br label %next

next:
  ret <18 x i8> %tt
}


define <18 x i16> @test_18xi16(<18 x i16> %x, <18 x i16>* %b) {
  %bb = load <18 x i16>* %b
  %tt = xor <18 x i16> %x, %bb
  store <18 x i16> %tt, <18 x i16>* %b
  br label %next

next:
  ret <18 x i16> %tt
}


define <18 x i32> @test_18xi32(<18 x i32> %x, <18 x i32>* %b) {
  %bb = load <18 x i32>* %b
  %tt = xor <18 x i32> %x, %bb
  store <18 x i32> %tt, <18 x i32>* %b
  br label %next

next:
  ret <18 x i32> %tt
}


define <18 x i64> @test_18xi64(<18 x i64> %x, <18 x i64>* %b) {
  %bb = load <18 x i64>* %b
  %tt = xor <18 x i64> %x, %bb
  store <18 x i64> %tt, <18 x i64>* %b
  br label %next

next:
  ret <18 x i64> %tt
}


define <18 x i128> @test_18xi128(<18 x i128> %x, <18 x i128>* %b) {
  %bb = load <18 x i128>* %b
  %tt = xor <18 x i128> %x, %bb
  store <18 x i128> %tt, <18 x i128>* %b
  br label %next

next:
  ret <18 x i128> %tt
}


define <18 x i256> @test_18xi256(<18 x i256> %x, <18 x i256>* %b) {
  %bb = load <18 x i256>* %b
  %tt = xor <18 x i256> %x, %bb
  store <18 x i256> %tt, <18 x i256>* %b
  br label %next

next:
  ret <18 x i256> %tt
}


define <18 x i512> @test_18xi512(<18 x i512> %x, <18 x i512>* %b) {
  %bb = load <18 x i512>* %b
  %tt = xor <18 x i512> %x, %bb
  store <18 x i512> %tt, <18 x i512>* %b
  br label %next

next:
  ret <18 x i512> %tt
}


define <19 x i8> @test_19xi8(<19 x i8> %x, <19 x i8>* %b) {
  %bb = load <19 x i8>* %b
  %tt = xor <19 x i8> %x, %bb
  store <19 x i8> %tt, <19 x i8>* %b
  br label %next

next:
  ret <19 x i8> %tt
}


define <19 x i16> @test_19xi16(<19 x i16> %x, <19 x i16>* %b) {
  %bb = load <19 x i16>* %b
  %tt = xor <19 x i16> %x, %bb
  store <19 x i16> %tt, <19 x i16>* %b
  br label %next

next:
  ret <19 x i16> %tt
}


define <19 x i32> @test_19xi32(<19 x i32> %x, <19 x i32>* %b) {
  %bb = load <19 x i32>* %b
  %tt = xor <19 x i32> %x, %bb
  store <19 x i32> %tt, <19 x i32>* %b
  br label %next

next:
  ret <19 x i32> %tt
}


define <19 x i64> @test_19xi64(<19 x i64> %x, <19 x i64>* %b) {
  %bb = load <19 x i64>* %b
  %tt = xor <19 x i64> %x, %bb
  store <19 x i64> %tt, <19 x i64>* %b
  br label %next

next:
  ret <19 x i64> %tt
}


define <19 x i128> @test_19xi128(<19 x i128> %x, <19 x i128>* %b) {
  %bb = load <19 x i128>* %b
  %tt = xor <19 x i128> %x, %bb
  store <19 x i128> %tt, <19 x i128>* %b
  br label %next

next:
  ret <19 x i128> %tt
}


define <19 x i256> @test_19xi256(<19 x i256> %x, <19 x i256>* %b) {
  %bb = load <19 x i256>* %b
  %tt = xor <19 x i256> %x, %bb
  store <19 x i256> %tt, <19 x i256>* %b
  br label %next

next:
  ret <19 x i256> %tt
}


define <19 x i512> @test_19xi512(<19 x i512> %x, <19 x i512>* %b) {
  %bb = load <19 x i512>* %b
  %tt = xor <19 x i512> %x, %bb
  store <19 x i512> %tt, <19 x i512>* %b
  br label %next

next:
  ret <19 x i512> %tt
}

