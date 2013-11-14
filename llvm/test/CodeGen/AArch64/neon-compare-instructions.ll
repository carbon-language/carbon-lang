; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define <8 x i8> @cmeq8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp eq <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmeq16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp eq <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmeq4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp eq <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmeq8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp eq <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmeq2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp eq <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmeq4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp eq <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmeq2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp eq <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmne8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmne16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmne4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmne8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmne2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmne4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmne2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmgt8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp sgt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgt16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp sgt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgt4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp sgt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgt8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp sgt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgt2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp sgt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgt4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp sgt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgt2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp sgt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlt8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp slt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlt16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp slt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlt4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp slt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlt8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp slt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlt2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp slt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlt4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp slt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlt2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp slt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmge8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp sge <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmge16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp sge <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmge4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp sge <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmge8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp sge <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmge2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp sge <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmge4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp sge <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmge2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp sge <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmle8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp sle <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmle16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp sle <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmle4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp sle <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmle8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp sle <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmle2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp sle <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmle4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp sle <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmle2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp sle <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhi8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ugt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhi16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ugt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhi4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp ugt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhi8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp ugt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhi2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp ugt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhi4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp ugt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhi2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp ugt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlo8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ult <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlo16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ult <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlo4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ult <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlo8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ult <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlo2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ult <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlo4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ult <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlo2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ult <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhs8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp uge <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhs16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp uge <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhs4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp uge <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhs8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp uge <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhs2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp uge <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhs4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp uge <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhs2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp uge <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmls8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ule <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmls16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ule <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmls4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ule <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmls8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ule <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmls2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ule <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmls4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ule <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmls2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ule <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmtst8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = and <8 x i8> %A, %B
	%tmp4 = icmp ne <8 x i8> %tmp3, zeroinitializer
   %tmp5 = sext <8 x i1> %tmp4 to <8 x i8>
	ret <8 x i8> %tmp5
}

define <16 x i8> @cmtst16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = and <16 x i8> %A, %B
	%tmp4 = icmp ne <16 x i8> %tmp3, zeroinitializer
   %tmp5 = sext <16 x i1> %tmp4 to <16 x i8>
	ret <16 x i8> %tmp5
}

define <4 x i16> @cmtst4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = and <4 x i16> %A, %B
	%tmp4 = icmp ne <4 x i16> %tmp3, zeroinitializer
   %tmp5 = sext <4 x i1> %tmp4 to <4 x i16>
	ret <4 x i16> %tmp5
}

define <8 x i16> @cmtst8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = and <8 x i16> %A, %B
	%tmp4 = icmp ne <8 x i16> %tmp3, zeroinitializer
   %tmp5 = sext <8 x i1> %tmp4 to <8 x i16>
	ret <8 x i16> %tmp5
}

define <2 x i32> @cmtst2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmtst {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = and <2 x i32> %A, %B
	%tmp4 = icmp ne <2 x i32> %tmp3, zeroinitializer
   %tmp5 = sext <2 x i1> %tmp4 to <2 x i32>
	ret <2 x i32> %tmp5
}

define <4 x i32> @cmtst4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmtst {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = and <4 x i32> %A, %B
	%tmp4 = icmp ne <4 x i32> %tmp3, zeroinitializer
   %tmp5 = sext <4 x i1> %tmp4 to <4 x i32>
	ret <4 x i32> %tmp5
}

define <2 x i64> @cmtst2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = and <2 x i64> %A, %B
	%tmp4 = icmp ne <2 x i64> %tmp3, zeroinitializer
   %tmp5 = sext <2 x i1> %tmp4 to <2 x i64>
	ret <2 x i64> %tmp5
}



define <8 x i8> @cmeqz8xi8(<8 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
	%tmp3 = icmp eq <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmeqz16xi8(<16 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
	%tmp3 = icmp eq <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmeqz4xi16(<4 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
	%tmp3 = icmp eq <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmeqz8xi16(<8 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
	%tmp3 = icmp eq <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmeqz2xi32(<2 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
	%tmp3 = icmp eq <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmeqz4xi32(<4 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
	%tmp3 = icmp eq <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmeqz2xi64(<2 x i64> %A) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
	%tmp3 = icmp eq <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmgez8xi8(<8 x i8> %A) {
;CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
	%tmp3 = icmp sge <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgez16xi8(<16 x i8> %A) {
;CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
	%tmp3 = icmp sge <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgez4xi16(<4 x i16> %A) {
;CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
	%tmp3 = icmp sge <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgez8xi16(<8 x i16> %A) {
;CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
	%tmp3 = icmp sge <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgez2xi32(<2 x i32> %A) {
;CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
	%tmp3 = icmp sge <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgez4xi32(<4 x i32> %A) {
;CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
	%tmp3 = icmp sge <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgez2xi64(<2 x i64> %A) {
;CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
	%tmp3 = icmp sge <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmgtz8xi8(<8 x i8> %A) {
;CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
	%tmp3 = icmp sgt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgtz16xi8(<16 x i8> %A) {
;CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
	%tmp3 = icmp sgt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgtz4xi16(<4 x i16> %A) {
;CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
	%tmp3 = icmp sgt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgtz8xi16(<8 x i16> %A) {
;CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
	%tmp3 = icmp sgt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgtz2xi32(<2 x i32> %A) {
;CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
	%tmp3 = icmp sgt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgtz4xi32(<4 x i32> %A) {
;CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
	%tmp3 = icmp sgt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgtz2xi64(<2 x i64> %A) {
;CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
	%tmp3 = icmp sgt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlez8xi8(<8 x i8> %A) {
;CHECK: cmle {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
	%tmp3 = icmp sle <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlez16xi8(<16 x i8> %A) {
;CHECK: cmle {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
	%tmp3 = icmp sle <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlez4xi16(<4 x i16> %A) {
;CHECK: cmle {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
	%tmp3 = icmp sle <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlez8xi16(<8 x i16> %A) {
;CHECK: cmle {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
	%tmp3 = icmp sle <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlez2xi32(<2 x i32> %A) {
;CHECK: cmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
	%tmp3 = icmp sle <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlez4xi32(<4 x i32> %A) {
;CHECK: cmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
	%tmp3 = icmp sle <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlez2xi64(<2 x i64> %A) {
;CHECK: cmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
	%tmp3 = icmp sle <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmltz8xi8(<8 x i8> %A) {
;CHECK: cmlt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
	%tmp3 = icmp slt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmltz16xi8(<16 x i8> %A) {
;CHECK: cmlt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
	%tmp3 = icmp slt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmltz4xi16(<4 x i16> %A) {
;CHECK: cmlt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
	%tmp3 = icmp slt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmltz8xi16(<8 x i16> %A) {
;CHECK: cmlt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
	%tmp3 = icmp slt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmltz2xi32(<2 x i32> %A) {
;CHECK: cmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
	%tmp3 = icmp slt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmltz4xi32(<4 x i32> %A) {
;CHECK: cmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
	%tmp3 = icmp slt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmltz2xi64(<2 x i64> %A) {
;CHECK: cmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
	%tmp3 = icmp slt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmneqz8xi8(<8 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmneqz16xi8(<16 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmneqz4xi16(<4 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmneqz8xi16(<8 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmneqz2xi32(<2 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmneqz4xi32(<4 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmneqz2xi64(<2 x i64> %A) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0x0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhsz8xi8(<8 x i8> %A) {
;CHECK: movi {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp uge <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhsz16xi8(<16 x i8> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp uge <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhsz4xi16(<4 x i16> %A) {
;CHECK: movi {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp uge <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhsz8xi16(<8 x i16> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp uge <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhsz2xi32(<2 x i32> %A) {
;CHECK: movi {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp uge <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhsz4xi32(<4 x i32> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp uge <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhsz2xi64(<2 x i64> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp uge <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmhiz8xi8(<8 x i8> %A) {
;CHECK: movi {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ugt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhiz16xi8(<16 x i8> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ugt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhiz4xi16(<4 x i16> %A) {
;CHECK: movi {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp ugt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhiz8xi16(<8 x i16> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp ugt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhiz2xi32(<2 x i32> %A) {
;CHECK: movi {{v[0-9]+}}.8b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp ugt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhiz4xi32(<4 x i32> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp ugt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhiz2xi64(<2 x i64> %A) {
;CHECK: movi {{v[0-9]+}}.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp ugt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlsz8xi8(<8 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.8b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ule <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlsz16xi8(<16 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ule <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlsz4xi16(<4 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.8b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ule <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlsz8xi16(<8 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ule <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlsz2xi32(<2 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.8b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ule <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlsz4xi32(<4 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ule <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlsz2xi64(<2 x i64> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ule <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmloz8xi8(<8 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.8b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.8b, v1.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ult <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmloz16xi8(<16 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ult <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmloz4xi16(<4 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.8b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ult <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmloz8xi16(<8 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ult <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmloz2xi32(<2 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.8b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ult <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmloz4xi32(<4 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ult <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmloz2xi64(<2 x i64> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v1.16b, #0x0
;CHECK-NEXT: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ult <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmoeq2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
   %tmp3 = fcmp oeq <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoeq4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
   %tmp3 = fcmp oeq <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmoeq2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
   %tmp3 = fcmp oeq <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmoge2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
   %tmp3 = fcmp oge <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoge4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
   %tmp3 = fcmp oge <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmoge2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
   %tmp3 = fcmp oge <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmogt2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
   %tmp3 = fcmp ogt <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmogt4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
   %tmp3 = fcmp ogt <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmogt2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
   %tmp3 = fcmp ogt <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmole2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.2s, v1.2s, v0.2s
   %tmp3 = fcmp ole <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmole4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.4s, v1.4s, v0.4s
   %tmp3 = fcmp ole <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmole2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.2d, v1.2d, v0.2d
   %tmp3 = fcmp ole <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmolt2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
;CHECK: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
   %tmp3 = fcmp olt <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmolt4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
;CHECK: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
   %tmp3 = fcmp olt <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmolt2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
;CHECK: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
   %tmp3 = fcmp olt <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmone2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ONE = OGT | OLT, OLT implemented as OGT so check reversed operands
;CHECK: fcmgt {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp one <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmone4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ONE = OGT | OLT, OLT implemented as OGT so check reversed operands
;CHECK: fcmgt {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp one <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmone2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ONE = OGT | OLT, OLT implemented as OGT so check reversed operands
;CHECK: fcmgt {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; todo check reversed operands
   %tmp3 = fcmp one <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmord2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ORD = OGE | OLT, OLT implemented as OGT, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ord <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}


define <4 x i32> @fcmord4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ORD = OGE | OLT, OLT implemented as OGT, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmord2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ORD = OGE | OLT, OLT implemented as OGT, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmuno2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UNO = !(OGE | OLT), OLT implemented as OGT, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uno <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmuno4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UNO = !(OGE | OLT), OLT implemented as OGT, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmuno2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UNO = !(OGE | OLT), OLT implemented as OGT, so check reversed operands.
;CHECK: fcmge {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmueq2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UEQ = !ONE = !(OGT | OLT), OLT implemented as OGT so check reversed operands
;CHECK: fcmgt {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ueq <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmueq4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UEQ = !ONE = !(OGT | OLT), OLT implemented as OGT so check reversed operands
;CHECK: fcmgt {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmueq2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UEQ = !ONE = !(OGT | OLT), OLT implemented as OGT so check reversed operands
;CHECK: fcmgt {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmuge2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UGE = ULE with swapped operands, ULE implemented as !OGT.
;CHECK: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uge <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmuge4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UGE = ULE with swapped operands, ULE implemented as !OGT.
;CHECK: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmuge2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UGE = ULE with swapped operands, ULE implemented as !OGT.
;CHECK: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmugt2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UGT = ULT with swapped operands, ULT implemented as !OGE.
;CHECK: fcmge {{v[0-9]+}}.2s, v1.2s, v0.2s
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ugt <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmugt4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UGT = ULT with swapped operands, ULT implemented as !OGE.
;CHECK: fcmge {{v[0-9]+}}.4s, v1.4s, v0.4s
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmugt2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: fcmge {{v[0-9]+}}.2d, v1.2d, v0.2d
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmule2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ULE implemented as !OGT.
;CHECK: fcmgt {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ule <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmule4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ULE implemented as !OGT.
;CHECK: fcmgt {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmule2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ULE implemented as !OGT.
;CHECK: fcmgt {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmult2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ULT implemented as !OGE.
;CHECK: fcmge {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ult <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmult4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ULT implemented as !OGE.
;CHECK: fcmge {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmult2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; ULT implemented as !OGE.
;CHECK: fcmge {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmune2xfloat(<2 x float> %A, <2 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UNE = !OEQ.
;CHECK: fcmeq {{v[0-9]+}}.2s, v0.2s, v1.2s
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp une <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmune4xfloat(<4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UNE = !OEQ.
;CHECK: fcmeq {{v[0-9]+}}.4s, v0.4s, v1.4s
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmune2xdouble(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; UNE = !OEQ.
;CHECK: fcmeq {{v[0-9]+}}.2d, v0.2d, v1.2d
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmoeqz2xfloat(<2 x float> %A) {
;CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
   %tmp3 = fcmp oeq <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoeqz4xfloat(<4 x float> %A) {
;CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
   %tmp3 = fcmp oeq <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmoeqz2xdouble(<2 x double> %A) {
;CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
   %tmp3 = fcmp oeq <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmogez2xfloat(<2 x float> %A) {
;CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
   %tmp3 = fcmp oge <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmogez4xfloat(<4 x float> %A) {
;CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
   %tmp3 = fcmp oge <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmogez2xdouble(<2 x double> %A) {
;CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
   %tmp3 = fcmp oge <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmogtz2xfloat(<2 x float> %A) {
;CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
   %tmp3 = fcmp ogt <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmogtz4xfloat(<4 x float> %A) {
;CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
   %tmp3 = fcmp ogt <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmogtz2xdouble(<2 x double> %A) {
;CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
   %tmp3 = fcmp ogt <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmoltz2xfloat(<2 x float> %A) {
;CHECK: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
   %tmp3 = fcmp olt <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoltz4xfloat(<4 x float> %A) {
;CHECK: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
   %tmp3 = fcmp olt <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmoltz2xdouble(<2 x double> %A) {
;CHECK: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
   %tmp3 = fcmp olt <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmolez2xfloat(<2 x float> %A) {
;CHECK: fcmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
   %tmp3 = fcmp ole <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmolez4xfloat(<4 x float> %A) {
;CHECK: fcmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
   %tmp3 = fcmp ole <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmolez2xdouble(<2 x double> %A) {
;CHECK: fcmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
   %tmp3 = fcmp ole <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmonez2xfloat(<2 x float> %A) {
; ONE with zero = OLT | OGT
;CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp one <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmonez4xfloat(<4 x float> %A) {
; ONE with zero = OLT | OGT
;CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp one <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmonez2xdouble(<2 x double> %A) {
; ONE with zero = OLT | OGT
;CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp one <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmordz2xfloat(<2 x float> %A) {
; ORD with zero = OLT | OGE
;CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ord <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmordz4xfloat(<4 x float> %A) {
; ORD with zero = OLT | OGE
;CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmordz2xdouble(<2 x double> %A) {
; ORD with zero = OLT | OGE
;CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmueqz2xfloat(<2 x float> %A) {
; UEQ with zero = !ONE = !(OLT |OGT)
;CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ueq <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmueqz4xfloat(<4 x float> %A) {
; UEQ with zero = !ONE = !(OLT |OGT)
;CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmueqz2xdouble(<2 x double> %A) {
; UEQ with zero = !ONE = !(OLT |OGT)
;CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmugez2xfloat(<2 x float> %A) {
; UGE with zero = !OLT
;CHECK: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uge <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmugez4xfloat(<4 x float> %A) {
; UGE with zero = !OLT
;CHECK: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmugez2xdouble(<2 x double> %A) {
; UGE with zero = !OLT
;CHECK: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmugtz2xfloat(<2 x float> %A) {
; UGT with zero = !OLE
;CHECK: fcmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ugt <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmugtz4xfloat(<4 x float> %A) {
; UGT with zero = !OLE
;CHECK: fcmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmugtz2xdouble(<2 x double> %A) {
; UGT with zero = !OLE
;CHECK: fcmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmultz2xfloat(<2 x float> %A) {
; ULT with zero = !OGE
;CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ult <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmultz4xfloat(<4 x float> %A) {
;CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmultz2xdouble(<2 x double> %A) {
;CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmulez2xfloat(<2 x float> %A) {
; ULE with zero = !OGT
;CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ule <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmulez4xfloat(<4 x float> %A) {
; ULE with zero = !OGT
;CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmulez2xdouble(<2 x double> %A) {
; ULE with zero = !OGT
;CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmunez2xfloat(<2 x float> %A) {
; UNE with zero = !OEQ with zero
;CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp une <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmunez4xfloat(<4 x float> %A) {
; UNE with zero = !OEQ with zero
;CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmunez2xdouble(<2 x double> %A) {
; UNE with zero = !OEQ with zero
;CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmunoz2xfloat(<2 x float> %A) {
; UNO with zero = !ORD = !(OLT | OGE)
;CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
;CHECK-NEXT: not {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uno <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmunoz4xfloat(<4 x float> %A) {
; UNO with zero = !ORD = !(OLT | OGE)
;CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmunoz2xdouble(<2 x double> %A) {
; UNO with zero = !ORD = !(OLT | OGE)
;CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0.0
;CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: not {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4

}
