; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s

define <8 x i8> @addhn8b(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: addhn8b:
;CHECK: addhn.8b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.addhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @addhn4h(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: addhn4h:
;CHECK: addhn.4h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @addhn2s(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: addhn2s:
;CHECK: addhn.2s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.addhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @addhn2_16b(<8 x i16> %a, <8 x i16> %b) nounwind {
;CHECK-LABEL: addhn2_16b:
;CHECK: addhn.8b
;CHECK-NEXT: addhn2.16b
  %vaddhn2.i = tail call <8 x i8> @llvm.aarch64.neon.addhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %vaddhn_high2.i = tail call <8 x i8> @llvm.aarch64.neon.addhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %res = shufflevector <8 x i8> %vaddhn2.i, <8 x i8> %vaddhn_high2.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %res
}

define <8 x i16> @addhn2_8h(<4 x i32> %a, <4 x i32> %b) nounwind {
;CHECK-LABEL: addhn2_8h:
;CHECK: addhn.4h
;CHECK-NEXT: addhn2.8h
  %vaddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %vaddhn_high3.i = tail call <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %res = shufflevector <4 x i16> %vaddhn2.i, <4 x i16> %vaddhn_high3.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %res
}

define <4 x i32> @addhn2_4s(<2 x i64> %a, <2 x i64> %b) nounwind {
;CHECK-LABEL: addhn2_4s:
;CHECK: addhn.2s
;CHECK-NEXT: addhn2.4s
  %vaddhn2.i = tail call <2 x i32> @llvm.aarch64.neon.addhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %vaddhn_high3.i = tail call <2 x i32> @llvm.aarch64.neon.addhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %res = shufflevector <2 x i32> %vaddhn2.i, <2 x i32> %vaddhn_high3.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

declare <2 x i32> @llvm.aarch64.neon.addhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.addhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.addhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone


define <8 x i8> @raddhn8b(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: raddhn8b:
;CHECK: raddhn.8b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i8> %tmp3
}

define <4 x i16> @raddhn4h(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: raddhn4h:
;CHECK: raddhn.4h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i16> %tmp3
}

define <2 x i32> @raddhn2s(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: raddhn2s:
;CHECK: raddhn.2s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i32> %tmp3
}

define <16 x i8> @raddhn2_16b(<8 x i16> %a, <8 x i16> %b) nounwind {
;CHECK-LABEL: raddhn2_16b:
;CHECK: raddhn.8b
;CHECK-NEXT: raddhn2.16b
  %vraddhn2.i = tail call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %vraddhn_high2.i = tail call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b) nounwind
  %res = shufflevector <8 x i8> %vraddhn2.i, <8 x i8> %vraddhn_high2.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %res
}

define <8 x i16> @raddhn2_8h(<4 x i32> %a, <4 x i32> %b) nounwind {
;CHECK-LABEL: raddhn2_8h:
;CHECK: raddhn.4h
;CHECK-NEXT: raddhn2.8h
  %vraddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %vraddhn_high3.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b) nounwind
  %res = shufflevector <4 x i16> %vraddhn2.i, <4 x i16> %vraddhn_high3.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %res
}

define <4 x i32> @raddhn2_4s(<2 x i64> %a, <2 x i64> %b) nounwind {
;CHECK-LABEL: raddhn2_4s:
;CHECK: raddhn.2s
;CHECK-NEXT: raddhn2.4s
  %vraddhn2.i = tail call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %vraddhn_high3.i = tail call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b) nounwind
  %res = shufflevector <2 x i32> %vraddhn2.i, <2 x i32> %vraddhn_high3.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

declare <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone

define <8 x i16> @saddl8h(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: saddl8h:
;CHECK: saddl.8h
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
  %tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
  %tmp5 = add <8 x i16> %tmp3, %tmp4
        ret <8 x i16> %tmp5
}

define <4 x i32> @saddl4s(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: saddl4s:
;CHECK: saddl.4s
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
  %tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
  %tmp5 = add <4 x i32> %tmp3, %tmp4
        ret <4 x i32> %tmp5
}

define <2 x i64> @saddl2d(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: saddl2d:
;CHECK: saddl.2d
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
  %tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
  %tmp5 = add <2 x i64> %tmp3, %tmp4
        ret <2 x i64> %tmp5
}

define <8 x i16> @saddl2_8h(<16 x i8> %a, <16 x i8> %b) nounwind  {
; CHECK-LABEL: saddl2_8h:
; CHECK-NEXT: saddl2.8h v0, v0, v1
; CHECK-NEXT: ret
  %tmp = bitcast <16 x i8> %a to <2 x i64>
  %shuffle.i.i.i = shufflevector <2 x i64> %tmp, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp1 = bitcast <1 x i64> %shuffle.i.i.i to <8 x i8>
  %vmovl.i.i.i = sext <8 x i8> %tmp1 to <8 x i16>
  %tmp2 = bitcast <16 x i8> %b to <2 x i64>
  %shuffle.i.i4.i = shufflevector <2 x i64> %tmp2, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp3 = bitcast <1 x i64> %shuffle.i.i4.i to <8 x i8>
  %vmovl.i.i5.i = sext <8 x i8> %tmp3 to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i.i, %vmovl.i.i5.i
  ret <8 x i16> %add.i
}

define <4 x i32> @saddl2_4s(<8 x i16> %a, <8 x i16> %b) nounwind  {
; CHECK-LABEL: saddl2_4s:
; CHECK-NEXT: saddl2.4s v0, v0, v1
; CHECK-NEXT: ret
  %tmp = bitcast <8 x i16> %a to <2 x i64>
  %shuffle.i.i.i = shufflevector <2 x i64> %tmp, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp1 = bitcast <1 x i64> %shuffle.i.i.i to <4 x i16>
  %vmovl.i.i.i = sext <4 x i16> %tmp1 to <4 x i32>
  %tmp2 = bitcast <8 x i16> %b to <2 x i64>
  %shuffle.i.i4.i = shufflevector <2 x i64> %tmp2, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp3 = bitcast <1 x i64> %shuffle.i.i4.i to <4 x i16>
  %vmovl.i.i5.i = sext <4 x i16> %tmp3 to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i.i, %vmovl.i.i5.i
  ret <4 x i32> %add.i
}

define <2 x i64> @saddl2_2d(<4 x i32> %a, <4 x i32> %b) nounwind  {
; CHECK-LABEL: saddl2_2d:
; CHECK-NEXT: saddl2.2d v0, v0, v1
; CHECK-NEXT: ret
  %tmp = bitcast <4 x i32> %a to <2 x i64>
  %shuffle.i.i.i = shufflevector <2 x i64> %tmp, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp1 = bitcast <1 x i64> %shuffle.i.i.i to <2 x i32>
  %vmovl.i.i.i = sext <2 x i32> %tmp1 to <2 x i64>
  %tmp2 = bitcast <4 x i32> %b to <2 x i64>
  %shuffle.i.i4.i = shufflevector <2 x i64> %tmp2, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp3 = bitcast <1 x i64> %shuffle.i.i4.i to <2 x i32>
  %vmovl.i.i5.i = sext <2 x i32> %tmp3 to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i.i, %vmovl.i.i5.i
  ret <2 x i64> %add.i
}

define <8 x i16> @uaddl8h(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uaddl8h:
;CHECK: uaddl.8h
  %tmp1 = load <8 x i8>, <8 x i8>* %A
  %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
  %tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
  %tmp5 = add <8 x i16> %tmp3, %tmp4
  ret <8 x i16> %tmp5
}

define <4 x i32> @uaddl4s(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uaddl4s:
;CHECK: uaddl.4s
  %tmp1 = load <4 x i16>, <4 x i16>* %A
  %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
  %tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
  %tmp5 = add <4 x i32> %tmp3, %tmp4
  ret <4 x i32> %tmp5
}

define <2 x i64> @uaddl2d(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uaddl2d:
;CHECK: uaddl.2d
  %tmp1 = load <2 x i32>, <2 x i32>* %A
  %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
  %tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
  %tmp5 = add <2 x i64> %tmp3, %tmp4
  ret <2 x i64> %tmp5
}


define <8 x i16> @uaddl2_8h(<16 x i8> %a, <16 x i8> %b) nounwind  {
; CHECK-LABEL: uaddl2_8h:
; CHECK-NEXT: uaddl2.8h v0, v0, v1
; CHECK-NEXT: ret
  %tmp = bitcast <16 x i8> %a to <2 x i64>
  %shuffle.i.i.i = shufflevector <2 x i64> %tmp, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp1 = bitcast <1 x i64> %shuffle.i.i.i to <8 x i8>
  %vmovl.i.i.i = zext <8 x i8> %tmp1 to <8 x i16>
  %tmp2 = bitcast <16 x i8> %b to <2 x i64>
  %shuffle.i.i4.i = shufflevector <2 x i64> %tmp2, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp3 = bitcast <1 x i64> %shuffle.i.i4.i to <8 x i8>
  %vmovl.i.i5.i = zext <8 x i8> %tmp3 to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i.i, %vmovl.i.i5.i
  ret <8 x i16> %add.i
}

define <4 x i32> @uaddl2_4s(<8 x i16> %a, <8 x i16> %b) nounwind  {
; CHECK-LABEL: uaddl2_4s:
; CHECK-NEXT: uaddl2.4s v0, v0, v1
; CHECK-NEXT: ret
  %tmp = bitcast <8 x i16> %a to <2 x i64>
  %shuffle.i.i.i = shufflevector <2 x i64> %tmp, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp1 = bitcast <1 x i64> %shuffle.i.i.i to <4 x i16>
  %vmovl.i.i.i = zext <4 x i16> %tmp1 to <4 x i32>
  %tmp2 = bitcast <8 x i16> %b to <2 x i64>
  %shuffle.i.i4.i = shufflevector <2 x i64> %tmp2, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp3 = bitcast <1 x i64> %shuffle.i.i4.i to <4 x i16>
  %vmovl.i.i5.i = zext <4 x i16> %tmp3 to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i.i, %vmovl.i.i5.i
  ret <4 x i32> %add.i
}

define <2 x i64> @uaddl2_2d(<4 x i32> %a, <4 x i32> %b) nounwind  {
; CHECK-LABEL: uaddl2_2d:
; CHECK-NEXT: uaddl2.2d v0, v0, v1
; CHECK-NEXT: ret
  %tmp = bitcast <4 x i32> %a to <2 x i64>
  %shuffle.i.i.i = shufflevector <2 x i64> %tmp, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp1 = bitcast <1 x i64> %shuffle.i.i.i to <2 x i32>
  %vmovl.i.i.i = zext <2 x i32> %tmp1 to <2 x i64>
  %tmp2 = bitcast <4 x i32> %b to <2 x i64>
  %shuffle.i.i4.i = shufflevector <2 x i64> %tmp2, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp3 = bitcast <1 x i64> %shuffle.i.i4.i to <2 x i32>
  %vmovl.i.i5.i = zext <2 x i32> %tmp3 to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i.i, %vmovl.i.i5.i
  ret <2 x i64> %add.i
}

define <8 x i16> @uaddw8h(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: uaddw8h:
;CHECK: uaddw.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
  %tmp4 = add <8 x i16> %tmp1, %tmp3
        ret <8 x i16> %tmp4
}

define <4 x i32> @uaddw4s(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uaddw4s:
;CHECK: uaddw.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
  %tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
  %tmp4 = add <4 x i32> %tmp1, %tmp3
        ret <4 x i32> %tmp4
}

define <2 x i64> @uaddw2d(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uaddw2d:
;CHECK: uaddw.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
  %tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
  %tmp4 = add <2 x i64> %tmp1, %tmp3
        ret <2 x i64> %tmp4
}

define <8 x i16> @uaddw2_8h(<8 x i16>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: uaddw2_8h:
;CHECK: uaddw.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A

        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %high2 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %ext2 = zext <8 x i8> %high2 to <8 x i16>

        %res = add <8 x i16> %tmp1, %ext2
        ret <8 x i16> %res
}

define <4 x i32> @uaddw2_4s(<4 x i32>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uaddw2_4s:
;CHECK: uaddw.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A

        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %high2 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %ext2 = zext <4 x i16> %high2 to <4 x i32>

        %res = add <4 x i32> %tmp1, %ext2
        ret <4 x i32> %res
}

define <2 x i64> @uaddw2_2d(<2 x i64>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uaddw2_2d:
;CHECK: uaddw.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A

        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %high2 = shufflevector <4 x i32> %tmp2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %ext2 = zext <2 x i32> %high2 to <2 x i64>

        %res = add <2 x i64> %tmp1, %ext2
        ret <2 x i64> %res
}

define <8 x i16> @saddw8h(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: saddw8h:
;CHECK: saddw.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
        %tmp4 = add <8 x i16> %tmp1, %tmp3
        ret <8 x i16> %tmp4
}

define <4 x i32> @saddw4s(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: saddw4s:
;CHECK: saddw.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
        %tmp4 = add <4 x i32> %tmp1, %tmp3
        ret <4 x i32> %tmp4
}

define <2 x i64> @saddw2d(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: saddw2d:
;CHECK: saddw.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
        %tmp4 = add <2 x i64> %tmp1, %tmp3
        ret <2 x i64> %tmp4
}

define <8 x i16> @saddw2_8h(<8 x i16>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: saddw2_8h:
;CHECK: saddw.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A

        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %high2 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        %ext2 = sext <8 x i8> %high2 to <8 x i16>

        %res = add <8 x i16> %tmp1, %ext2
        ret <8 x i16> %res
}

define <4 x i32> @saddw2_4s(<4 x i32>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: saddw2_4s:
;CHECK: saddw.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A

        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %high2 = shufflevector <8 x i16> %tmp2, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
        %ext2 = sext <4 x i16> %high2 to <4 x i32>

        %res = add <4 x i32> %tmp1, %ext2
        ret <4 x i32> %res
}

define <2 x i64> @saddw2_2d(<2 x i64>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: saddw2_2d:
;CHECK: saddw.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A

        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %high2 = shufflevector <4 x i32> %tmp2, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
        %ext2 = sext <2 x i32> %high2 to <2 x i64>

        %res = add <2 x i64> %tmp1, %ext2
        ret <2 x i64> %res
}

define <4 x i16> @saddlp4h(<8 x i8>* %A) nounwind {
;CHECK-LABEL: saddlp4h:
;CHECK: saddlp.4h
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8> %tmp1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @saddlp2s(<4 x i16>* %A) nounwind {
;CHECK-LABEL: saddlp2s:
;CHECK: saddlp.2s
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16> %tmp1)
        ret <2 x i32> %tmp3
}

define <1 x i64> @saddlp1d(<2 x i32>* %A) nounwind {
;CHECK-LABEL: saddlp1d:
;CHECK: saddlp.1d
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <1 x i64> @llvm.aarch64.neon.saddlp.v1i64.v2i32(<2 x i32> %tmp1)
        ret <1 x i64> %tmp3
}

define <8 x i16> @saddlp8h(<16 x i8>* %A) nounwind {
;CHECK-LABEL: saddlp8h:
;CHECK: saddlp.8h
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8> %tmp1)
        ret <8 x i16> %tmp3
}

define <4 x i32> @saddlp4s(<8 x i16>* %A) nounwind {
;CHECK-LABEL: saddlp4s:
;CHECK: saddlp.4s
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> %tmp1)
        ret <4 x i32> %tmp3
}

define <2 x i64> @saddlp2d(<4 x i32>* %A) nounwind {
;CHECK-LABEL: saddlp2d:
;CHECK: saddlp.2d
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32> %tmp1)
        ret <2 x i64> %tmp3
}

declare <4 x i16>  @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.saddlp.v1i64.v2i32(<2 x i32>) nounwind readnone

declare <8 x i16>  @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32>) nounwind readnone

define <4 x i16> @uaddlp4h(<8 x i8>* %A) nounwind {
;CHECK-LABEL: uaddlp4h:
;CHECK: uaddlp.4h
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8> %tmp1)
        ret <4 x i16> %tmp3
}

define <2 x i32> @uaddlp2s(<4 x i16>* %A) nounwind {
;CHECK-LABEL: uaddlp2s:
;CHECK: uaddlp.2s
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16> %tmp1)
        ret <2 x i32> %tmp3
}

define <1 x i64> @uaddlp1d(<2 x i32>* %A) nounwind {
;CHECK-LABEL: uaddlp1d:
;CHECK: uaddlp.1d
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp3 = call <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32> %tmp1)
        ret <1 x i64> %tmp3
}

define <8 x i16> @uaddlp8h(<16 x i8>* %A) nounwind {
;CHECK-LABEL: uaddlp8h:
;CHECK: uaddlp.8h
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8> %tmp1)
        ret <8 x i16> %tmp3
}

define <4 x i32> @uaddlp4s(<8 x i16>* %A) nounwind {
;CHECK-LABEL: uaddlp4s:
;CHECK: uaddlp.4s
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> %tmp1)
        ret <4 x i32> %tmp3
}

define <2 x i64> @uaddlp2d(<4 x i32>* %A) nounwind {
;CHECK-LABEL: uaddlp2d:
;CHECK: uaddlp.2d
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32> %tmp1)
        ret <2 x i64> %tmp3
}

declare <4 x i16>  @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32>) nounwind readnone

declare <8 x i16>  @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32>) nounwind readnone

define <4 x i16> @sadalp4h(<8 x i8>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: sadalp4h:
;CHECK: sadalp.4h
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8> %tmp1)
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp5 = add <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @sadalp2s(<4 x i16>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: sadalp2s:
;CHECK: sadalp.2s
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16> %tmp1)
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp5 = add <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <8 x i16> @sadalp8h(<16 x i8>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: sadalp8h:
;CHECK: sadalp.8h
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8> %tmp1)
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp5 = add <8 x i16> %tmp3, %tmp4
        ret <8 x i16> %tmp5
}

define <4 x i32> @sadalp4s(<8 x i16>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: sadalp4s:
;CHECK: sadalp.4s
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> %tmp1)
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp5 = add <4 x i32> %tmp3, %tmp4
        ret <4 x i32> %tmp5
}

define <2 x i64> @sadalp2d(<4 x i32>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: sadalp2d:
;CHECK: sadalp.2d
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32> %tmp1)
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp5 = add <2 x i64> %tmp3, %tmp4
        ret <2 x i64> %tmp5
}

define <4 x i16> @uadalp4h(<8 x i8>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: uadalp4h:
;CHECK: uadalp.4h
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8> %tmp1)
        %tmp4 = load <4 x i16>, <4 x i16>* %B
        %tmp5 = add <4 x i16> %tmp3, %tmp4
        ret <4 x i16> %tmp5
}

define <2 x i32> @uadalp2s(<4 x i16>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: uadalp2s:
;CHECK: uadalp.2s
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16> %tmp1)
        %tmp4 = load <2 x i32>, <2 x i32>* %B
        %tmp5 = add <2 x i32> %tmp3, %tmp4
        ret <2 x i32> %tmp5
}

define <8 x i16> @uadalp8h(<16 x i8>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: uadalp8h:
;CHECK: uadalp.8h
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8> %tmp1)
        %tmp4 = load <8 x i16>, <8 x i16>* %B
        %tmp5 = add <8 x i16> %tmp3, %tmp4
        ret <8 x i16> %tmp5
}

define <4 x i32> @uadalp4s(<8 x i16>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: uadalp4s:
;CHECK: uadalp.4s
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> %tmp1)
        %tmp4 = load <4 x i32>, <4 x i32>* %B
        %tmp5 = add <4 x i32> %tmp3, %tmp4
        ret <4 x i32> %tmp5
}

define <2 x i64> @uadalp2d(<4 x i32>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: uadalp2d:
;CHECK: uadalp.2d
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32> %tmp1)
        %tmp4 = load <2 x i64>, <2 x i64>* %B
        %tmp5 = add <2 x i64> %tmp3, %tmp4
        ret <2 x i64> %tmp5
}

define <8 x i8> @addp_8b(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: addp_8b:
;CHECK: addp.8b
        %tmp1 = load <8 x i8>, <8 x i8>* %A
        %tmp2 = load <8 x i8>, <8 x i8>* %B
        %tmp3 = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
        ret <8 x i8> %tmp3
}

define <16 x i8> @addp_16b(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: addp_16b:
;CHECK: addp.16b
        %tmp1 = load <16 x i8>, <16 x i8>* %A
        %tmp2 = load <16 x i8>, <16 x i8>* %B
        %tmp3 = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
        ret <16 x i8> %tmp3
}

define <4 x i16> @addp_4h(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: addp_4h:
;CHECK: addp.4h
        %tmp1 = load <4 x i16>, <4 x i16>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
        ret <4 x i16> %tmp3
}

define <8 x i16> @addp_8h(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: addp_8h:
;CHECK: addp.8h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
        ret <8 x i16> %tmp3
}

define <2 x i32> @addp_2s(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: addp_2s:
;CHECK: addp.2s
        %tmp1 = load <2 x i32>, <2 x i32>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
        ret <2 x i32> %tmp3
}

define <4 x i32> @addp_4s(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: addp_4s:
;CHECK: addp.4s
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %tmp3 = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
        ret <4 x i32> %tmp3
}

define <2 x i64> @addp_2d(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: addp_2d:
;CHECK: addp.2d
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %tmp3 = call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
        ret <2 x i64> %tmp3
}

declare <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <2 x float> @faddp_2s(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: faddp_2s:
;CHECK: faddp.2s
        %tmp1 = load <2 x float>, <2 x float>* %A
        %tmp2 = load <2 x float>, <2 x float>* %B
        %tmp3 = call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
        ret <2 x float> %tmp3
}

define <4 x float> @faddp_4s(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: faddp_4s:
;CHECK: faddp.4s
        %tmp1 = load <4 x float>, <4 x float>* %A
        %tmp2 = load <4 x float>, <4 x float>* %B
        %tmp3 = call <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
        ret <4 x float> %tmp3
}

define <2 x double> @faddp_2d(<2 x double>* %A, <2 x double>* %B) nounwind {
;CHECK-LABEL: faddp_2d:
;CHECK: faddp.2d
        %tmp1 = load <2 x double>, <2 x double>* %A
        %tmp2 = load <2 x double>, <2 x double>* %B
        %tmp3 = call <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double> %tmp1, <2 x double> %tmp2)
        ret <2 x double> %tmp3
}

declare <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double>, <2 x double>) nounwind readnone

define <2 x i64> @uaddl_duprhs(<4 x i32> %lhs, i32 %rhs) {
; CHECK-LABEL: uaddl_duprhs
; CHECK-NOT: ext.16b
; CHECK: uaddl.2d
  %rhsvec.tmp = insertelement <2 x i32> undef, i32 %rhs, i32 0
  %rhsvec = insertelement <2 x i32> %rhsvec.tmp, i32 %rhs, i32 1

  %lhs.high = shufflevector <4 x i32> %lhs, <4 x i32> undef, <2 x i32> <i32 0, i32 1>

  %lhs.ext = zext <2 x i32> %lhs.high to <2 x i64>
  %rhs.ext = zext <2 x i32> %rhsvec to <2 x i64>

  %res = add <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @uaddl2_duprhs(<4 x i32> %lhs, i32 %rhs) {
; CHECK-LABEL: uaddl2_duprhs
; CHECK-NOT: ext.16b
; CHECK: uaddl2.2d
  %rhsvec.tmp = insertelement <2 x i32> undef, i32 %rhs, i32 0
  %rhsvec = insertelement <2 x i32> %rhsvec.tmp, i32 %rhs, i32 1

  %lhs.high = shufflevector <4 x i32> %lhs, <4 x i32> undef, <2 x i32> <i32 2, i32 3>

  %lhs.ext = zext <2 x i32> %lhs.high to <2 x i64>
  %rhs.ext = zext <2 x i32> %rhsvec to <2 x i64>

  %res = add <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @saddl_duplhs(i32 %lhs, <4 x i32> %rhs) {
; CHECK-LABEL: saddl_duplhs
; CHECK-NOT: ext.16b
; CHECK: saddl.2d
  %lhsvec.tmp = insertelement <2 x i32> undef, i32 %lhs, i32 0
  %lhsvec = insertelement <2 x i32> %lhsvec.tmp, i32 %lhs, i32 1

  %rhs.high = shufflevector <4 x i32> %rhs, <4 x i32> undef, <2 x i32> <i32 0, i32 1>

  %lhs.ext = sext <2 x i32> %lhsvec to <2 x i64>
  %rhs.ext = sext <2 x i32> %rhs.high to <2 x i64>

  %res = add <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @saddl2_duplhs(i32 %lhs, <4 x i32> %rhs) {
; CHECK-LABEL: saddl2_duplhs
; CHECK-NOT: ext.16b
; CHECK: saddl2.2d
  %lhsvec.tmp = insertelement <2 x i32> undef, i32 %lhs, i32 0
  %lhsvec = insertelement <2 x i32> %lhsvec.tmp, i32 %lhs, i32 1

  %rhs.high = shufflevector <4 x i32> %rhs, <4 x i32> undef, <2 x i32> <i32 2, i32 3>

  %lhs.ext = sext <2 x i32> %lhsvec to <2 x i64>
  %rhs.ext = sext <2 x i32> %rhs.high to <2 x i64>

  %res = add <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @usubl_duprhs(<4 x i32> %lhs, i32 %rhs) {
; CHECK-LABEL: usubl_duprhs
; CHECK-NOT: ext.16b
; CHECK: usubl.2d
  %rhsvec.tmp = insertelement <2 x i32> undef, i32 %rhs, i32 0
  %rhsvec = insertelement <2 x i32> %rhsvec.tmp, i32 %rhs, i32 1

  %lhs.high = shufflevector <4 x i32> %lhs, <4 x i32> undef, <2 x i32> <i32 0, i32 1>

  %lhs.ext = zext <2 x i32> %lhs.high to <2 x i64>
  %rhs.ext = zext <2 x i32> %rhsvec to <2 x i64>

  %res = sub <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @usubl2_duprhs(<4 x i32> %lhs, i32 %rhs) {
; CHECK-LABEL: usubl2_duprhs
; CHECK-NOT: ext.16b
; CHECK: usubl2.2d
  %rhsvec.tmp = insertelement <2 x i32> undef, i32 %rhs, i32 0
  %rhsvec = insertelement <2 x i32> %rhsvec.tmp, i32 %rhs, i32 1

  %lhs.high = shufflevector <4 x i32> %lhs, <4 x i32> undef, <2 x i32> <i32 2, i32 3>

  %lhs.ext = zext <2 x i32> %lhs.high to <2 x i64>
  %rhs.ext = zext <2 x i32> %rhsvec to <2 x i64>

  %res = sub <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @ssubl_duplhs(i32 %lhs, <4 x i32> %rhs) {
; CHECK-LABEL: ssubl_duplhs:
; CHECK-NOT: ext.16b
; CHECK: ssubl.2d
  %lhsvec.tmp = insertelement <2 x i32> undef, i32 %lhs, i32 0
  %lhsvec = insertelement <2 x i32> %lhsvec.tmp, i32 %lhs, i32 1

  %rhs.high = shufflevector <4 x i32> %rhs, <4 x i32> undef, <2 x i32> <i32 0, i32 1>

  %lhs.ext = sext <2 x i32> %lhsvec to <2 x i64>
  %rhs.ext = sext <2 x i32> %rhs.high to <2 x i64>

  %res = sub <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <2 x i64> @ssubl2_duplhs(i32 %lhs, <4 x i32> %rhs) {
; CHECK-LABEL: ssubl2_duplhs:
; CHECK-NOT: ext.16b
; CHECK: ssubl2.2d
  %lhsvec.tmp = insertelement <2 x i32> undef, i32 %lhs, i32 0
  %lhsvec = insertelement <2 x i32> %lhsvec.tmp, i32 %lhs, i32 1

  %rhs.high = shufflevector <4 x i32> %rhs, <4 x i32> undef, <2 x i32> <i32 2, i32 3>

  %lhs.ext = sext <2 x i32> %lhsvec to <2 x i64>
  %rhs.ext = sext <2 x i32> %rhs.high to <2 x i64>

  %res = sub <2 x i64> %lhs.ext, %rhs.ext
  ret <2 x i64> %res
}

define <8 x i8> @addhn8b_natural(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: addhn8b_natural:
;CHECK: addhn.8b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %sum = add <8 x i16> %tmp1, %tmp2
        %high_bits = lshr <8 x i16> %sum, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
        %narrowed = trunc <8 x i16> %high_bits to <8 x i8>
        ret <8 x i8> %narrowed
}

define <4 x i16> @addhn4h_natural(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: addhn4h_natural:
;CHECK: addhn.4h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %sum = add <4 x i32> %tmp1, %tmp2
        %high_bits = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
        %narrowed = trunc <4 x i32> %high_bits to <4 x i16>
        ret <4 x i16> %narrowed
}

define <2 x i32> @addhn2s_natural(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: addhn2s_natural:
;CHECK: addhn.2s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %sum = add <2 x i64> %tmp1, %tmp2
        %high_bits = lshr <2 x i64> %sum, <i64 32, i64 32>
        %narrowed = trunc <2 x i64> %high_bits to <2 x i32>
        ret <2 x i32> %narrowed
}

define <16 x i8> @addhn2_16b_natural(<8 x i8> %low, <8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: addhn2_16b_natural:
;CHECK: addhn2.16b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %sum = add <8 x i16> %tmp1, %tmp2
        %high_bits = lshr <8 x i16> %sum, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
        %narrowed = trunc <8 x i16> %high_bits to <8 x i8>
        %res = shufflevector <8 x i8> %low, <8 x i8> %narrowed, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %res
}

define <8 x i16> @addhn2_8h_natural(<4 x i16> %low, <4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: addhn2_8h_natural:
;CHECK: addhn2.8h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %sum = add <4 x i32> %tmp1, %tmp2
        %high_bits = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
        %narrowed = trunc <4 x i32> %high_bits to <4 x i16>
        %res = shufflevector <4 x i16> %low, <4 x i16> %narrowed, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %res
}

define <4 x i32> @addhn2_4s_natural(<2 x i32> %low, <2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: addhn2_4s_natural:
;CHECK: addhn2.4s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %sum = add <2 x i64> %tmp1, %tmp2
        %high_bits = lshr <2 x i64> %sum, <i64 32, i64 32>
        %narrowed = trunc <2 x i64> %high_bits to <2 x i32>
        %res = shufflevector <2 x i32> %low, <2 x i32> %narrowed, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %res
}

define <8 x i8> @subhn8b_natural(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: subhn8b_natural:
;CHECK: subhn.8b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %diff = sub <8 x i16> %tmp1, %tmp2
        %high_bits = lshr <8 x i16> %diff, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
        %narrowed = trunc <8 x i16> %high_bits to <8 x i8>
        ret <8 x i8> %narrowed
}

define <4 x i16> @subhn4h_natural(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: subhn4h_natural:
;CHECK: subhn.4h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %diff = sub <4 x i32> %tmp1, %tmp2
        %high_bits = lshr <4 x i32> %diff, <i32 16, i32 16, i32 16, i32 16>
        %narrowed = trunc <4 x i32> %high_bits to <4 x i16>
        ret <4 x i16> %narrowed
}

define <2 x i32> @subhn2s_natural(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: subhn2s_natural:
;CHECK: subhn.2s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %diff = sub <2 x i64> %tmp1, %tmp2
        %high_bits = lshr <2 x i64> %diff, <i64 32, i64 32>
        %narrowed = trunc <2 x i64> %high_bits to <2 x i32>
        ret <2 x i32> %narrowed
}

define <16 x i8> @subhn2_16b_natural(<8 x i8> %low, <8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: subhn2_16b_natural:
;CHECK: subhn2.16b
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %diff = sub <8 x i16> %tmp1, %tmp2
        %high_bits = lshr <8 x i16> %diff, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
        %narrowed = trunc <8 x i16> %high_bits to <8 x i8>
        %res = shufflevector <8 x i8> %low, <8 x i8> %narrowed, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
        ret <16 x i8> %res
}

define <8 x i16> @subhn2_8h_natural(<4 x i16> %low, <4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: subhn2_8h_natural:
;CHECK: subhn2.8h
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i32>, <4 x i32>* %B
        %diff = sub <4 x i32> %tmp1, %tmp2
        %high_bits = lshr <4 x i32> %diff, <i32 16, i32 16, i32 16, i32 16>
        %narrowed = trunc <4 x i32> %high_bits to <4 x i16>
        %res = shufflevector <4 x i16> %low, <4 x i16> %narrowed, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
        ret <8 x i16> %res
}

define <4 x i32> @subhn2_4s_natural(<2 x i32> %low, <2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: subhn2_4s_natural:
;CHECK: subhn2.4s
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i64>, <2 x i64>* %B
        %diff = sub <2 x i64> %tmp1, %tmp2
        %high_bits = lshr <2 x i64> %diff, <i64 32, i64 32>
        %narrowed = trunc <2 x i64> %high_bits to <2 x i32>
        %res = shufflevector <2 x i32> %low, <2 x i32> %narrowed, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
        ret <4 x i32> %res
}
