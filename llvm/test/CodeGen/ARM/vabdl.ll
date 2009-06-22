; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vabdl\\.s8} %t | count 1
; RUN: grep {vabdl\\.s16} %t | count 1
; RUN: grep {vabdl\\.s32} %t | count 1
; RUN: grep {vabdl\\.u8} %t | count 1
; RUN: grep {vabdl\\.u16} %t | count 1
; RUN: grep {vabdl\\.u32} %t | count 1

define <8 x i16> @vabdls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vabdls.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vabdls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vabdls.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vabdls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vabdls.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vabdlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vabdlu.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vabdlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vabdlu.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vabdlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vabdlu.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vabdls.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabdls.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vabdls.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vabdlu.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabdlu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vabdlu.v2i64(<2 x i32>, <2 x i32>) nounwind readnone
