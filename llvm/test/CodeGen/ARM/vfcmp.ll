; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vceq\\.f32} %t | count 1
; RUN: grep {vcgt\\.f32} %t | count 9
; RUN: grep {vcge\\.f32} %t | count 5
; RUN: grep vorr %t | count 4
; RUN: grep vmvn %t | count 7

; This tests vfcmp operations that do not map directly to NEON instructions.

; une is implemented with VCEQ/VMVN
define <2 x i32> @vcunef32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp une <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; olt is implemented with VCGT
define <2 x i32> @vcoltf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp olt <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; ole is implemented with VCGE
define <2 x i32> @vcolef32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ole <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; uge is implemented with VCGT/VMVN
define <2 x i32> @vcugef32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp uge <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; ule is implemented with VCGT/VMVN
define <2 x i32> @vculef32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ule <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; ugt is implemented with VCGE/VMVN
define <2 x i32> @vcugtf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ugt <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; ult is implemented with VCGE/VMVN
define <2 x i32> @vcultf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ult <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; ueq is implemented with VCGT/VCGT/VORR/VMVN
define <2 x i32> @vcueqf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ueq <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; one is implemented with VCGT/VCGT/VORR
define <2 x i32> @vconef32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp one <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; uno is implemented with VCGT/VCGE/VORR/VMVN
define <2 x i32> @vcunof32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp uno <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; ord is implemented with VCGT/VCGE/VORR
define <2 x i32> @vcordf32(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = vfcmp ord <2 x float> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}
