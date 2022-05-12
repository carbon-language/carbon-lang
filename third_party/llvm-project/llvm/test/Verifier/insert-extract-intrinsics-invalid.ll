; RUN: not opt -verify -S < %s 2>&1 >/dev/null | FileCheck %s

;
; Test that extractions/insertion indices are validated.
;

; CHECK: experimental_vector_extract index must be a constant multiple of the result type's known minimum vector length.
define <4 x i32> @extract_idx_not_constant_multiple(<8 x i32> %vec) {
  %1 = call <4 x i32> @llvm.experimental.vector.extract.v4i32.v8i32(<8 x i32> %vec, i64 1)
  ret <4 x i32> %1
}

; CHECK: experimental_vector_insert index must be a constant multiple of the subvector's known minimum vector length.
define <8 x i32> @insert_idx_not_constant_multiple(<8 x i32> %vec, <4 x i32> %subvec) {
  %1 = call <8 x i32> @llvm.experimental.vector.insert.v8i32.v4i32(<8 x i32> %vec, <4 x i32> %subvec, i64 2)
  ret <8 x i32> %1
}

;
; Test that extractions/insertions which 'overrun' are captured.
;

; CHECK: experimental_vector_extract would overrun.
define <3 x i32> @extract_overrun_fixed_fixed(<8 x i32> %vec) {
  %1 = call <3 x i32> @llvm.experimental.vector.extract.v8i32.v3i32(<8 x i32> %vec, i64 6)
  ret <3 x i32> %1
}

; CHECK: experimental_vector_extract would overrun.
define <vscale x 3 x i32> @extract_overrun_scalable_scalable(<vscale x 8 x i32> %vec) {
  %1 = call <vscale x 3 x i32> @llvm.experimental.vector.extract.nxv8i32.nxv3i32(<vscale x 8 x i32> %vec, i64 6)
  ret <vscale x 3 x i32> %1
}

; We cannot statically check whether or not an extraction of a fixed vector
; from a scalable vector would overrun, because we can't compare the sizes of
; the two. Therefore, this function should not raise verifier errors.
; CHECK-NOT: experimental_vector_extract
define <3 x i32> @extract_overrun_scalable_fixed(<vscale x 8 x i32> %vec) {
  %1 = call <3 x i32> @llvm.experimental.vector.extract.nxv8i32.v3i32(<vscale x 8 x i32> %vec, i64 6)
  ret <3 x i32> %1
}

; CHECK: subvector operand of experimental_vector_insert would overrun the vector being inserted into.
define <8 x i32> @insert_overrun_fixed_fixed(<8 x i32> %vec, <3 x i32> %subvec) {
  %1 = call <8 x i32> @llvm.experimental.vector.insert.v8i32.v3i32(<8 x i32> %vec, <3 x i32> %subvec, i64 6)
  ret <8 x i32> %1
}

; CHECK: subvector operand of experimental_vector_insert would overrun the vector being inserted into.
define <vscale x 8 x i32> @insert_overrun_scalable_scalable(<vscale x 8 x i32> %vec, <vscale x 3 x i32> %subvec) {
  %1 = call <vscale x 8 x i32> @llvm.experimental.vector.insert.nxv8i32.nxv3i32(<vscale x 8 x i32> %vec, <vscale x 3 x i32> %subvec, i64 6)
  ret <vscale x 8 x i32> %1
}

; We cannot statically check whether or not an insertion of a fixed vector into
; a scalable vector would overrun, because we can't compare the sizes of the
; two. Therefore, this function should not raise verifier errors.
; CHECK-NOT: experimental_vector_insert
define <vscale x 8 x i32> @insert_overrun_scalable_fixed(<vscale x 8 x i32> %vec, <3 x i32> %subvec) {
  %1 = call <vscale x 8 x i32> @llvm.experimental.vector.insert.nxv8i32.v3i32(<vscale x 8 x i32> %vec, <3 x i32> %subvec, i64 6)
  ret <vscale x 8 x i32> %1
}

declare <vscale x 3 x i32> @llvm.experimental.vector.extract.nxv8i32.nxv3i32(<vscale x 8 x i32>, i64)
declare <vscale x 8 x i32> @llvm.experimental.vector.insert.nxv8i32.nxv3i32(<vscale x 8 x i32>, <vscale x 3 x i32>, i64)
declare <vscale x 8 x i32> @llvm.experimental.vector.insert.nxv8i32.v3i32(<vscale x 8 x i32>, <3 x i32>, i64)
declare <3 x i32> @llvm.experimental.vector.extract.nxv8i32.v3i32(<vscale x 8 x i32>, i64)
declare <3 x i32> @llvm.experimental.vector.extract.v8i32.v3i32(<8 x i32>, i64)
declare <4 x i32> @llvm.experimental.vector.extract.v4i32.v8i32(<8 x i32>, i64)
declare <8 x i32> @llvm.experimental.vector.insert.v8i32.v3i32(<8 x i32>, <3 x i32>, i64)
declare <8 x i32> @llvm.experimental.vector.insert.v8i32.v4i32(<8 x i32>, <4 x i32>, i64)
