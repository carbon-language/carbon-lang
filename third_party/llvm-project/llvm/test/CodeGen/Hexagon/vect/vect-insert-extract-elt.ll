; RUN: llc -march=hexagon < %s
; Used to fail with an infinite recursion in the insn selection.
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon-unknown-linux-gnu"

%struct.elt = type { [2 x [4 x %struct.block]] }
%struct.block = type { [2 x i16] }

define void @foo(%struct.elt* noalias nocapture %p0, %struct.elt* noalias nocapture %p1) nounwind {
entry:
  %arrayidx1 = getelementptr inbounds %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 3
  %arrayidx4 = getelementptr inbounds %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 2
  %arrayidx7 = getelementptr inbounds %struct.elt, %struct.elt* %p0, i32 0, i32 0, i32 0, i32 3
  %0 = bitcast %struct.block* %arrayidx7 to i32*
  %1 = bitcast %struct.block* %arrayidx4 to i32*
  %2 = load i32, i32* %0, align 4
  store i32 %2, i32* %1, align 4
  %3 = bitcast %struct.block* %arrayidx1 to i32*
  store i32 %2, i32* %3, align 4
  %arrayidx10 = getelementptr inbounds %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 1
  %arrayidx16 = getelementptr inbounds %struct.elt, %struct.elt* %p0, i32 0, i32 0, i32 0, i32 2
  %4 = bitcast %struct.block* %arrayidx16 to i32*
  %5 = bitcast %struct.elt* %p1 to i32*
  %6 = load i32, i32* %4, align 4
  store i32 %6, i32* %5, align 4
  %7 = bitcast %struct.block* %arrayidx10 to i32*
  store i32 %6, i32* %7, align 4
  %p_arrayidx26 = getelementptr %struct.elt, %struct.elt* %p0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %p_arrayidx2632 = getelementptr %struct.elt, %struct.elt* %p0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  %p_arrayidx2633 = getelementptr %struct.elt, %struct.elt* %p0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 1
  %p_arrayidx2634 = getelementptr %struct.elt, %struct.elt* %p0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 1
  %p_arrayidx20 = getelementptr %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %p_arrayidx2035 = getelementptr %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  %p_arrayidx2036 = getelementptr %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 2, i32 0, i32 1
  %p_arrayidx2037 = getelementptr %struct.elt, %struct.elt* %p1, i32 0, i32 0, i32 0, i32 3, i32 0, i32 1
  %8 = lshr i32 %6, 16
  %9 = trunc i32 %8 to i16
  %_p_vec_ = insertelement <4 x i16> undef, i16 %9, i32 0
  %_p_vec_39 = insertelement <4 x i16> %_p_vec_, i16 %9, i32 1
  %10 = lshr i32 %2, 16
  %11 = trunc i32 %10 to i16
  %_p_vec_41 = insertelement <4 x i16> %_p_vec_39, i16 %11, i32 2
  %_p_vec_43 = insertelement <4 x i16> %_p_vec_41, i16 %11, i32 3
  %shlp_vec = shl <4 x i16> %_p_vec_43, <i16 1, i16 1, i16 1, i16 1>
  %12 = extractelement <4 x i16> %shlp_vec, i32 0
  store i16 %12, i16* %p_arrayidx20, align 2
  %13 = extractelement <4 x i16> %shlp_vec, i32 1
  store i16 %13, i16* %p_arrayidx2035, align 2
  %14 = extractelement <4 x i16> %shlp_vec, i32 2
  store i16 %14, i16* %p_arrayidx2036, align 2
  %15 = extractelement <4 x i16> %shlp_vec, i32 3
  store i16 %15, i16* %p_arrayidx2037, align 2
  %_p_scalar_44 = load i16, i16* %p_arrayidx26, align 2
  %_p_vec_45 = insertelement <4 x i16> undef, i16 %_p_scalar_44, i32 0
  %_p_scalar_46 = load i16, i16* %p_arrayidx2632, align 2
  %_p_vec_47 = insertelement <4 x i16> %_p_vec_45, i16 %_p_scalar_46, i32 1
  %_p_scalar_48 = load i16, i16* %p_arrayidx2633, align 2
  %_p_vec_49 = insertelement <4 x i16> %_p_vec_47, i16 %_p_scalar_48, i32 2
  %_p_scalar_50 = load i16, i16* %p_arrayidx2634, align 2
  %_p_vec_51 = insertelement <4 x i16> %_p_vec_49, i16 %_p_scalar_50, i32 3
  %shl28p_vec = shl <4 x i16> %_p_vec_51, <i16 1, i16 1, i16 1, i16 1>
  %16 = extractelement <4 x i16> %shl28p_vec, i32 0
  store i16 %16, i16* %p_arrayidx26, align 2
  %17 = extractelement <4 x i16> %shl28p_vec, i32 1
  store i16 %17, i16* %p_arrayidx2632, align 2
  %18 = extractelement <4 x i16> %shl28p_vec, i32 2
  store i16 %18, i16* %p_arrayidx2633, align 2
  %19 = extractelement <4 x i16> %shl28p_vec, i32 3
  store i16 %19, i16* %p_arrayidx2634, align 2
  ret void
}
