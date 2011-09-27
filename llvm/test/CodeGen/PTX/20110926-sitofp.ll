; RUN: llc < %s -march=ptx32 | FileCheck %s

@A = common global [1536 x [1536 x float]] zeroinitializer, align 4
@B = common global [1536 x [1536 x float]] zeroinitializer, align 4

define internal ptx_device void @init_array(i32 %x, i32 %y) {
  %arrayidx103 = getelementptr [1536 x [1536 x float]]* @A, i32 0, i32 %x, i32 %y
  %arrayidx224 = getelementptr [1536 x [1536 x float]]* @B, i32 0, i32 %x, i32 %y
  %mul5 = mul i32 %x, %y
  %rem = srem i32 %mul5, 1024
  %add = add nsw i32 %rem, 1
; CHECK: cvt.rn.f64.s32 %fd{{[0-9]+}}, %r{{[0-9]+}}
  %conv = sitofp i32 %add to double
  %div = fmul double %conv, 5.000000e-01
  %conv7 = fptrunc double %div to float
  store float %conv7, float* %arrayidx103, align 4
  %rem14 = srem i32 %mul5, 1024
  %add15 = add nsw i32 %rem14, 1
  %conv16 = sitofp i32 %add15 to double
  %div17 = fmul double %conv16, 5.000000e-01
  %conv18 = fptrunc double %div17 to float
  store float %conv18, float* %arrayidx224, align 4
  ret void
}
