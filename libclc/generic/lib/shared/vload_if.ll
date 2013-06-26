;Start int global vload

declare <2 x i32> @__clc_vload2_impl_i32__global(i32 %x, i32 %y)
declare <3 x i32> @__clc_vload3_impl_i32__global(i32 %x, i32 %y)
declare <4 x i32> @__clc_vload4_impl_i32__global(i32 %x, i32 %y)
declare <8 x i32> @__clc_vload8_impl_i32__global(i32 %x, i32 %y)
declare <16 x i32> @__clc_vload16_impl_i32__global(i32 %x, i32 %y)

define <2 x i32> @__clc_vload2_int__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <2 x i32> @__clc_vload2_impl_i32__global(i32 %x, i32 %y)
  ret <2 x i32> %call
}

define <3 x i32> @__clc_vload3_int__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <3 x i32> @__clc_vload3_impl_i32__global(i32 %x, i32 %y)
  ret <3 x i32> %call
}

define <4 x i32> @__clc_vload4_int__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <4 x i32> @__clc_vload4_impl_i32__global(i32 %x, i32 %y)
  ret <4 x i32> %call
}

define <8 x i32> @__clc_vload8_int__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <8 x i32> @__clc_vload8_impl_i32__global(i32 %x, i32 %y)
  ret <8 x i32> %call
}

define <16 x i32> @__clc_vload16_int__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <16 x i32> @__clc_vload16_impl_i32__global(i32 %x, i32 %y)
  ret <16 x i32> %call
}


;Start uint global vload

define <2 x i32> @__clc_vload2_uint__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <2 x i32> @__clc_vload2_impl_i32__global(i32 %x, i32 %y)
  ret <2 x i32> %call
}

define <3 x i32> @__clc_vload3_uint__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <3 x i32> @__clc_vload3_impl_i32__global(i32 %x, i32 %y)
  ret <3 x i32> %call
}

define <4 x i32> @__clc_vload4_uint__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <4 x i32> @__clc_vload4_impl_i32__global(i32 %x, i32 %y)
  ret <4 x i32> %call
}

define <8 x i32> @__clc_vload8_uint__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <8 x i32> @__clc_vload8_impl_i32__global(i32 %x, i32 %y)
  ret <8 x i32> %call
}

define <16 x i32> @__clc_vload16_uint__global(i32 %x, i32 %y) nounwind readonly alwaysinline {
  %call = call <16 x i32> @__clc_vload16_impl_i32__global(i32 %x, i32 %y)
  ret <16 x i32> %call
}
