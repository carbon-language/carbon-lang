;Start int global vstore

declare void @__clc_vstore2_impl_i32__global(<2 x i32> %vec, i32 %x, i32 %y)
declare void @__clc_vstore3_impl_i32__global(<3 x i32> %vec, i32 %x, i32 %y)
declare void @__clc_vstore4_impl_i32__global(<4 x i32> %vec, i32 %x, i32 %y)
declare void @__clc_vstore8_impl_i32__global(<8 x i32> %vec, i32 %x, i32 %y)
declare void @__clc_vstore16_impl_i32__global(<16 x i32> %vec, i32 %x, i32 %y)

define void @__clc_vstore2_int__global(<2 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore2_impl_i32__global(<2 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore3_int__global(<3 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore3_impl_i32__global(<3 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore4_int__global(<4 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore4_impl_i32__global(<4 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore8_int__global(<8 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore8_impl_i32__global(<8 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore16_int__global(<16 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore16_impl_i32__global(<16 x i32> %vec, i32 %x, i32 %y)
  ret void
}


;Start uint global vstore
define void @__clc_vstore2_uint__global(<2 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore2_impl_i32__global(<2 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore3_uint__global(<3 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore3_impl_i32__global(<3 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore4_uint__global(<4 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore4_impl_i32__global(<4 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore8_uint__global(<8 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore8_impl_i32__global(<8 x i32> %vec, i32 %x, i32 %y)
  ret void
}

define void @__clc_vstore16_uint__global(<16 x i32> %vec, i32 %x, i32 %y) nounwind alwaysinline {
  call void @__clc_vstore16_impl_i32__global(<16 x i32> %vec, i32 %x, i32 %y)
  ret void
}