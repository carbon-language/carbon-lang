; This provides optimized implementations of vstore4/8/16 for 32-bit int/uint

define void @__clc_vstore2_impl_i32__global(<2 x i32> %vec, i32 %offset,  i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = ptrtoint i32 addrspace(1)* %addr to i32
  %2 = add i32 %1, %offset
  %3 = inttoptr i32 %2 to <2 x i32> addrspace(1)*
  store <2 x i32> %vec, <2 x i32> addrspace(1)* %3, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore3_impl_i32__global(<3 x i32> %vec, i32 %offset,  i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = ptrtoint i32 addrspace(1)* %addr to i32
  %2 = add i32 %1, %offset
  %3 = inttoptr i32 %2 to <3 x i32> addrspace(1)*
  store <3 x i32> %vec, <3 x i32> addrspace(1)* %3, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore4_impl_i32__global(<4 x i32> %vec, i32 %offset,  i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = ptrtoint i32 addrspace(1)* %addr to i32
  %2 = add i32 %1, %offset
  %3 = inttoptr i32 %2 to <4 x i32> addrspace(1)*
  store <4 x i32> %vec, <4 x i32> addrspace(1)* %3, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore8_impl_i32__global(<8 x i32> %vec, i32 %offset,  i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = ptrtoint i32 addrspace(1)* %addr to i32
  %2 = add i32 %1, %offset
  %3 = inttoptr i32 %2 to <8 x i32> addrspace(1)*
  store <8 x i32> %vec, <8 x i32> addrspace(1)* %3, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore16_impl_i32__global(<16 x i32> %vec, i32 %offset,  i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = ptrtoint i32 addrspace(1)* %addr to i32
  %2 = add i32 %1, %offset
  %3 = inttoptr i32 %2 to <16 x i32> addrspace(1)*
  store <16 x i32> %vec, <16 x i32> addrspace(1)* %3, align 4, !tbaa !3
  ret void
}


!1 = metadata !{metadata !"char", metadata !5}
!2 = metadata !{metadata !"short", metadata !5}
!3 = metadata !{metadata !"int", metadata !5}
!4 = metadata !{metadata !"long", metadata !5}
!5 = metadata !{metadata !"omnipotent char", metadata !6}
!6 = metadata !{metadata !"Simple C/C++ TBAA"}

