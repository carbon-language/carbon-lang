; This provides optimized implementations of vstore4/8/16 for 32-bit int/uint

define void @__clc_vstore2_i32__addr1(<2 x i32> %vec, i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <2 x i32> addrspace(1)*
  store <2 x i32> %vec, <2 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore3_i32__addr1(<3 x i32> %vec, i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <3 x i32> addrspace(1)*
  store <3 x i32> %vec, <3 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore4_i32__addr1(<4 x i32> %vec, i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <4 x i32> addrspace(1)*
  store <4 x i32> %vec, <4 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore8_i32__addr1(<8 x i32> %vec, i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <8 x i32> addrspace(1)*
  store <8 x i32> %vec, <8 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret void
}

define void @__clc_vstore16_i32__addr1(<16 x i32> %vec, i32 addrspace(1)* nocapture %addr) nounwind alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <16 x i32> addrspace(1)*
  store <16 x i32> %vec, <16 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret void
}

!1 = metadata !{metadata !"char", metadata !5}
!2 = metadata !{metadata !"short", metadata !5}
!3 = metadata !{metadata !"int", metadata !5}
!4 = metadata !{metadata !"long", metadata !5}
!5 = metadata !{metadata !"omnipotent char", metadata !6}
!6 = metadata !{metadata !"Simple C/C++ TBAA"}

