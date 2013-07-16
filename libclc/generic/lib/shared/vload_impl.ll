; This provides optimized implementations of vload4/8/16 for 32-bit int/uint

define <2 x i32> @__clc_vload2_i32__addr1(i32 addrspace(1)* nocapture %addr) nounwind readonly alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <2 x i32> addrspace(1)*
  %2 = load <2 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret <2 x i32> %2
}

define <3 x i32> @__clc_vload3_i32__addr1(i32 addrspace(1)* nocapture %addr) nounwind readonly alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <3 x i32> addrspace(1)*
  %2 = load <3 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret <3 x i32> %2
}

define <4 x i32> @__clc_vload4_i32__addr1(i32 addrspace(1)* nocapture %addr) nounwind readonly alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <4 x i32> addrspace(1)*
  %2 = load <4 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret <4 x i32> %2
}

define <8 x i32> @__clc_vload8_i32__addr1(i32 addrspace(1)* nocapture %addr) nounwind readonly alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <8 x i32> addrspace(1)*
  %2 = load <8 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret <8 x i32> %2
}

define <16 x i32> @__clc_vload16_i32__addr1(i32 addrspace(1)* nocapture %addr) nounwind readonly alwaysinline {
  %1 = bitcast i32 addrspace(1)* %addr to <16 x i32> addrspace(1)*
  %2 = load <16 x i32> addrspace(1)* %1, align 4, !tbaa !3
  ret <16 x i32> %2
}

!1 = metadata !{metadata !"char", metadata !5}
!2 = metadata !{metadata !"short", metadata !5}
!3 = metadata !{metadata !"int", metadata !5}
!4 = metadata !{metadata !"long", metadata !5}
!5 = metadata !{metadata !"omnipotent char", metadata !6}
!6 = metadata !{metadata !"Simple C/C++ TBAA"}

