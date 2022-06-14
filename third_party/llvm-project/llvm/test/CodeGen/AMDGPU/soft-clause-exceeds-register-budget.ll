; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck %s

define protected amdgpu_kernel void @excess_soft_clause_reg_pressure(float addrspace(4)* %wei_ptr, float addrspace(1)* %out_ptr, float addrspace(1)* %in) {
; CHECK-LABEL: excess_soft_clause_reg_pressure:
; CHECK:  BB0_1: ; %for.cond28.preheader
; CHECK:         s_load_dwordx16
; CHECK-NEXT:    s_load_dwordx16

; CHECK:         global_load_dword
; CHECK-NEXT:    global_load_dword
; CHECK-NEXT:    global_load_dword
; CHECK-NEXT:    global_load_dword

; CHECK:         s_load_dwordx16
; CHECK-NEXT:    s_load_dwordx16

; CHECK-NOT: v_writelane_b32
; CHECK-NOT: v_readlane_b32

; CHECK:         s_load_dwordx16
; CHECK:         s_load_dwordx16
; CHECK:         s_load_dwordx16

; CHECK-NOT: v_writelane_b32
; CHECK-NOT: v_readlane_b32
entry:
  %i = tail call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %i1 = bitcast i8 addrspace(4)* %i to i64 addrspace(4)*
  %i2 = load i64, i64 addrspace(4)* %i1, align 8
  %i3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %i4 = shl i32 %i3, 8
  %i5 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !5
  %i6 = add i32 %i4, %i5
  %i7 = trunc i64 %i2 to i32
  %conv = add i32 %i6, %i7
  %conv.frozen = freeze i32 %conv
  %div = udiv i32 %conv.frozen, 49
  %add.ptr22 = getelementptr inbounds float, float addrspace(4)* %wei_ptr, i64 undef
  %in.ptr1 = getelementptr inbounds float, float addrspace(1)* %in, i32 %i5
  br label %for.cond28.preheader

for.cond28.preheader:                             ; preds = %for.cond28.preheader, %entry
  %accum.sroa.110.0 = phi float [ 0.000000e+00, %entry ], [ %i251, %for.cond28.preheader ]
  %accum.sroa.106.0 = phi float [ 0.000000e+00, %entry ], [ %i247, %for.cond28.preheader ]
  %accum.sroa.102.0 = phi float [ 0.000000e+00, %entry ], [ %i243, %for.cond28.preheader ]
  %accum.sroa.98.0 = phi float [ 0.000000e+00, %entry ], [ %i239, %for.cond28.preheader ]
  %accum.sroa.94.0 = phi float [ 0.000000e+00, %entry ], [ %i235, %for.cond28.preheader ]
  %accum.sroa.90.0 = phi float [ 0.000000e+00, %entry ], [ %i231, %for.cond28.preheader ]
  %accum.sroa.86.0 = phi float [ 0.000000e+00, %entry ], [ %i227, %for.cond28.preheader ]
  %accum.sroa.82.0 = phi float [ 0.000000e+00, %entry ], [ %i223, %for.cond28.preheader ]
  %accum.sroa.78.0 = phi float [ 0.000000e+00, %entry ], [ %i219, %for.cond28.preheader ]
  %accum.sroa.74.0 = phi float [ 0.000000e+00, %entry ], [ %i215, %for.cond28.preheader ]
  %accum.sroa.70.0 = phi float [ 0.000000e+00, %entry ], [ %i211, %for.cond28.preheader ]
  %accum.sroa.66.0 = phi float [ 0.000000e+00, %entry ], [ %i207, %for.cond28.preheader ]
  %accum.sroa.62.0 = phi float [ 0.000000e+00, %entry ], [ %i203, %for.cond28.preheader ]
  %accum.sroa.58.0 = phi float [ 0.000000e+00, %entry ], [ %i199, %for.cond28.preheader ]
  %accum.sroa.54.0 = phi float [ 0.000000e+00, %entry ], [ %i195, %for.cond28.preheader ]
  %accum.sroa.50.0 = phi float [ 0.000000e+00, %entry ], [ %i191, %for.cond28.preheader ]
  %accum.sroa.46.0 = phi float [ 0.000000e+00, %entry ], [ %i187, %for.cond28.preheader ]
  %accum.sroa.42.0 = phi float [ 0.000000e+00, %entry ], [ %i183, %for.cond28.preheader ]
  %accum.sroa.38.0 = phi float [ 0.000000e+00, %entry ], [ %i179, %for.cond28.preheader ]
  %accum.sroa.34.0 = phi float [ 0.000000e+00, %entry ], [ %i175, %for.cond28.preheader ]
  %accum.sroa.30.0 = phi float [ 0.000000e+00, %entry ], [ %i171, %for.cond28.preheader ]
  %accum.sroa.26.0 = phi float [ 0.000000e+00, %entry ], [ %i167, %for.cond28.preheader ]
  %accum.sroa.22.0 = phi float [ 0.000000e+00, %entry ], [ %i163, %for.cond28.preheader ]
  %accum.sroa.18.0 = phi float [ 0.000000e+00, %entry ], [ %i159, %for.cond28.preheader ]
  %accum.sroa.14.0 = phi float [ 0.000000e+00, %entry ], [ %i155, %for.cond28.preheader ]
  %accum.sroa.10.0 = phi float [ 0.000000e+00, %entry ], [ %i151, %for.cond28.preheader ]
  %accum.sroa.6.0 = phi float [ 0.000000e+00, %entry ], [ %i147, %for.cond28.preheader ]
  %accum.sroa.0.0 = phi float [ 0.000000e+00, %entry ], [ %i143, %for.cond28.preheader ]
  %accum.sroa.114.0 = phi float [ 0.000000e+00, %entry ], [ %i255, %for.cond28.preheader ]
  %accum.sroa.118.0 = phi float [ 0.000000e+00, %entry ], [ %i259, %for.cond28.preheader ]
  %accum.sroa.122.0 = phi float [ 0.000000e+00, %entry ], [ %i263, %for.cond28.preheader ]
  %accum.sroa.126.0 = phi float [ 0.000000e+00, %entry ], [ %i267, %for.cond28.preheader ]
  %i_ptr.0288 = phi float addrspace(1)* [ %in.ptr1, %entry ], [ %add.ptr47.3, %for.cond28.preheader ]
  %w_ptr.0287 = phi float addrspace(4)* [ %add.ptr22, %entry ], [ %add.ptr74, %for.cond28.preheader ]
  %ci.0286 = phi i32 [ 0, %entry ], [ %inc116, %for.cond28.preheader ]
  %i8 = load float, float addrspace(1)* %i_ptr.0288, align 4
  %add.ptr47 = getelementptr inbounds float, float addrspace(1)* %i_ptr.0288, i64 49
  %i9 = load float, float addrspace(1)* %add.ptr47, align 4
  %add.ptr47.1 = getelementptr inbounds float, float addrspace(1)* %i_ptr.0288, i64 98
  %i10 = load float, float addrspace(1)* %add.ptr47.1, align 4
  %add.ptr47.2 = getelementptr inbounds float, float addrspace(1)* %i_ptr.0288, i64 147
  %i11 = load float, float addrspace(1)* %add.ptr47.2, align 4
  %i12 = load float, float addrspace(4)* %w_ptr.0287, align 4
  %add.ptr66 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1024
  %i13 = load float, float addrspace(4)* %add.ptr66, align 4
  %add.ptr66.1 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2048
  %i14 = load float, float addrspace(4)* %add.ptr66.1, align 4
  %add.ptr66.2 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3072
  %i15 = load float, float addrspace(4)* %add.ptr66.2, align 4
  %add.ptr70 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1
  %i16 = load float, float addrspace(4)* %add.ptr70, align 4
  %add.ptr66.1291 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1025
  %i17 = load float, float addrspace(4)* %add.ptr66.1291, align 4
  %add.ptr66.1.1 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2049
  %i18 = load float, float addrspace(4)* %add.ptr66.1.1, align 4
  %add.ptr66.2.1 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3073
  %i19 = load float, float addrspace(4)* %add.ptr66.2.1, align 4
  %add.ptr70.1 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2
  %i20 = load float, float addrspace(4)* %add.ptr70.1, align 4
  %add.ptr66.2293 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1026
  %i21 = load float, float addrspace(4)* %add.ptr66.2293, align 4
  %add.ptr66.1.2 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2050
  %i22 = load float, float addrspace(4)* %add.ptr66.1.2, align 4
  %add.ptr66.2.2 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3074
  %i23 = load float, float addrspace(4)* %add.ptr66.2.2, align 4
  %add.ptr70.2 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3
  %i24 = load float, float addrspace(4)* %add.ptr70.2, align 4
  %add.ptr66.3 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1027
  %i25 = load float, float addrspace(4)* %add.ptr66.3, align 4
  %add.ptr66.1.3 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2051
  %i26 = load float, float addrspace(4)* %add.ptr66.1.3, align 4
  %add.ptr66.2.3 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3075
  %i27 = load float, float addrspace(4)* %add.ptr66.2.3, align 4
  %add.ptr70.3 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 4
  %i28 = load float, float addrspace(4)* %add.ptr70.3, align 4
  %add.ptr66.4 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1028
  %i29 = load float, float addrspace(4)* %add.ptr66.4, align 4
  %add.ptr66.1.4 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2052
  %i30 = load float, float addrspace(4)* %add.ptr66.1.4, align 4
  %add.ptr66.2.4 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3076
  %i31 = load float, float addrspace(4)* %add.ptr66.2.4, align 4
  %add.ptr70.4 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 5
  %i32 = load float, float addrspace(4)* %add.ptr70.4, align 4
  %add.ptr66.5 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1029
  %i33 = load float, float addrspace(4)* %add.ptr66.5, align 4
  %add.ptr66.1.5 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2053
  %i34 = load float, float addrspace(4)* %add.ptr66.1.5, align 4
  %add.ptr66.2.5 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3077
  %i35 = load float, float addrspace(4)* %add.ptr66.2.5, align 4
  %add.ptr70.5 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 6
  %i36 = load float, float addrspace(4)* %add.ptr70.5, align 4
  %add.ptr66.6 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1030
  %i37 = load float, float addrspace(4)* %add.ptr66.6, align 4
  %add.ptr66.1.6 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2054
  %i38 = load float, float addrspace(4)* %add.ptr66.1.6, align 4
  %add.ptr66.2.6 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3078
  %i39 = load float, float addrspace(4)* %add.ptr66.2.6, align 4
  %add.ptr70.6 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 7
  %i40 = load float, float addrspace(4)* %add.ptr70.6, align 4
  %add.ptr66.7 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1031
  %i41 = load float, float addrspace(4)* %add.ptr66.7, align 4
  %add.ptr66.1.7 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2055
  %i42 = load float, float addrspace(4)* %add.ptr66.1.7, align 4
  %add.ptr66.2.7 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3079
  %i43 = load float, float addrspace(4)* %add.ptr66.2.7, align 4
  %add.ptr70.7 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 8
  %i44 = load float, float addrspace(4)* %add.ptr70.7, align 4
  %add.ptr66.8 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1032
  %i45 = load float, float addrspace(4)* %add.ptr66.8, align 4
  %add.ptr66.1.8 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2056
  %i46 = load float, float addrspace(4)* %add.ptr66.1.8, align 4
  %add.ptr66.2.8 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3080
  %i47 = load float, float addrspace(4)* %add.ptr66.2.8, align 4
  %add.ptr70.8 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 9
  %i48 = load float, float addrspace(4)* %add.ptr70.8, align 4
  %add.ptr66.9 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1033
  %i49 = load float, float addrspace(4)* %add.ptr66.9, align 4
  %add.ptr66.1.9 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2057
  %i50 = load float, float addrspace(4)* %add.ptr66.1.9, align 4
  %add.ptr66.2.9 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3081
  %i51 = load float, float addrspace(4)* %add.ptr66.2.9, align 4
  %add.ptr70.9 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 10
  %i52 = load float, float addrspace(4)* %add.ptr70.9, align 4
  %add.ptr66.10 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1034
  %i53 = load float, float addrspace(4)* %add.ptr66.10, align 4
  %add.ptr66.1.10 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2058
  %i54 = load float, float addrspace(4)* %add.ptr66.1.10, align 4
  %add.ptr66.2.10 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3082
  %i55 = load float, float addrspace(4)* %add.ptr66.2.10, align 4
  %add.ptr70.10 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 11
  %i56 = load float, float addrspace(4)* %add.ptr70.10, align 4
  %add.ptr66.11 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1035
  %i57 = load float, float addrspace(4)* %add.ptr66.11, align 4
  %add.ptr66.1.11 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2059
  %i58 = load float, float addrspace(4)* %add.ptr66.1.11, align 4
  %add.ptr66.2.11 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3083
  %i59 = load float, float addrspace(4)* %add.ptr66.2.11, align 4
  %add.ptr70.11 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 12
  %i60 = load float, float addrspace(4)* %add.ptr70.11, align 4
  %add.ptr66.12 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1036
  %i61 = load float, float addrspace(4)* %add.ptr66.12, align 4
  %add.ptr66.1.12 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2060
  %i62 = load float, float addrspace(4)* %add.ptr66.1.12, align 4
  %add.ptr66.2.12 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3084
  %i63 = load float, float addrspace(4)* %add.ptr66.2.12, align 4
  %add.ptr70.12 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 13
  %i64 = load float, float addrspace(4)* %add.ptr70.12, align 4
  %add.ptr66.13 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1037
  %i65 = load float, float addrspace(4)* %add.ptr66.13, align 4
  %add.ptr66.1.13 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2061
  %i66 = load float, float addrspace(4)* %add.ptr66.1.13, align 4
  %add.ptr66.2.13 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3085
  %i67 = load float, float addrspace(4)* %add.ptr66.2.13, align 4
  %add.ptr70.13 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 14
  %i68 = load float, float addrspace(4)* %add.ptr70.13, align 4
  %add.ptr66.14 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1038
  %i69 = load float, float addrspace(4)* %add.ptr66.14, align 4
  %add.ptr66.1.14 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2062
  %i70 = load float, float addrspace(4)* %add.ptr66.1.14, align 4
  %add.ptr66.2.14 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3086
  %i71 = load float, float addrspace(4)* %add.ptr66.2.14, align 4
  %add.ptr70.14 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 15
  %i72 = load float, float addrspace(4)* %add.ptr70.14, align 4
  %add.ptr66.15 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1039
  %i73 = load float, float addrspace(4)* %add.ptr66.15, align 4
  %add.ptr66.1.15 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2063
  %i74 = load float, float addrspace(4)* %add.ptr66.1.15, align 4
  %add.ptr66.2.15 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3087
  %i75 = load float, float addrspace(4)* %add.ptr66.2.15, align 4
  %add.ptr70.15 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 16
  %i76 = load float, float addrspace(4)* %add.ptr70.15, align 4
  %add.ptr66.16 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1040
  %i77 = load float, float addrspace(4)* %add.ptr66.16, align 4
  %add.ptr66.1.16 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2064
  %i78 = load float, float addrspace(4)* %add.ptr66.1.16, align 4
  %add.ptr66.2.16 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3088
  %i79 = load float, float addrspace(4)* %add.ptr66.2.16, align 4
  %add.ptr70.16 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 17
  %i80 = load float, float addrspace(4)* %add.ptr70.16, align 4
  %add.ptr66.17 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1041
  %i81 = load float, float addrspace(4)* %add.ptr66.17, align 4
  %add.ptr66.1.17 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2065
  %i82 = load float, float addrspace(4)* %add.ptr66.1.17, align 4
  %add.ptr66.2.17 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3089
  %i83 = load float, float addrspace(4)* %add.ptr66.2.17, align 4
  %add.ptr70.17 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 18
  %i84 = load float, float addrspace(4)* %add.ptr70.17, align 4
  %add.ptr66.18 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1042
  %i85 = load float, float addrspace(4)* %add.ptr66.18, align 4
  %add.ptr66.1.18 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2066
  %i86 = load float, float addrspace(4)* %add.ptr66.1.18, align 4
  %add.ptr66.2.18 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3090
  %i87 = load float, float addrspace(4)* %add.ptr66.2.18, align 4
  %add.ptr70.18 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 19
  %i88 = load float, float addrspace(4)* %add.ptr70.18, align 4
  %add.ptr66.19 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1043
  %i89 = load float, float addrspace(4)* %add.ptr66.19, align 4
  %add.ptr66.1.19 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2067
  %i90 = load float, float addrspace(4)* %add.ptr66.1.19, align 4
  %add.ptr66.2.19 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3091
  %i91 = load float, float addrspace(4)* %add.ptr66.2.19, align 4
  %add.ptr70.19 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 20
  %i92 = load float, float addrspace(4)* %add.ptr70.19, align 4
  %add.ptr66.20 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1044
  %i93 = load float, float addrspace(4)* %add.ptr66.20, align 4
  %add.ptr66.1.20 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2068
  %i94 = load float, float addrspace(4)* %add.ptr66.1.20, align 4
  %add.ptr66.2.20 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3092
  %i95 = load float, float addrspace(4)* %add.ptr66.2.20, align 4
  %add.ptr70.20 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 21
  %i96 = load float, float addrspace(4)* %add.ptr70.20, align 4
  %add.ptr66.21 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1045
  %i97 = load float, float addrspace(4)* %add.ptr66.21, align 4
  %add.ptr66.1.21 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2069
  %i98 = load float, float addrspace(4)* %add.ptr66.1.21, align 4
  %add.ptr66.2.21 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3093
  %i99 = load float, float addrspace(4)* %add.ptr66.2.21, align 4
  %add.ptr70.21 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 22
  %i100 = load float, float addrspace(4)* %add.ptr70.21, align 4
  %add.ptr66.22 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1046
  %i101 = load float, float addrspace(4)* %add.ptr66.22, align 4
  %add.ptr66.1.22 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2070
  %i102 = load float, float addrspace(4)* %add.ptr66.1.22, align 4
  %add.ptr66.2.22 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3094
  %i103 = load float, float addrspace(4)* %add.ptr66.2.22, align 4
  %add.ptr70.22 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 23
  %i104 = load float, float addrspace(4)* %add.ptr70.22, align 4
  %add.ptr66.23 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1047
  %i105 = load float, float addrspace(4)* %add.ptr66.23, align 4
  %add.ptr66.1.23 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2071
  %i106 = load float, float addrspace(4)* %add.ptr66.1.23, align 4
  %add.ptr66.2.23 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3095
  %i107 = load float, float addrspace(4)* %add.ptr66.2.23, align 4
  %add.ptr70.23 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 24
  %i108 = load float, float addrspace(4)* %add.ptr70.23, align 4
  %add.ptr66.24 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1048
  %i109 = load float, float addrspace(4)* %add.ptr66.24, align 4
  %add.ptr66.1.24 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2072
  %i110 = load float, float addrspace(4)* %add.ptr66.1.24, align 4
  %add.ptr66.2.24 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3096
  %i111 = load float, float addrspace(4)* %add.ptr66.2.24, align 4
  %add.ptr70.24 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 25
  %i112 = load float, float addrspace(4)* %add.ptr70.24, align 4
  %add.ptr66.25 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1049
  %i113 = load float, float addrspace(4)* %add.ptr66.25, align 4
  %add.ptr66.1.25 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2073
  %i114 = load float, float addrspace(4)* %add.ptr66.1.25, align 4
  %add.ptr66.2.25 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3097
  %i115 = load float, float addrspace(4)* %add.ptr66.2.25, align 4
  %add.ptr70.25 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 26
  %i116 = load float, float addrspace(4)* %add.ptr70.25, align 4
  %add.ptr66.26 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1050
  %i117 = load float, float addrspace(4)* %add.ptr66.26, align 4
  %add.ptr66.1.26 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2074
  %i118 = load float, float addrspace(4)* %add.ptr66.1.26, align 4
  %add.ptr66.2.26 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3098
  %i119 = load float, float addrspace(4)* %add.ptr66.2.26, align 4
  %add.ptr70.26 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 27
  %i120 = load float, float addrspace(4)* %add.ptr70.26, align 4
  %add.ptr66.27 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1051
  %i121 = load float, float addrspace(4)* %add.ptr66.27, align 4
  %add.ptr66.1.27 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2075
  %i122 = load float, float addrspace(4)* %add.ptr66.1.27, align 4
  %add.ptr66.2.27 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3099
  %i123 = load float, float addrspace(4)* %add.ptr66.2.27, align 4
  %add.ptr70.27 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 28
  %i124 = load float, float addrspace(4)* %add.ptr70.27, align 4
  %add.ptr66.28 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1052
  %i125 = load float, float addrspace(4)* %add.ptr66.28, align 4
  %add.ptr66.1.28 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2076
  %i126 = load float, float addrspace(4)* %add.ptr66.1.28, align 4
  %add.ptr66.2.28 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3100
  %i127 = load float, float addrspace(4)* %add.ptr66.2.28, align 4
  %add.ptr70.28 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 29
  %i128 = load float, float addrspace(4)* %add.ptr70.28, align 4
  %add.ptr66.29 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1053
  %i129 = load float, float addrspace(4)* %add.ptr66.29, align 4
  %add.ptr66.1.29 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2077
  %i130 = load float, float addrspace(4)* %add.ptr66.1.29, align 4
  %add.ptr66.2.29 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3101
  %i131 = load float, float addrspace(4)* %add.ptr66.2.29, align 4
  %add.ptr70.29 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 30
  %i132 = load float, float addrspace(4)* %add.ptr70.29, align 4
  %add.ptr66.30 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1054
  %i133 = load float, float addrspace(4)* %add.ptr66.30, align 4
  %add.ptr66.1.30 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2078
  %i134 = load float, float addrspace(4)* %add.ptr66.1.30, align 4
  %add.ptr66.2.30 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3102
  %i135 = load float, float addrspace(4)* %add.ptr66.2.30, align 4
  %add.ptr70.30 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 31
  %i136 = load float, float addrspace(4)* %add.ptr70.30, align 4
  %add.ptr66.31 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 1055
  %i137 = load float, float addrspace(4)* %add.ptr66.31, align 4
  %add.ptr66.1.31 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 2079
  %i138 = load float, float addrspace(4)* %add.ptr66.1.31, align 4
  %add.ptr66.2.31 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 3103
  %i139 = load float, float addrspace(4)* %add.ptr66.2.31, align 4
  %add.ptr47.3 = getelementptr inbounds float, float addrspace(1)* %i_ptr.0288, i64 196
  %i140 = tail call float @llvm.fmuladd.f32(float %i8, float %i12, float %accum.sroa.0.0)
  %i141 = tail call float @llvm.fmuladd.f32(float %i9, float %i13, float %i140)
  %i142 = tail call float @llvm.fmuladd.f32(float %i10, float %i14, float %i141)
  %i143 = tail call float @llvm.fmuladd.f32(float %i11, float %i15, float %i142)
  %i144 = tail call float @llvm.fmuladd.f32(float %i8, float %i16, float %accum.sroa.6.0)
  %i145 = tail call float @llvm.fmuladd.f32(float %i9, float %i17, float %i144)
  %i146 = tail call float @llvm.fmuladd.f32(float %i10, float %i18, float %i145)
  %i147 = tail call float @llvm.fmuladd.f32(float %i11, float %i19, float %i146)
  %i148 = tail call float @llvm.fmuladd.f32(float %i8, float %i20, float %accum.sroa.10.0)
  %i149 = tail call float @llvm.fmuladd.f32(float %i9, float %i21, float %i148)
  %i150 = tail call float @llvm.fmuladd.f32(float %i10, float %i22, float %i149)
  %i151 = tail call float @llvm.fmuladd.f32(float %i11, float %i23, float %i150)
  %i152 = tail call float @llvm.fmuladd.f32(float %i8, float %i24, float %accum.sroa.14.0)
  %i153 = tail call float @llvm.fmuladd.f32(float %i9, float %i25, float %i152)
  %i154 = tail call float @llvm.fmuladd.f32(float %i10, float %i26, float %i153)
  %i155 = tail call float @llvm.fmuladd.f32(float %i11, float %i27, float %i154)
  %i156 = tail call float @llvm.fmuladd.f32(float %i8, float %i28, float %accum.sroa.18.0)
  %i157 = tail call float @llvm.fmuladd.f32(float %i9, float %i29, float %i156)
  %i158 = tail call float @llvm.fmuladd.f32(float %i10, float %i30, float %i157)
  %i159 = tail call float @llvm.fmuladd.f32(float %i11, float %i31, float %i158)
  %i160 = tail call float @llvm.fmuladd.f32(float %i8, float %i32, float %accum.sroa.22.0)
  %i161 = tail call float @llvm.fmuladd.f32(float %i9, float %i33, float %i160)
  %i162 = tail call float @llvm.fmuladd.f32(float %i10, float %i34, float %i161)
  %i163 = tail call float @llvm.fmuladd.f32(float %i11, float %i35, float %i162)
  %i164 = tail call float @llvm.fmuladd.f32(float %i8, float %i36, float %accum.sroa.26.0)
  %i165 = tail call float @llvm.fmuladd.f32(float %i9, float %i37, float %i164)
  %i166 = tail call float @llvm.fmuladd.f32(float %i10, float %i38, float %i165)
  %i167 = tail call float @llvm.fmuladd.f32(float %i11, float %i39, float %i166)
  %i168 = tail call float @llvm.fmuladd.f32(float %i8, float %i40, float %accum.sroa.30.0)
  %i169 = tail call float @llvm.fmuladd.f32(float %i9, float %i41, float %i168)
  %i170 = tail call float @llvm.fmuladd.f32(float %i10, float %i42, float %i169)
  %i171 = tail call float @llvm.fmuladd.f32(float %i11, float %i43, float %i170)
  %i172 = tail call float @llvm.fmuladd.f32(float %i8, float %i44, float %accum.sroa.34.0)
  %i173 = tail call float @llvm.fmuladd.f32(float %i9, float %i45, float %i172)
  %i174 = tail call float @llvm.fmuladd.f32(float %i10, float %i46, float %i173)
  %i175 = tail call float @llvm.fmuladd.f32(float %i11, float %i47, float %i174)
  %i176 = tail call float @llvm.fmuladd.f32(float %i8, float %i48, float %accum.sroa.38.0)
  %i177 = tail call float @llvm.fmuladd.f32(float %i9, float %i49, float %i176)
  %i178 = tail call float @llvm.fmuladd.f32(float %i10, float %i50, float %i177)
  %i179 = tail call float @llvm.fmuladd.f32(float %i11, float %i51, float %i178)
  %i180 = tail call float @llvm.fmuladd.f32(float %i8, float %i52, float %accum.sroa.42.0)
  %i181 = tail call float @llvm.fmuladd.f32(float %i9, float %i53, float %i180)
  %i182 = tail call float @llvm.fmuladd.f32(float %i10, float %i54, float %i181)
  %i183 = tail call float @llvm.fmuladd.f32(float %i11, float %i55, float %i182)
  %i184 = tail call float @llvm.fmuladd.f32(float %i8, float %i56, float %accum.sroa.46.0)
  %i185 = tail call float @llvm.fmuladd.f32(float %i9, float %i57, float %i184)
  %i186 = tail call float @llvm.fmuladd.f32(float %i10, float %i58, float %i185)
  %i187 = tail call float @llvm.fmuladd.f32(float %i11, float %i59, float %i186)
  %i188 = tail call float @llvm.fmuladd.f32(float %i8, float %i60, float %accum.sroa.50.0)
  %i189 = tail call float @llvm.fmuladd.f32(float %i9, float %i61, float %i188)
  %i190 = tail call float @llvm.fmuladd.f32(float %i10, float %i62, float %i189)
  %i191 = tail call float @llvm.fmuladd.f32(float %i11, float %i63, float %i190)
  %i192 = tail call float @llvm.fmuladd.f32(float %i8, float %i64, float %accum.sroa.54.0)
  %i193 = tail call float @llvm.fmuladd.f32(float %i9, float %i65, float %i192)
  %i194 = tail call float @llvm.fmuladd.f32(float %i10, float %i66, float %i193)
  %i195 = tail call float @llvm.fmuladd.f32(float %i11, float %i67, float %i194)
  %i196 = tail call float @llvm.fmuladd.f32(float %i8, float %i68, float %accum.sroa.58.0)
  %i197 = tail call float @llvm.fmuladd.f32(float %i9, float %i69, float %i196)
  %i198 = tail call float @llvm.fmuladd.f32(float %i10, float %i70, float %i197)
  %i199 = tail call float @llvm.fmuladd.f32(float %i11, float %i71, float %i198)
  %i200 = tail call float @llvm.fmuladd.f32(float %i8, float %i72, float %accum.sroa.62.0)
  %i201 = tail call float @llvm.fmuladd.f32(float %i9, float %i73, float %i200)
  %i202 = tail call float @llvm.fmuladd.f32(float %i10, float %i74, float %i201)
  %i203 = tail call float @llvm.fmuladd.f32(float %i11, float %i75, float %i202)
  %i204 = tail call float @llvm.fmuladd.f32(float %i8, float %i76, float %accum.sroa.66.0)
  %i205 = tail call float @llvm.fmuladd.f32(float %i9, float %i77, float %i204)
  %i206 = tail call float @llvm.fmuladd.f32(float %i10, float %i78, float %i205)
  %i207 = tail call float @llvm.fmuladd.f32(float %i11, float %i79, float %i206)
  %i208 = tail call float @llvm.fmuladd.f32(float %i8, float %i80, float %accum.sroa.70.0)
  %i209 = tail call float @llvm.fmuladd.f32(float %i9, float %i81, float %i208)
  %i210 = tail call float @llvm.fmuladd.f32(float %i10, float %i82, float %i209)
  %i211 = tail call float @llvm.fmuladd.f32(float %i11, float %i83, float %i210)
  %i212 = tail call float @llvm.fmuladd.f32(float %i8, float %i84, float %accum.sroa.74.0)
  %i213 = tail call float @llvm.fmuladd.f32(float %i9, float %i85, float %i212)
  %i214 = tail call float @llvm.fmuladd.f32(float %i10, float %i86, float %i213)
  %i215 = tail call float @llvm.fmuladd.f32(float %i11, float %i87, float %i214)
  %i216 = tail call float @llvm.fmuladd.f32(float %i8, float %i88, float %accum.sroa.78.0)
  %i217 = tail call float @llvm.fmuladd.f32(float %i9, float %i89, float %i216)
  %i218 = tail call float @llvm.fmuladd.f32(float %i10, float %i90, float %i217)
  %i219 = tail call float @llvm.fmuladd.f32(float %i11, float %i91, float %i218)
  %i220 = tail call float @llvm.fmuladd.f32(float %i8, float %i92, float %accum.sroa.82.0)
  %i221 = tail call float @llvm.fmuladd.f32(float %i9, float %i93, float %i220)
  %i222 = tail call float @llvm.fmuladd.f32(float %i10, float %i94, float %i221)
  %i223 = tail call float @llvm.fmuladd.f32(float %i11, float %i95, float %i222)
  %i224 = tail call float @llvm.fmuladd.f32(float %i8, float %i96, float %accum.sroa.86.0)
  %i225 = tail call float @llvm.fmuladd.f32(float %i9, float %i97, float %i224)
  %i226 = tail call float @llvm.fmuladd.f32(float %i10, float %i98, float %i225)
  %i227 = tail call float @llvm.fmuladd.f32(float %i11, float %i99, float %i226)
  %i228 = tail call float @llvm.fmuladd.f32(float %i8, float %i100, float %accum.sroa.90.0)
  %i229 = tail call float @llvm.fmuladd.f32(float %i9, float %i101, float %i228)
  %i230 = tail call float @llvm.fmuladd.f32(float %i10, float %i102, float %i229)
  %i231 = tail call float @llvm.fmuladd.f32(float %i11, float %i103, float %i230)
  %i232 = tail call float @llvm.fmuladd.f32(float %i8, float %i104, float %accum.sroa.94.0)
  %i233 = tail call float @llvm.fmuladd.f32(float %i9, float %i105, float %i232)
  %i234 = tail call float @llvm.fmuladd.f32(float %i10, float %i106, float %i233)
  %i235 = tail call float @llvm.fmuladd.f32(float %i11, float %i107, float %i234)
  %i236 = tail call float @llvm.fmuladd.f32(float %i8, float %i108, float %accum.sroa.98.0)
  %i237 = tail call float @llvm.fmuladd.f32(float %i9, float %i109, float %i236)
  %i238 = tail call float @llvm.fmuladd.f32(float %i10, float %i110, float %i237)
  %i239 = tail call float @llvm.fmuladd.f32(float %i11, float %i111, float %i238)
  %i240 = tail call float @llvm.fmuladd.f32(float %i8, float %i112, float %accum.sroa.102.0)
  %i241 = tail call float @llvm.fmuladd.f32(float %i9, float %i113, float %i240)
  %i242 = tail call float @llvm.fmuladd.f32(float %i10, float %i114, float %i241)
  %i243 = tail call float @llvm.fmuladd.f32(float %i11, float %i115, float %i242)
  %i244 = tail call float @llvm.fmuladd.f32(float %i8, float %i116, float %accum.sroa.106.0)
  %i245 = tail call float @llvm.fmuladd.f32(float %i9, float %i117, float %i244)
  %i246 = tail call float @llvm.fmuladd.f32(float %i10, float %i118, float %i245)
  %i247 = tail call float @llvm.fmuladd.f32(float %i11, float %i119, float %i246)
  %i248 = tail call float @llvm.fmuladd.f32(float %i8, float %i120, float %accum.sroa.110.0)
  %i249 = tail call float @llvm.fmuladd.f32(float %i9, float %i121, float %i248)
  %i250 = tail call float @llvm.fmuladd.f32(float %i10, float %i122, float %i249)
  %i251 = tail call float @llvm.fmuladd.f32(float %i11, float %i123, float %i250)
  %i252 = tail call float @llvm.fmuladd.f32(float %i8, float %i124, float %accum.sroa.114.0)
  %i253 = tail call float @llvm.fmuladd.f32(float %i9, float %i125, float %i252)
  %i254 = tail call float @llvm.fmuladd.f32(float %i10, float %i126, float %i253)
  %i255 = tail call float @llvm.fmuladd.f32(float %i11, float %i127, float %i254)
  %i256 = tail call float @llvm.fmuladd.f32(float %i8, float %i128, float %accum.sroa.118.0)
  %i257 = tail call float @llvm.fmuladd.f32(float %i9, float %i129, float %i256)
  %i258 = tail call float @llvm.fmuladd.f32(float %i10, float %i130, float %i257)
  %i259 = tail call float @llvm.fmuladd.f32(float %i11, float %i131, float %i258)
  %i260 = tail call float @llvm.fmuladd.f32(float %i8, float %i132, float %accum.sroa.122.0)
  %i261 = tail call float @llvm.fmuladd.f32(float %i9, float %i133, float %i260)
  %i262 = tail call float @llvm.fmuladd.f32(float %i10, float %i134, float %i261)
  %i263 = tail call float @llvm.fmuladd.f32(float %i11, float %i135, float %i262)
  %i264 = tail call float @llvm.fmuladd.f32(float %i8, float %i136, float %accum.sroa.126.0)
  %i265 = tail call float @llvm.fmuladd.f32(float %i9, float %i137, float %i264)
  %i266 = tail call float @llvm.fmuladd.f32(float %i10, float %i138, float %i265)
  %i267 = tail call float @llvm.fmuladd.f32(float %i11, float %i139, float %i266)
  %add.ptr74 = getelementptr inbounds float, float addrspace(4)* %w_ptr.0287, i64 4096
  %inc116 = add nuw nsw i32 %ci.0286, 1
  %exitcond.not = icmp eq i32 %inc116, 512
  br i1 %exitcond.not, label %for.cond.cleanup26, label %for.cond28.preheader

for.cond.cleanup26:                               ; preds = %for.cond28.preheader
  %mul119 = shl nuw nsw i32 undef, 1
  %mul120 = mul i32 %div, 200704
  %mul121 = mul i32 undef, 6272
  %add122 = add i32 %mul120, %mul121
  %mul123 = mul nuw nsw i32 undef, 28
  %add124 = add i32 %add122, %mul123
  %add126 = add i32 %add124, %mul119
  %idx.ext127 = zext i32 %add126 to i64
  %add.ptr128 = getelementptr inbounds float, float addrspace(1)* %out_ptr, i64 %idx.ext127
  store float %i143, float addrspace(1)* %add.ptr128, align 4
  %add.ptr184 = getelementptr inbounds float, float addrspace(1)* %add.ptr128, i64 196
  store float %i147, float addrspace(1)* %add.ptr184, align 4
  %add.ptr167.1 = getelementptr inbounds float, float addrspace(1)* %add.ptr184, i64 14
  store float 0.000000e+00, float addrspace(1)* %add.ptr167.1, align 4
  %add.ptr175.1.1 = getelementptr inbounds float, float addrspace(1)* %add.ptr167.1, i64 1
  store float 0.000000e+00, float addrspace(1)* %add.ptr175.1.1, align 4
  %add.ptr184.1 = getelementptr inbounds float, float addrspace(1)* %add.ptr184, i64 196
  store float %i151, float addrspace(1)* %add.ptr184.1, align 4
  %add.ptr184.2 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.1, i64 196
  store float %i155, float addrspace(1)* %add.ptr184.2, align 4
  %add.ptr184.3 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.2, i64 196
  store float %i159, float addrspace(1)* %add.ptr184.3, align 4
  %add.ptr184.4 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.3, i64 196
  store float %i163, float addrspace(1)* %add.ptr184.4, align 4
  %add.ptr154.5 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.4, i64 1
  store float 0.000000e+00, float addrspace(1)* %add.ptr154.5, align 4
  %add.ptr184.5 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.4, i64 196
  store float %i167, float addrspace(1)* %add.ptr184.5, align 4
  %add.ptr154.6 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.5, i64 1
  store float 0.000000e+00, float addrspace(1)* %add.ptr154.6, align 4
  %add.ptr184.6 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.5, i64 196
  store float %i171, float addrspace(1)* %add.ptr184.6, align 4
  %add.ptr184.7 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.6, i64 196
  store float %i175, float addrspace(1)* %add.ptr184.7, align 4
  %add.ptr167.8 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.7, i64 14
  store float 0.000000e+00, float addrspace(1)* %add.ptr167.8, align 4
  %add.ptr175.1.8 = getelementptr inbounds float, float addrspace(1)* %add.ptr167.8, i64 1
  store float 0.000000e+00, float addrspace(1)* %add.ptr175.1.8, align 4
  %add.ptr184.8 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.7, i64 196
  store float %i179, float addrspace(1)* %add.ptr184.8, align 4
  %add.ptr184.9 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.8, i64 196
  store float %i183, float addrspace(1)* %add.ptr184.9, align 4
  %add.ptr184.10 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.9, i64 196
  store float %i187, float addrspace(1)* %add.ptr184.10, align 4
  %add.ptr184.11 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.10, i64 196
  store float %i191, float addrspace(1)* %add.ptr184.11, align 4
  %add.ptr184.12 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.11, i64 196
  store float %i195, float addrspace(1)* %add.ptr184.12, align 4
  %add.ptr184.13 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.12, i64 196
  store float %i199, float addrspace(1)* %add.ptr184.13, align 4
  %add.ptr184.14 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.13, i64 196
  store float %i203, float addrspace(1)* %add.ptr184.14, align 4
  %add.ptr184.15 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.14, i64 196
  store float %i207, float addrspace(1)* %add.ptr184.15, align 4
  %add.ptr184.16 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.15, i64 196
  store float %i211, float addrspace(1)* %add.ptr184.16, align 4
  %add.ptr184.17 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.16, i64 196
  store float %i215, float addrspace(1)* %add.ptr184.17, align 4
  %add.ptr184.18 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.17, i64 196
  store float %i219, float addrspace(1)* %add.ptr184.18, align 4
  %add.ptr184.19 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.18, i64 196
  store float %i223, float addrspace(1)* %add.ptr184.19, align 4
  %add.ptr184.20 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.19, i64 196
  store float %i227, float addrspace(1)* %add.ptr184.20, align 4
  %add.ptr184.21 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.20, i64 196
  store float %i231, float addrspace(1)* %add.ptr184.21, align 4
  %add.ptr184.22 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.21, i64 196
  store float %i235, float addrspace(1)* %add.ptr184.22, align 4
  %add.ptr184.23 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.22, i64 196
  store float %i239, float addrspace(1)* %add.ptr184.23, align 4
  %add.ptr184.24 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.23, i64 196
  store float %i243, float addrspace(1)* %add.ptr184.24, align 4
  %add.ptr184.25 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.24, i64 196
  store float %i247, float addrspace(1)* %add.ptr184.25, align 4
  %add.ptr184.26 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.25, i64 196
  store float %i251, float addrspace(1)* %add.ptr184.26, align 4
  %add.ptr184.27 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.26, i64 196
  store float %i255, float addrspace(1)* %add.ptr184.27, align 4
  %add.ptr184.28 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.27, i64 196
  store float %i259, float addrspace(1)* %add.ptr184.28, align 4
  %add.ptr184.29 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.28, i64 196
  store float %i263, float addrspace(1)* %add.ptr184.29, align 4
  %add.ptr184.30 = getelementptr inbounds float, float addrspace(1)* %add.ptr184.29, i64 196
  store float %i267, float addrspace(1)* %add.ptr184.30, align 4
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) #0
declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare align 4 i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #1

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { nounwind readnone speculatable willreturn }

!0 = !{i32 1, i32 2, i32 1, i32 0}
!1 = !{!"none", !"none", !"none", !"none"}
!2 = !{!"float*", !"float*", !"float*", !"float"}
!3 = !{!"restrict const", !"restrict const", !"restrict", !""}
!4 = !{i32 256, i32 1, i32 1}
!5 = !{i32 0, i32 1024}
