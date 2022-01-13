; RUN: llc -mtriple=amdgcn-amd-amdhsa -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s
; Make sure the expected regmask is generated for sub/superregisters.

; CHECK-DAG: csr Clobbered Registers: $vgpr0 $vgpr0_hi16 $vgpr0_lo16 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31 $vgpr0_vgpr1_vgpr2_vgpr3 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15 $vgpr0_vgpr1 $vgpr0_vgpr1_vgpr2 {{$}}
define void @csr() #0 {
  call void asm sideeffect "", "~{v0},~{v44},~{v45}"() #0
  ret void
}

; CHECK-DAG: subregs_for_super Clobbered Registers: $vgpr0 $vgpr1 $vgpr0_hi16 $vgpr1_hi16 $vgpr0_lo16 $vgpr1_lo16 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31_vgpr32 $vgpr0_vgpr1_vgpr2_vgpr3 $vgpr1_vgpr2_vgpr3_vgpr4 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16 $vgpr0_vgpr1 $vgpr1_vgpr2 $vgpr0_vgpr1_vgpr2 $vgpr1_vgpr2_vgpr3 {{$}}
define void @subregs_for_super() #0 {
  call void asm sideeffect "", "~{v0},~{v1}"() #0
  ret void
}

; CHECK-DAG: clobbered_reg_with_sub Clobbered Registers: $vgpr0 $vgpr1 $vgpr0_hi16 $vgpr1_hi16 $vgpr0_lo16 $vgpr1_lo16 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16_vgpr17_vgpr18_vgpr19_vgpr20_vgpr21_vgpr22_vgpr23_vgpr24_vgpr25_vgpr26_vgpr27_vgpr28_vgpr29_vgpr30_vgpr31_vgpr32 $vgpr0_vgpr1_vgpr2_vgpr3 $vgpr1_vgpr2_vgpr3_vgpr4 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8 $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15 $vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7_vgpr8_vgpr9_vgpr10_vgpr11_vgpr12_vgpr13_vgpr14_vgpr15_vgpr16 $vgpr0_vgpr1 $vgpr1_vgpr2 $vgpr0_vgpr1_vgpr2 $vgpr1_vgpr2_vgpr3 {{$}}
define void @clobbered_reg_with_sub() #0 {
  call void asm sideeffect "", "~{v[0:1]}"() #0
  ret void
}

; CHECK-DAG: nothing Clobbered Registers: {{$}}
define void @nothing() #0 {
  ret void
}

; CHECK-DAG: special_regs Clobbered Registers: $scc $m0 $m0_hi16 $m0_lo16 {{$}}
define void @special_regs() #0 {
  call void asm sideeffect "", "~{m0},~{scc}"() #0
  ret void
}

; CHECK-DAG: vcc Clobbered Registers: $vcc $vcc_hi $vcc_lo $vcc_hi_hi16 $vcc_hi_lo16 $vcc_lo_hi16 $vcc_lo_lo16 {{$}}
define void @vcc() #0 {
  call void asm sideeffect "", "~{vcc}"() #0
  ret void
}

@llvm.used = appending global [6 x i8*] [i8* bitcast (void ()* @csr to i8*),
                                         i8* bitcast (void ()* @subregs_for_super to i8*),
                                         i8* bitcast (void ()* @clobbered_reg_with_sub to i8*),
                                         i8* bitcast (void ()* @nothing to i8*),
                                         i8* bitcast (void ()* @special_regs to i8*),
                                         i8* bitcast (void ()* @vcc to i8*)]

attributes #0 = { nounwind }
