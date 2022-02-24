@ RUN: llvm-mc < %s -triple armv7m -mattr=+vfp4 -filetype=obj | llvm-objdump -d - | FileCheck %s

.eabi_attribute Tag_CPU_arch, 10 // v7
.eabi_attribute Tag_FP_arch, 0 // VFP4

.thumb
vfp2:
  vmla.f32 s0, s1, s2

@CHECK-LABEL: vfp2
@CHECK-NOT: 00 ee 81 0a vmla.f32 s0, s1, s2

.thumb
vfp4:
  vmov.f32 s0, #0.5

@CHECK-LABEL:vfp4
@CHECK-NOT: b6 ee 00 0a vmov.f32 s0, #5.000000e-01
