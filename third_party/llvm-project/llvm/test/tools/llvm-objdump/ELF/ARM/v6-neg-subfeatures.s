@ RUN: llvm-mc < %s -triple armv6 -mattr=+vfp2 -filetype=obj | llvm-objdump -d - | FileCheck %s

.eabi_attribute Tag_FP_arch, 1 // VFP2

vfp2:
  vadd.f32 s0, s1, s2

@CHECK-LABEL: vfp2
@CHECK-NOT: 81 0a 30 ee vadd.f32 s0, s1, s2
@CHECK: unknown
