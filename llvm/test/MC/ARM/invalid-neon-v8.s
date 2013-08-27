@ RUN: not llvm-mc -triple armv8 -mattr=+neon -show-encoding < %s 2>&1 | FileCheck %s

vmaxnm.f32 s4, d5, q1
@ CHECK: error: invalid operand for instruction
vmaxnm.f64.f64 s4, d5, q1
@ CHECK: error: invalid operand for instruction
vmaxnmge.f64.f64 s4, d5, q1
@ CHECK: error: instruction 'vmaxnm' is not predicable, but condition code specified

vcvta.s32.f32 s1, s2
@ CHECK: error: instruction requires: V8FP
vcvtp.u32.f32 s1, d2
@ CHECK: error: invalid operand for instruction
vcvtp.f32.u32 d1, q2
@ CHECK: error: invalid operand for instruction
vcvtplo.f32.u32 s1, s2
@ CHECK: error: instruction 'vcvtp' is not predicable, but condition code specified

vrinta.f64.f64 s3, d12
@ CHECK: error: invalid operand for instruction
vrintn.f32 d3, q12
@ CHECK: error: invalid operand for instruction
vrintz.f32 d3, q12
@ CHECK: error: invalid operand for instruction
vrintmge.f32.f32 d3, d4
@ CHECK: error: instruction 'vrintm' is not predicable, but condition code specified
