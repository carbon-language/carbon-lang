// RUN: %clang_cc1 -x cl %s -verify
// expected-no-diagnostics

// Test the forced language options for OpenCL are set correctly.

__constant int v0[(sizeof(int) == 4) -1];
__constant int v1[(__alignof(int) == 4) -1];
__constant int v2[(sizeof(long) == 8) -1];
__constant int v3[(__alignof(long) == 8) -1];
__constant int v4[(sizeof(long long) == 16) -1];
__constant int v5[(__alignof(long long) == 16) -1];
__constant int v6[(sizeof(float) == 4) -1];
__constant int v7[(__alignof(float) == 4) -1];
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__constant int v8[(sizeof(double)==8) -1];
__constant int v9[(__alignof(double)==8) -1];
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant int v10[(sizeof(half) == 2) -1];
__constant int v11[(__alignof(half) == 2) -1];
