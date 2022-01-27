// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck --check-prefix=CHECK %s

// CHECK:  .amd_amdgpu_hsa_metadata
// CHECK:    Version: [ 1, 0 ]
// CHECK:    Kernels:
// CHECK:      - Name:       test_kernel
// CHECK:        SymbolName: 'test_kernel@kd'
// CHECK:        DebugProps:
// CHECK:          DebuggerABIVersion:                [ 1, 0 ]
// CHECK:          ReservedNumVGPRs:                  4
// CHECK:          ReservedFirstVGPR:                 11
// CHECK:          PrivateSegmentBufferSGPR:          0
// CHECK:          WavefrontPrivateSegmentOffsetSGPR: 11
.amd_amdgpu_hsa_metadata
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      SymbolName:      test_kernel@kd
      DebugProps:
        DebuggerABIVersion:                [ 1, 0 ]
        ReservedNumVGPRs:                  4
        ReservedFirstVGPR:                 11
        PrivateSegmentBufferSGPR:          0
        WavefrontPrivateSegmentOffsetSGPR: 11
.end_amd_amdgpu_hsa_metadata
