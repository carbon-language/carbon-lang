// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck --check-prefix=CHECK %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck --check-prefix=CHECK %s

// CHECK:  .amd_amdgpu_hsa_metadata
// CHECK:    Version: [ 1, 0 ]
// CHECK:    Printf:
// CHECK:      - '1:1:4:%d\n'
// CHECK:      - '2:1:8:%g\n'
// CHECK:    Kernels:
// CHECK:      - Name:            test_kernel
// CHECK:        SymbolName:      'test_kernel@kd'
// CHECK:        Language:        OpenCL C
// CHECK:        LanguageVersion: [ 2, 0 ]
// CHECK:    Attrs:
// CHECK:        ReqdWorkGroupSize: [ 1, 2, 4 ]
// CHECK:        WorkGroupSizeHint: [ 8, 16, 32 ]
// CHECK:        VecTypeHint:       int
// CHECK: .end_amd_amdgpu_hsa_metadata
.amd_amdgpu_hsa_metadata
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      SymbolName:      test_kernel@kd
      Language:        OpenCL C
      LanguageVersion: [ 2, 0 ]
      Attrs:
        ReqdWorkGroupSize: [ 1, 2, 4 ]
        WorkGroupSizeHint: [ 8, 16, 32 ]
        VecTypeHint:       int
.end_amd_amdgpu_hsa_metadata
