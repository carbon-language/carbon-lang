// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=2 -show-encoding %s | FileCheck %s --check-prefix=ASM --check-prefix=ASM_700
// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=gfx803 --amdhsa-code-object-version=2 -show-encoding %s | FileCheck %s --check-prefix=ASM --check-prefix=ASM_803
// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=stoney --amdhsa-code-object-version=2 -show-encoding %s | FileCheck %s --check-prefix=ASM --check-prefix=ASM_810
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=2 -show-encoding %s | llvm-readobj -S --sd - | FileCheck %s --check-prefix=ELF --check-prefix=ELF_700
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=gfx803 --amdhsa-code-object-version=2 -show-encoding %s | llvm-readobj -S --sd - | FileCheck %s --check-prefix=ELF --check-prefix=ELF_803
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=stoney --amdhsa-code-object-version=2 -show-encoding %s | llvm-readobj -S --sd - | FileCheck %s --check-prefix=ELF --check-prefix=ELF_810

// ELF: SHT_NOTE
// ELF: 0000: 04000000 08000000 01000000 414D4400
// ELF: 0010: 01000000 00000000 04000000 1B000000
// ELF_700: 0020: 03000000 414D4400 04000700 07000000
// ELF_700: 0030: 00000000 00000000 414D4400 414D4447
// ELF_803: 0020: 03000000 414D4400 04000700 08000000
// ELF_803: 0030: 00000000 03000000 414D4400 414D4447
// ELF_810: 0020: 03000000 414D4400 04000700 08000000
// ELF_810: 0030: 01000000 00000000 414D4400 414D4447
// ELF: 0040: 50550000

.hsa_code_object_version 1,0
// ASM: .hsa_code_object_version 1,0

// Test defaults
.hsa_code_object_isa
// ASM_700: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
// ASM_803: .hsa_code_object_isa 8,0,3,"AMD","AMDGPU"
// ASM_810: .hsa_code_object_isa 8,1,0,"AMD","AMDGPU"

// Test expressions and symbols
.set A,2
.hsa_code_object_isa A+1,A*2,A/A+4,"AMD","AMDGPU"
// ASM: .hsa_code_object_isa 3,4,5,"AMD","AMDGPU"
