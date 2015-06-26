// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | llvm-readobj -s -sd | FileCheck %s --check-prefix=ELF

// ELF: SHT_NOTE
// ELF: 0000: 04000000 08000000 01000000 414D4400
// ELF: 0010: 01000000 00000000 04000000 1B000000
// ELF: 0020: 03000000 414D4400 04000700 07000000
// ELF: 0030: 00000000 00000000 414D4400 414D4447
// ELF: 0040: 50550000

.hsa_code_object_version 1,0
// ASM: .hsa_code_object_version 1,0

.hsa_code_object_isa
// ASM: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"

