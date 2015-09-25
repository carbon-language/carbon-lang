// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | llvm-readobj -s -sd | FileCheck %s --check-prefix=ELF

// For compatibility reasons we treat convert .text sections to .hsatext

// ELF: Section {

// We want to avoid emitting an empty .text section.
// ELF-NOT: Name: .text

// ELF: Name: .hsatext
// ELF: Type: SHT_PROGBITS (0x1)
// ELF: Flags [ (0xC00007)
// ELF: SHF_ALLOC (0x2)
// ELF: SHF_AMDGPU_HSA_AGENT (0x800000)
// ELF: SHF_AMDGPU_HSA_CODE (0x400000)
// ELF: SHF_EXECINSTR (0x4)
// ELF: SHF_WRITE (0x1)
// ELF: Size: 260
// ELF: }

.hsa_code_object_version 1,0
// ASM: .hsa_code_object_version 1,0

.hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
// ASM: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"

.text
// ASM: .hsatext

.amd_kernel_code_t
.end_amd_kernel_code_t

s_endpgm
