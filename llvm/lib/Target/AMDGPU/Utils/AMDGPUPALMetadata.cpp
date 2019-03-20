//===-- AMDGPUPALMetadata.cpp - Accumulate and print AMDGPU PAL metadata  -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This class has methods called by AMDGPUAsmPrinter to accumulate and print
/// the PAL metadata.
//
//===----------------------------------------------------------------------===//
//

#include "AMDGPUPALMetadata.h"
#include "AMDGPU.h"
#include "AMDGPUAsmPrinter.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "SIDefines.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace llvm::AMDGPU;

// Read the amdgpu.pal.metadata supplied by the
// frontend into our Registers, ready for per-function modification.  It
// is a NamedMD containing an MDTuple containing a number of MDNodes each of
// which is an integer value, and each two integer values forms a key=value
// pair that we store as Registers[key]=value in the map.
void AMDGPUPALMetadata::readFromIR(Module &M) {
  auto NamedMD = M.getNamedMetadata("amdgpu.pal.metadata");
  if (!NamedMD || !NamedMD->getNumOperands())
    return;
  auto Tuple = dyn_cast<MDTuple>(NamedMD->getOperand(0));
  if (!Tuple)
    return;
  for (unsigned I = 0, E = Tuple->getNumOperands() & -2; I != E; I += 2) {
    auto Key = mdconst::dyn_extract<ConstantInt>(Tuple->getOperand(I));
    auto Val = mdconst::dyn_extract<ConstantInt>(Tuple->getOperand(I + 1));
    if (!Key || !Val)
      continue;
    Registers[Key->getZExtValue()] = Val->getZExtValue();
  }
}

// Set PAL metadata from a binary blob from the applicable .note record.
// Returns false if bad format.  Blob must remain valid for the lifetime of the
// Metadata.
bool AMDGPUPALMetadata::setFromBlob(unsigned Type, StringRef Blob) {
  assert(Type == ELF::NT_AMD_AMDGPU_PAL_METADATA);
  auto Data = reinterpret_cast<const uint32_t *>(Blob.data());
  for (unsigned I = 0; I != Blob.size() / sizeof(uint32_t) / 2; ++I)
    setRegister(Data[I * 2], Data[I * 2 + 1]);
  return true;
}

// Given the calling convention, calculate the register number for rsrc1. In
// principle the register number could change in future hardware, but we know
// it is the same for gfx6-9 (except that LS and ES don't exist on gfx9), so
// we can use fixed values.
static unsigned getRsrc1Reg(CallingConv::ID CC) {
  switch (CC) {
  default:
    return PALMD::R_2E12_COMPUTE_PGM_RSRC1;
  case CallingConv::AMDGPU_LS:
    return PALMD::R_2D4A_SPI_SHADER_PGM_RSRC1_LS;
  case CallingConv::AMDGPU_HS:
    return PALMD::R_2D0A_SPI_SHADER_PGM_RSRC1_HS;
  case CallingConv::AMDGPU_ES:
    return PALMD::R_2CCA_SPI_SHADER_PGM_RSRC1_ES;
  case CallingConv::AMDGPU_GS:
    return PALMD::R_2C8A_SPI_SHADER_PGM_RSRC1_GS;
  case CallingConv::AMDGPU_VS:
    return PALMD::R_2C4A_SPI_SHADER_PGM_RSRC1_VS;
  case CallingConv::AMDGPU_PS:
    return PALMD::R_2C0A_SPI_SHADER_PGM_RSRC1_PS;
  }
}

// Calculate the PAL metadata key for *S_SCRATCH_SIZE. It can be used
// with a constant offset to access any non-register shader-specific PAL
// metadata key.
static unsigned getScratchSizeKey(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::AMDGPU_PS:
    return PALMD::Key::PS_SCRATCH_SIZE;
  case CallingConv::AMDGPU_VS:
    return PALMD::Key::VS_SCRATCH_SIZE;
  case CallingConv::AMDGPU_GS:
    return PALMD::Key::GS_SCRATCH_SIZE;
  case CallingConv::AMDGPU_ES:
    return PALMD::Key::ES_SCRATCH_SIZE;
  case CallingConv::AMDGPU_HS:
    return PALMD::Key::HS_SCRATCH_SIZE;
  case CallingConv::AMDGPU_LS:
    return PALMD::Key::LS_SCRATCH_SIZE;
  default:
    return PALMD::Key::CS_SCRATCH_SIZE;
  }
}

// Set the rsrc1 register in the metadata for a particular shader stage.
// In fact this ORs the value into any previous setting of the register.
void AMDGPUPALMetadata::setRsrc1(CallingConv::ID CC, unsigned Val) {
  setRegister(getRsrc1Reg(CC), Val);
}

// Set the rsrc2 register in the metadata for a particular shader stage.
// In fact this ORs the value into any previous setting of the register.
void AMDGPUPALMetadata::setRsrc2(CallingConv::ID CC, unsigned Val) {
  setRegister(getRsrc1Reg(CC) + 1, Val);
}

// Set the SPI_PS_INPUT_ENA register in the metadata.
// In fact this ORs the value into any previous setting of the register.
void AMDGPUPALMetadata::setSpiPsInputEna(unsigned Val) {
  setRegister(PALMD::R_A1B3_SPI_PS_INPUT_ENA, Val);
}

// Set the SPI_PS_INPUT_ADDR register in the metadata.
// In fact this ORs the value into any previous setting of the register.
void AMDGPUPALMetadata::setSpiPsInputAddr(unsigned Val) {
  setRegister(PALMD::R_A1B4_SPI_PS_INPUT_ADDR, Val);
}

// Get a register from the metadata, or 0 if not currently set.
unsigned AMDGPUPALMetadata::getRegister(unsigned Reg) { return Registers[Reg]; }

// Set a register in the metadata.
// In fact this ORs the value into any previous setting of the register.
void AMDGPUPALMetadata::setRegister(unsigned Reg, unsigned Val) {
  Registers[Reg] |= Val;
}

// Set the number of used vgprs in the metadata. This is an optional advisory
// record for logging etc; wave dispatch actually uses the rsrc1 register for
// the shader stage to determine the number of vgprs to allocate.
void AMDGPUPALMetadata::setNumUsedVgprs(CallingConv::ID CC, unsigned Val) {
  unsigned NumUsedVgprsKey = getScratchSizeKey(CC) +
                             PALMD::Key::VS_NUM_USED_VGPRS -
                             PALMD::Key::VS_SCRATCH_SIZE;
  Registers[NumUsedVgprsKey] = Val;
}

// Set the number of used sgprs in the metadata. This is an optional advisory
// record for logging etc; wave dispatch actually uses the rsrc1 register for
// the shader stage to determine the number of sgprs to allocate.
void AMDGPUPALMetadata::setNumUsedSgprs(CallingConv::ID CC, unsigned Val) {
  unsigned NumUsedSgprsKey = getScratchSizeKey(CC) +
                             PALMD::Key::VS_NUM_USED_SGPRS -
                             PALMD::Key::VS_SCRATCH_SIZE;
  Registers[NumUsedSgprsKey] = Val;
}

// Set the scratch size in the metadata.
void AMDGPUPALMetadata::setScratchSize(CallingConv::ID CC, unsigned Val) {
  Registers[getScratchSizeKey(CC)] = Val;
}

// Convert the accumulated PAL metadata into an asm directive.
void AMDGPUPALMetadata::toString(std::string &String) {
  String.clear();
  if (Registers.empty())
    return;
  raw_string_ostream Stream(String);
  Stream << '\t' << AMDGPU::PALMD::AssemblerDirective << ' ';
  for (auto I = Registers.begin(), E = Registers.end(); I != E; ++I) {
    if (I != Registers.begin())
      Stream << ',';
    Stream << "0x" << Twine::utohexstr(I->first) << ",0x"
           << Twine::utohexstr(I->second);
  }
  Stream << '\n';
}

// Convert the accumulated PAL metadata into a binary blob for writing as
// a .note record of the specified AMD type.
void AMDGPUPALMetadata::toBlob(unsigned Type, std::string &Blob) {
  Blob.clear();
  if (Type != ELF::NT_AMD_AMDGPU_PAL_METADATA)
    return;
  if (Registers.empty())
    return;
  raw_string_ostream OS(Blob);
  support::endian::Writer EW(OS, support::endianness::little);
  for (auto I : Registers) {
    EW.write(uint32_t(I.first));
    EW.write(uint32_t(I.second));
  }
}

