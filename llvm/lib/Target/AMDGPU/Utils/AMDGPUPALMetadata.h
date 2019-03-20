//===-- AMDGPUPALMetadata.h - PAL metadata handling -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// PAL metadata handling
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUPALMETADATA_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUPALMETADATA_H

#include "llvm/ADT/StringRef.h"
#include <map>

namespace llvm {

class AMDGPUTargetStreamer;
class formatted_raw_ostream;
class MCStreamer;
class Module;

class AMDGPUPALMetadata {
  std::map<uint32_t, uint32_t> Registers;

public:
  // Read the amdgpu.pal.metadata supplied by the frontend, ready for
  // per-function modification.
  void readFromIR(Module &M);

  // Set PAL metadata from a binary blob from the applicable .note record.
  // Returns false if bad format.  Blob must remain valid for the lifetime of
  // the Metadata.
  bool setFromBlob(unsigned Type, StringRef Blob);

  // Set the rsrc1 register in the metadata for a particular shader stage.
  // In fact this ORs the value into any previous setting of the register.
  void setRsrc1(unsigned CC, unsigned Val);

  // Set the rsrc2 register in the metadata for a particular shader stage.
  // In fact this ORs the value into any previous setting of the register.
  void setRsrc2(unsigned CC, unsigned Val);

  // Set the SPI_PS_INPUT_ENA register in the metadata.
  // In fact this ORs the value into any previous setting of the register.
  void setSpiPsInputEna(unsigned Val);

  // Set the SPI_PS_INPUT_ADDR register in the metadata.
  // In fact this ORs the value into any previous setting of the register.
  void setSpiPsInputAddr(unsigned Val);

  // Get a register from the metadata, or 0 if not currently set.
  unsigned getRegister(unsigned Reg);

  // Set a register in the metadata.
  // In fact this ORs the value into any previous setting of the register.
  void setRegister(unsigned Reg, unsigned Val);

  // Set the number of used vgprs in the metadata. This is an optional advisory
  // record for logging etc; wave dispatch actually uses the rsrc1 register for
  // the shader stage to determine the number of vgprs to allocate.
  void setNumUsedVgprs(unsigned CC, unsigned Val);

  // Set the number of used sgprs in the metadata. This is an optional advisory
  // record for logging etc; wave dispatch actually uses the rsrc1 register for
  // the shader stage to determine the number of sgprs to allocate.
  void setNumUsedSgprs(unsigned CC, unsigned Val);

  // Set the scratch size in the metadata.
  void setScratchSize(unsigned CC, unsigned Val);

  // Emit the accumulated PAL metadata as an asm directive.
  // This is called from AMDGPUTargetAsmStreamer::Finish().
  void toString(std::string &S);

  // Emit the accumulated PAL metadata as a binary blob.
  // This is called from AMDGPUTargetELFStreamer::Finish().
  void toBlob(unsigned Type, std::string &S);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUPALMETADATA_H
