//===-- AMDGPUCodeEmitter.h - AMDGPU Code Emitter interface -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief CodeEmitter interface for R600 and SI codegen.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPUCODEEMITTER_H
#define AMDGPUCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCInst;
class MCOperand;

class AMDGPUMCCodeEmitter : public MCCodeEmitter {
public:

  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups) const;

  virtual uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups) const {
    return 0;
  }

  virtual unsigned GPR4AlignEncode(const MCInst  &MI, unsigned OpNo,
                                   SmallVectorImpl<MCFixup> &Fixups) const {
    return 0;
  }
  virtual unsigned GPR2AlignEncode(const MCInst &MI, unsigned OpNo,
                                   SmallVectorImpl<MCFixup> &Fixups) const {
    return 0;
  }
  virtual uint64_t VOPPostEncode(const MCInst &MI, uint64_t Value) const {
    return Value;
  }
  virtual uint64_t i32LiteralEncode(const MCInst &MI, unsigned OpNo,
                                   SmallVectorImpl<MCFixup> &Fixups) const {
    return 0;
  }
  virtual uint32_t SMRDmemriEncode(const MCInst &MI, unsigned OpNo,
                                   SmallVectorImpl<MCFixup> &Fixups) const {
    return 0;
  }
};

} // End namespace llvm

#endif // AMDGPUCODEEMITTER_H
