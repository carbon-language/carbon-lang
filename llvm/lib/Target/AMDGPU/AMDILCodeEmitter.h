//===-- AMDILCodeEmitter.h - AMDIL Code Emitter interface -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CodeEmitter interface for R600 and SI codegen.
//
//===----------------------------------------------------------------------===//

#ifndef AMDILCODEEMITTER_H
#define AMDILCODEEMITTER_H

namespace llvm {

  class AMDILCodeEmitter {
  public:
    uint64_t getBinaryCodeForInstr(const MachineInstr &MI) const;
    virtual uint64_t getMachineOpValue(const MachineInstr &MI,
                                   const MachineOperand &MO) const { return 0; }
    virtual unsigned GPR4AlignEncode(const MachineInstr  &MI,
                                     unsigned OpNo) const {
      return 0;
    }
    virtual unsigned GPR2AlignEncode(const MachineInstr &MI,
                                     unsigned OpNo) const {
      return 0;
    }
    virtual uint64_t VOPPostEncode(const MachineInstr &MI,
                                   uint64_t Value) const {
      return Value;
    }
    virtual uint64_t i32LiteralEncode(const MachineInstr &MI,
                                      unsigned OpNo) const {
      return 0;
    }
    virtual uint32_t SMRDmemriEncode(const MachineInstr &MI, unsigned OpNo)
                                                                     const {
      return 0;
    }
  };

} // End namespace llvm

#endif // AMDILCODEEMITTER_H
