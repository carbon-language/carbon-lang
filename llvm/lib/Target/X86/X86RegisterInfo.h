//===- X86RegisterInfo.h - X86 Register Information Impl ----------*-C++-*-===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86REGISTERINFO_H
#define X86REGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"

class Type;

struct X86RegisterInfo : public MRegisterInfo {
  X86RegisterInfo();

  MRegisterInfo::const_iterator const_regclass_begin() const;
  MRegisterInfo::const_iterator const_regclass_end() const;

  void copyReg2PCRel(MachineBasicBlock *MBB,
                     MachineBasicBlock::iterator &MBBI,
                     unsigned SrcReg, unsigned ImmOffset,
                     unsigned dataSize) const;

  void copyPCRel2Reg(MachineBasicBlock *MBB,
                     MachineBasicBlock::iterator &MBBI,
                     unsigned ImmOffset, unsigned DestReg,
                     unsigned dataSize) const;

  /// Returns register class appropriate for input SSA register
  /// 
  const TargetRegisterClass *getClassForReg(unsigned Reg) const;

  const TargetRegisterClass* getRegClassForType(const Type* Ty) const;

  unsigned getNumRegClasses() const;


};

#endif
