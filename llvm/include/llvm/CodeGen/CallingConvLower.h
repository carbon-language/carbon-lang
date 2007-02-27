//===-- llvm/CallingConvLower.h - Calling Conventions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CCState and CCValAssign classes, used for lowering
// and implementing calling conventions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CALLINGCONVLOWER_H
#define LLVM_CODEGEN_CALLINGCONVLOWER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {
  class MRegisterInfo;
  
/// CCState - This class holds information needed while lowering arguments and
/// return values.  It captures which registers are already assigned and which
/// stack slots are used.  It provides accessors to allocate these values.
class CCState {
  unsigned StackOffset;
  const MRegisterInfo &MRI;
  SmallVector<uint32_t, 16> UsedRegs;
public:
  CCState(const MRegisterInfo &mri);
  
  unsigned getNextStackOffset() const { return StackOffset; }

  /// isAllocated - Return true if the specified register (or an alias) is
  /// allocated.
  bool isAllocated(unsigned Reg) const {
    return UsedRegs[Reg/32] & (1 << (Reg&31));
  }

  /// getFirstUnallocated - Return the first unallocated register in the set, or
  /// NumRegs if they are all allocated.
  unsigned getFirstUnallocated(const unsigned *Regs, unsigned NumRegs) const {
    for (unsigned i = 0; i != NumRegs; ++i)
      if (!isAllocated(Regs[i]))
        return i;
    return NumRegs;
  }
  
  /// AllocateReg - Attempt to allocate one of the specified registers.  If none
  /// are available, return zero.  Otherwise, return the first one available,
  /// marking it and any aliases as allocated.
  unsigned AllocateReg(const unsigned *Regs, unsigned NumRegs) {
    unsigned FirstUnalloc = getFirstUnallocated(Regs, NumRegs);
    if (FirstUnalloc == NumRegs)
      return 0;    // Didn't find the reg.
     
    // Mark the register and any aliases as allocated.
    unsigned Reg = Regs[FirstUnalloc];
    MarkAllocated(Reg);
    return Reg;
  }
  
  /// AllocateStack - Allocate a chunk of stack space with the specified size
  /// and alignment.
  unsigned AllocateStack(unsigned Size, unsigned Align) {
    assert(Align && ((Align-1) & Align) == 0); // Align is power of 2.
    StackOffset = ((StackOffset + Align-1) & ~(Align-1));
    unsigned Result = StackOffset;
    StackOffset += Size;
    return Result;
  }
private:
  /// MarkAllocated - Mark a register and all of its aliases as allocated.
  void MarkAllocated(unsigned Reg);
};

/// CCValAssign - Represent assignment of one arg/retval to a location.
class CCValAssign {
public:
  enum LocInfo {
    Full,   // The value fills the full location.
    SExt,   // The value is sign extended in the location.
    ZExt,   // The value is zero extended in the location.
    AExt    // The value is extended with undefined upper bits.
    // TODO: a subset of the value is in the location.
  };
private:
  /// ValNo - This is the value number begin assigned (e.g. an argument number).
  unsigned ValNo;
  
  /// Loc is either a stack offset or a register number.
  unsigned Loc;
  
  /// isMem - True if this is a memory loc, false if it is a register loc.
  bool isMem : 1;
  
  /// Information about how the value is assigned.
  LocInfo HTP : 7;
  
  /// ValVT - The type of the value being assigned.
  MVT::ValueType ValVT : 8;

  /// LocVT - The type of the location being assigned to.
  MVT::ValueType LocVT : 8;
public:
    
  static CCValAssign getReg(unsigned ValNo, MVT::ValueType ValVT,
                            unsigned RegNo, MVT::ValueType LocVT,
                            LocInfo HTP) {
    CCValAssign Ret;
    Ret.ValNo = ValNo;
    Ret.Loc = RegNo;
    Ret.isMem = false;
    Ret.HTP = HTP;
    Ret.ValVT = ValVT;
    Ret.LocVT = LocVT;
    return Ret;
  }
  static CCValAssign getMem(unsigned ValNo, MVT::ValueType ValVT,
                            unsigned Offset, MVT::ValueType LocVT,
                            LocInfo HTP) {
    CCValAssign Ret;
    Ret.ValNo = ValNo;
    Ret.Loc = Offset;
    Ret.isMem = true;
    Ret.HTP = HTP;
    Ret.ValVT = ValVT;
    Ret.LocVT = LocVT;
    return Ret;
  }
  
  unsigned getValNo() const { return ValNo; }
  MVT::ValueType getValVT() const { return ValVT; }

  bool isRegLoc() const { return !isMem; }
  bool isMemLoc() const { return isMem; }
  
  unsigned getLocReg() const { assert(isRegLoc()); return Loc; }
  unsigned getLocMemOffset() const { assert(isMemLoc()); return Loc; }
  MVT::ValueType getLocVT() const { return LocVT; }
  
  LocInfo getLocInfo() const { return HTP; }
};

} // end namespace llvm

#endif
