//===- Target/MRegisterInfo.h - Target Register Information -------*-C++-*-===//
//
// This file describes an abstract interface used to get information about a
// target machines register file.  This information is used for a variety of
// purposed, especially register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MREGISTERINFO_H
#define LLVM_TARGET_MREGISTERINFO_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include <assert.h>

class Type;
class MachineFunction;

/// MRegisterDesc - This record contains all of the information known about a
/// particular register.
///
struct MRegisterDesc {
  const char *Name;    // Assembly language name for the register
  unsigned   Flags;    // Flags identifying register properties (defined below)
  unsigned TSFlags;    // Target Specific Flags
};

/// MRF namespace - This namespace contains flags that pertain to machine
/// registers
///
namespace MRF {  // MRF = Machine Register Flags
  enum {
    INT8             =   1 << 0,   // This is an 8 bit integer register
    INT16            =   1 << 1,   // This is a 16 bit integer register
    INT32            =   1 << 2,   // This is a 32 bit integer register
    INT64            =   1 << 3,   // This is a 64 bit integer register
    INT128           =   1 << 4,   // This is a 128 bit integer register

    FP32             =   1 << 5,   // This is a 32 bit floating point register
    FP64             =   1 << 6,   // This is a 64 bit floating point register
    FP80             =   1 << 7,   // This is a 80 bit floating point register
    FP128            =   1 << 8,   // This is a 128 bit floating point register
  };
};

class TargetRegisterClass {
public:
  typedef const unsigned* iterator;
  typedef const unsigned* const_iterator;

private:
  const unsigned RegSize;               // Size of register in bytes
  const iterator RegsBegin, RegsEnd;
public:
  TargetRegisterClass(unsigned RS, iterator RB, iterator RE)
    : RegSize(RS), RegsBegin(RB), RegsEnd(RE) {}
  virtual ~TargetRegisterClass() {}     // Allow subclasses

  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsEnd; }

  unsigned getNumRegs() const { return RegsEnd-RegsBegin; }
  unsigned getRegister(unsigned i) const {
    assert(i < getNumRegs() && "Register number out of range!");
    return RegsBegin[i];
  }

  unsigned getDataSize() const { return RegSize; }

  //void getAliases(void);
};


/// MRegisterInfo base class - We assume that the target defines a static array
/// of MRegisterDesc objects that represent all of the machine registers that
/// the target has.  As such, we simply have to track a pointer to this array so
/// that we can turn register number into a register descriptor.
///
class MRegisterInfo {
  const MRegisterDesc *Desc;    // Pointer to the descriptor array
  unsigned NumRegs;             // Number of entries in the array
protected:
  MRegisterInfo(const MRegisterDesc *D, unsigned NR) : Desc(D), NumRegs(NR) {}
public:

  enum {                        // Define some target independant constants
    /// NoRegister - This 'hard' register is a 'noop' register for all backends.
    /// This is used as the destination register for instructions that do not
    /// produce a value.  Some frontends may use this as an operand register to
    /// mean special things, for example, the Sparc backend uses R0 to mean %g0
    /// which always PRODUCES the value 0.  The X86 backend does not use this
    /// value as an operand register.
    ///
    NoRegister = 0,

    /// FirstVirtualRegister - This is the first register number that is
    /// considered to be a 'virtual' register, which is part of the SSA
    /// namespace.  This must be the same for all targets, which means that each
    /// target is limited to 1024 registers.
    ///
    FirstVirtualRegister = 1024,
  };

  const MRegisterDesc &operator[](unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to access record for invalid register number!");
    return Desc[RegNo];
  }

  /// Provide a get method, equivalent to [], but more useful if we have a
  /// pointer to this object.
  ///
  const MRegisterDesc &get(unsigned RegNo) const { return operator[](RegNo); }


  virtual MachineBasicBlock::iterator
  storeReg2RegOffset(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator MBBI,
                     unsigned SrcReg, unsigned DestReg,
                     unsigned ImmOffset, unsigned dataSize) const = 0;

  virtual MachineBasicBlock::iterator
  loadRegOffset2Reg(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, unsigned SrcReg,
                    unsigned ImmOffset, unsigned dataSize) const = 0;

  virtual MachineBasicBlock::iterator
  moveReg2Reg(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator MBBI,
              unsigned DestReg, unsigned SrcReg, unsigned dataSize) const = 0;

  virtual MachineBasicBlock::iterator
  moveImm2Reg(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator MBBI,
              unsigned DestReg, unsigned Imm, unsigned dataSize) const = 0;

  virtual void
  emitPrologue(MachineFunction &MF, unsigned numBytes) const = 0;

  virtual void
  emitEpilogue(MachineBasicBlock &MBB, unsigned numBytes) const = 0;

  virtual const unsigned* getCalleeSaveRegs() const = 0;
  virtual const unsigned* getCallerSaveRegs() const = 0;

  virtual unsigned getFramePointer() const = 0;
  virtual unsigned getStackPointer() const = 0;

  /// Register class iterators
  typedef const TargetRegisterClass * const * const_iterator;

  virtual const_iterator regclass_begin() const = 0;
  virtual const_iterator regclass_end() const = 0;

  virtual unsigned getNumRegClasses() const = 0;
  virtual const TargetRegisterClass* getRegClassForType(const Type* Ty) const=0;
};

#endif
