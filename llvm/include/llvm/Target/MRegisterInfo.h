//===- Target/MRegisterInfo.h - Target Register Information -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes an abstract interface used to get information about a
// target machines register file.  This information is used for a variety of
// purposed, especially register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_MREGISTERINFO_H
#define LLVM_TARGET_MREGISTERINFO_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include <cassert>

class Type;
class MachineFunction;

/// MRegisterDesc - This record contains all of the information known about a
/// particular register.  The AliasSet field (if not null) contains a pointer to
/// a Zero terminated array of registers that this register aliases.  This is
/// needed for architectures like X86 which have AL alias AX alias EAX.
/// Registers that this does not apply to simply should set this to null.
///
struct MRegisterDesc {
  const char     *Name;       // Assembly language name for the register
  const unsigned *AliasSet;   // Register Alias Set, described above
  unsigned        Flags;      // Flags identifying register properties (below)
  unsigned        TSFlags;    // Target Specific Flags
};

class TargetRegisterClass {
public:
  typedef const unsigned* iterator;
  typedef const unsigned* const_iterator;

private:
  const unsigned RegSize, Alignment;    // Size & Alignment of register in bytes
  const iterator RegsBegin, RegsEnd;
public:
  TargetRegisterClass(unsigned RS, unsigned Al, iterator RB, iterator RE)
    : RegSize(RS), Alignment(Al), RegsBegin(RB), RegsEnd(RE) {}
  virtual ~TargetRegisterClass() {}     // Allow subclasses

  // begin/end - Return all of the registers in this class.
  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsEnd; }

  // getNumRegs - Return the number of registers in this class
  unsigned getNumRegs() const { return RegsEnd-RegsBegin; }

  // getRegister - Return the specified register in the class
  unsigned getRegister(unsigned i) const {
    assert(i < getNumRegs() && "Register number out of range!");
    return RegsBegin[i];
  }

  /// allocation_order_begin/end - These methods define a range of registers
  /// which specify the registers in this class that are valid to register
  /// allocate, and the preferred order to allocate them in.  For example,
  /// callee saved registers should be at the end of the list, because it is
  /// cheaper to allocate caller saved registers.
  ///
  /// These methods take a MachineFunction argument, which can be used to tune
  /// the allocatable registers based on the characteristics of the function.
  /// One simple example is that the frame pointer register can be used if
  /// frame-pointer-elimination is performed.
  ///
  /// By default, these methods return all registers in the class.
  ///
  virtual iterator allocation_order_begin(MachineFunction &MF) const {
    return begin();
  }
  virtual iterator allocation_order_end(MachineFunction &MF)   const {
    return end();
  }
  


  /// getSize - Return the size of the register in bytes, which is also the size
  /// of a stack slot allocated to hold a spilled copy of this register.
  unsigned getSize() const { return RegSize; }

  /// getAlignment - Return the minimum required alignment for a register of
  /// this class.
  unsigned getAlignment() const { return Alignment; }
};


/// MRegisterInfo base class - We assume that the target defines a static array
/// of MRegisterDesc objects that represent all of the machine registers that
/// the target has.  As such, we simply have to track a pointer to this array so
/// that we can turn register number into a register descriptor.
///
class MRegisterInfo {
public:
  typedef const TargetRegisterClass * const * regclass_iterator;
private:
  const MRegisterDesc *Desc;                  // Pointer to the descriptor array
  unsigned NumRegs;                           // Number of entries in the array

  regclass_iterator RegClassBegin, RegClassEnd;   // List of regclasses

  const TargetRegisterClass **PhysRegClasses; // Reg class for each register
  int CallFrameSetupOpcode, CallFrameDestroyOpcode;
protected:
  MRegisterInfo(const MRegisterDesc *D, unsigned NR,
                regclass_iterator RegClassBegin, regclass_iterator RegClassEnd,
		int CallFrameSetupOpcode = -1, int CallFrameDestroyOpcode = -1);
  virtual ~MRegisterInfo();
public:

  enum {                        // Define some target independent constants
    /// NoRegister - This 'hard' register is a 'noop' register for all backends.
    /// This is used as the destination register for instructions that do not
    /// produce a value.  Some frontends may use this as an operand register to
    /// mean special things, for example, the Sparc backend uses R0 to mean %g0
    /// which always PRODUCES the value 0.  The X86 backend does not use this
    /// value as an operand register, except for memory references.
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

  /// getRegClass - Return the register class for the specified physical
  /// register.
  ///
  const TargetRegisterClass *getRegClass(unsigned RegNo) const {
    assert(RegNo < NumRegs && "Register number out of range!");
    assert(PhysRegClasses[RegNo] && "Register is not in a class!");
    return PhysRegClasses[RegNo];
  }

  /// getAliasSet - Return the set of registers aliased by the specified
  /// register, or a null list of there are none.  The list returned is zero
  /// terminated.
  ///
  const unsigned *getAliasSet(unsigned RegNo) const {
    return get(RegNo).AliasSet;
  }

  /// getName - Return the symbolic target specific name for the specified
  /// physical register.
  const char *getName(unsigned RegNo) const {
    return get(RegNo).Name;
  }

  virtual const unsigned* getCalleeSaveRegs() const = 0;


  //===--------------------------------------------------------------------===//
  // Register Class Information
  //

  /// Register class iterators
  ///
  regclass_iterator regclass_begin() const { return RegClassBegin; }
  regclass_iterator regclass_end() const { return RegClassEnd; }

  unsigned getNumRegClasses() const {
    return regclass_end()-regclass_begin();
  }

  //===--------------------------------------------------------------------===//
  // All basic block modifier functions below return the number of
  // instructions added to (negative if removed from) the basic block
  // passed as their first argument.
  //
  // FIXME: This is only needed because we use a std::vector instead
  // of an ilist to keep MachineBasicBlock instructions. Inserting an
  // instruction to a MachineBasicBlock invalidates all iterators to
  // the basic block. The return value can be used to update an index
  // to the machine basic block instruction vector and circumvent the
  // iterator elimination problem but this is really not needed if we
  // move to a better representation.
  //

  //===--------------------------------------------------------------------===//
  // Interfaces used by the register allocator and stack frame
  // manipulation passes to move data around between registers,
  // immediates and memory.  The return value is the number of
  // instructions added to (negative if removed from) the basic block.
  //

  virtual int storeRegToStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator &MBBI,
                                  unsigned SrcReg, int FrameIndex,
                                  const TargetRegisterClass *RC) const = 0;

  virtual int loadRegFromStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator &MBBI,
                                   unsigned DestReg, int FrameIndex,
                                   const TargetRegisterClass *RC) const = 0;

  virtual int copyRegToReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI,
                           unsigned DestReg, unsigned SrcReg,
                           const TargetRegisterClass *RC) const = 0;


  /// getCallFrameSetup/DestroyOpcode - These methods return the opcode of the
  /// frame setup/destroy instructions if they exist (-1 otherwise).  Some
  /// targets use pseudo instructions in order to abstract away the difference
  /// between operating with a frame pointer and operating without, through the
  /// use of these two instructions.
  ///
  int getCallFrameSetupOpcode() const { return CallFrameSetupOpcode; }
  int getCallFrameDestroyOpcode() const { return CallFrameDestroyOpcode; }


  /// eliminateCallFramePseudoInstr - This method is called during prolog/epilog
  /// code insertion to eliminate call frame setup and destroy pseudo
  /// instructions (but only if the Target is using them).  It is responsible
  /// for eliminating these instructions, replacing them with concrete
  /// instructions.  This method need only be implemented if using call frame
  /// setup/destroy pseudo instructions. The return value is the number of
  /// instructions added to (negative if removed from) the basic block.
  ///
  virtual int eliminateCallFramePseudoInstr(MachineFunction &MF,
					     MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator &I) const {
    assert(getCallFrameSetupOpcode()== -1 && getCallFrameDestroyOpcode()== -1 &&
	   "eliminateCallFramePseudoInstr must be implemented if using"
	   " call frame setup/destroy pseudo instructions!");
    assert(0 && "Call Frame Pseudo Instructions do not exist on this target!");
  }

  /// processFunctionBeforeFrameFinalized - This method is called immediately
  /// before the specified functions frame layout (MF.getFrameInfo()) is
  /// finalized.  Once the frame is finalized, MO_FrameIndex operands are
  /// replaced with direct constants.  This method is optional. The return value
  /// is the number of instructions added to (negative if removed from) the
  /// basic block
  ///
  virtual int processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
    return 0;
  }

  /// eliminateFrameIndex - This method must be overriden to eliminate abstract
  /// frame indices from instructions which may use them.  The instruction
  /// referenced by the iterator contains an MO_FrameIndex operand which must be
  /// eliminated by this method.  This method may modify or replace the
  /// specified instruction, as long as it keeps the iterator pointing the the
  /// finished product. The return value is the number of instructions
  /// added to (negative if removed from) the basic block.
  ///
  virtual int eliminateFrameIndex(MachineFunction &MF,
                                  MachineBasicBlock::iterator &II) const = 0;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function. The return value is the number of instructions
  /// added to (negative if removed from) the basic block (entry for prologue).
  ///
  virtual int emitPrologue(MachineFunction &MF) const = 0;
  virtual int emitEpilogue(MachineFunction &MF,
                           MachineBasicBlock &MBB) const = 0;
};

#endif
