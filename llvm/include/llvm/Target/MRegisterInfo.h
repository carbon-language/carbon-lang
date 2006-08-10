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
#include "llvm/CodeGen/ValueTypes.h"
#include <cassert>
#include <functional>

namespace llvm {

class Type;
class MachineFunction;
class MachineInstr;
class MachineLocation;
class MachineMove;
class TargetRegisterClass;

/// TargetRegisterDesc - This record contains all of the information known about
/// a particular register.  The AliasSet field (if not null) contains a pointer
/// to a Zero terminated array of registers that this register aliases.  This is
/// needed for architectures like X86 which have AL alias AX alias EAX.
/// Registers that this does not apply to simply should set this to null.
///
struct TargetRegisterDesc {
  const char     *Name;         // Assembly language name for the register
  const unsigned *AliasSet;     // Register Alias Set, described above
};

class TargetRegisterClass {
public:
  typedef const unsigned* iterator;
  typedef const unsigned* const_iterator;

  typedef const MVT::ValueType* vt_iterator;
  typedef const TargetRegisterClass* const * sc_iterator;
private:
  unsigned ID;
  bool  isSubClass;
  const vt_iterator VTs;
  const sc_iterator SubClasses;
  const sc_iterator SuperClasses;
  const unsigned RegSize, Alignment;    // Size & Alignment of register in bytes
  const iterator RegsBegin, RegsEnd;
public:
  TargetRegisterClass(unsigned id,
                      const MVT::ValueType *vts,
                      const TargetRegisterClass * const *subcs,
                      const TargetRegisterClass * const *supcs,
                      unsigned RS, unsigned Al, iterator RB, iterator RE)
    : ID(id), VTs(vts), SubClasses(subcs), SuperClasses(supcs),
    RegSize(RS), Alignment(Al), RegsBegin(RB), RegsEnd(RE) {}
  virtual ~TargetRegisterClass() {}     // Allow subclasses
  
  /// getID() - Return the register class ID number.
  ///
  unsigned getID() const { return ID; }
  
  /// begin/end - Return all of the registers in this class.
  ///
  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsEnd; }

  /// getNumRegs - Return the number of registers in this class.
  ///
  unsigned getNumRegs() const { return RegsEnd-RegsBegin; }

  /// getRegister - Return the specified register in the class.
  ///
  unsigned getRegister(unsigned i) const {
    assert(i < getNumRegs() && "Register number out of range!");
    return RegsBegin[i];
  }

  /// contains - Return true if the specified register is included in this
  /// register class.
  bool contains(unsigned Reg) const {
    for (iterator I = begin(), E = end(); I != E; ++I)
      if (*I == Reg) return true;
    return false;
  }

  /// hasType - return true if this TargetRegisterClass has the ValueType vt.
  ///
  bool hasType(MVT::ValueType vt) const {
    for(int i = 0; VTs[i] != MVT::Other; ++i)
      if (VTs[i] == vt)
        return true;
    return false;
  }
  
  /// vt_begin / vt_end - Loop over all of the value types that can be
  /// represented by values in this register class.
  vt_iterator vt_begin() const {
    return VTs;
  }

  vt_iterator vt_end() const {
    vt_iterator I = VTs;
    while (*I != MVT::Other) ++I;
    return I;
  }

  /// hasSubRegClass - return true if the specified TargetRegisterClass is a
  /// sub-register class of this TargetRegisterClass.
  bool hasSubRegClass(const TargetRegisterClass *cs) const {
    for (int i = 0; SubClasses[i] != NULL; ++i) 
      if (SubClasses[i] == cs)
        return true;
    return false;
  }

  /// subclasses_begin / subclasses_end - Loop over all of the sub-classes of
  /// this register class.
  sc_iterator subclasses_begin() const {
    return SubClasses;
  }
  
  sc_iterator subclasses_end() const {
    sc_iterator I = SubClasses;
    while (*I != NULL) ++I;
    return I;
  }
  
  /// hasSuperRegClass - return true if the specified TargetRegisterClass is a
  /// super-register class of this TargetRegisterClass.
  bool hasSuperRegClass(const TargetRegisterClass *cs) const {
    for (int i = 0; SuperClasses[i] != NULL; ++i) 
      if (SuperClasses[i] == cs)
        return true;
    return false;
  }

  /// superclasses_begin / superclasses_end - Loop over all of the super-classes
  /// of this register class.
  sc_iterator superclasses_begin() const {
    return SuperClasses;
  }
  
  sc_iterator superclasses_end() const {
    sc_iterator I = SuperClasses;
    while (*I != NULL) ++I;
    return I;
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
/// of TargetRegisterDesc objects that represent all of the machine registers
/// that the target has.  As such, we simply have to track a pointer to this
/// array so that we can turn register number into a register descriptor.
///
class MRegisterInfo {
public:
  typedef const TargetRegisterClass * const * regclass_iterator;
private:
  const TargetRegisterDesc *Desc;             // Pointer to the descriptor array
  unsigned NumRegs;                           // Number of entries in the array

  regclass_iterator RegClassBegin, RegClassEnd;   // List of regclasses

  int CallFrameSetupOpcode, CallFrameDestroyOpcode;
protected:
  MRegisterInfo(const TargetRegisterDesc *D, unsigned NR,
                regclass_iterator RegClassBegin, regclass_iterator RegClassEnd,
                int CallFrameSetupOpcode = -1, int CallFrameDestroyOpcode = -1);
  virtual ~MRegisterInfo();
public:

  enum {                        // Define some target independent constants
    /// NoRegister - This physical register is not a real target register.  It
    /// is useful as a sentinal.
    NoRegister = 0,

    /// FirstVirtualRegister - This is the first register number that is
    /// considered to be a 'virtual' register, which is part of the SSA
    /// namespace.  This must be the same for all targets, which means that each
    /// target is limited to 1024 registers.
    FirstVirtualRegister = 1024
  };

  /// isPhysicalRegister - Return true if the specified register number is in
  /// the physical register namespace.
  static bool isPhysicalRegister(unsigned Reg) {
    assert(Reg && "this is not a register!");
    return Reg < FirstVirtualRegister;
  }

  /// isVirtualRegister - Return true if the specified register number is in
  /// the virtual register namespace.
  static bool isVirtualRegister(unsigned Reg) {
    assert(Reg && "this is not a register!");
    return Reg >= FirstVirtualRegister;
  }

  /// getAllocatableSet - Returns a bitset indexed by register number
  /// indicating if a register is allocatable or not.
  std::vector<bool> getAllocatableSet(MachineFunction &MF) const;

  const TargetRegisterDesc &operator[](unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to access record for invalid register number!");
    return Desc[RegNo];
  }

  /// Provide a get method, equivalent to [], but more useful if we have a
  /// pointer to this object.
  ///
  const TargetRegisterDesc &get(unsigned RegNo) const {
    return operator[](RegNo);
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

  /// getNumRegs - Return the number of registers this target has
  /// (useful for sizing arrays holding per register information)
  unsigned getNumRegs() const {
    return NumRegs;
  }

  /// areAliases - Returns true if the two registers alias each other,
  /// false otherwise
  bool areAliases(unsigned regA, unsigned regB) const {
    for (const unsigned *Alias = getAliasSet(regA); *Alias; ++Alias)
      if (*Alias == regB) return true;
    return false;
  }

  /// getCalleeSaveRegs - Return a null-terminated list of all of the
  /// callee-save registers on this target.
  virtual const unsigned* getCalleeSaveRegs() const = 0;

  /// getCalleeSaveRegClasses - Return a null-terminated list of the preferred
  /// register classes to spill each callee-saved register with.  The order and
  /// length of this list match the getCalleeSaveRegs() list.
  virtual const TargetRegisterClass* const *getCalleeSaveRegClasses() const = 0;

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
  
  /// getRegClass - Returns the register class associated with the enumeration
  /// value.  See class TargetOperandInfo.
  const TargetRegisterClass *getRegClass(unsigned i) const {
    assert(i <= getNumRegClasses() && "Register Class ID out of range");
    return i ? RegClassBegin[i - 1] : NULL;
  }

  //===--------------------------------------------------------------------===//
  // Interfaces used by the register allocator and stack frame
  // manipulation passes to move data around between registers,
  // immediates and memory.  The return value is the number of
  // instructions added to (negative if removed from) the basic block.
  //

  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, int FrameIndex,
                                   const TargetRegisterClass *RC) const = 0;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const = 0;

  virtual void copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *RC) const = 0;

  /// foldMemoryOperand - Attempt to fold a load or store of the
  /// specified stack slot into the specified machine instruction for
  /// the specified operand.  If this is possible, a new instruction
  /// is returned with the specified operand folded, otherwise NULL is
  /// returned. The client is responsible for removing the old
  /// instruction and adding the new one in the instruction stream
  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                          unsigned OpNum,
                                          int FrameIndex) const {
    return 0;
  }

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
  /// setup/destroy pseudo instructions.
  ///
  virtual void
  eliminateCallFramePseudoInstr(MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const {
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
  virtual void processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
  }

  /// eliminateFrameIndex - This method must be overriden to eliminate abstract
  /// frame indices from instructions which may use them.  The instruction
  /// referenced by the iterator contains an MO_FrameIndex operand which must be
  /// eliminated by this method.  This method may modify or replace the
  /// specified instruction, as long as it keeps the iterator pointing the the
  /// finished product. The return value is the number of instructions
  /// added to (negative if removed from) the basic block.
  ///
  virtual void eliminateFrameIndex(MachineBasicBlock::iterator MI) const = 0;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function. The return value is the number of instructions
  /// added to (negative if removed from) the basic block (entry for prologue).
  ///
  virtual void emitPrologue(MachineFunction &MF) const = 0;
  virtual void emitEpilogue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const = 0;
                            
  //===--------------------------------------------------------------------===//
  /// Debug information queries.
  
  /// getDwarfRegNum - Map a target register to an equivalent dwarf register
  /// number.  Returns -1 if there is no equivalent value.
  virtual int getDwarfRegNum(unsigned RegNum) const = 0;

  /// getFrameRegister - This method should return the register used as a base
  /// for values allocated in the current stack frame.
  virtual unsigned getFrameRegister(MachineFunction &MF) const = 0;
  
  /// getRARegister - This method should return the register where the return
  /// address can be found.
  virtual unsigned getRARegister() const = 0;
                            
  /// getLocation - This method should return the actual location of a frame
  /// variable given the frame index.  The location is returned in ML.
  /// Subclasses should override this method for special handling of frame
  /// variables and call MRegisterInfo::getLocation for the default action.
  virtual void getLocation(MachineFunction &MF, unsigned Index,
                           MachineLocation &ML) const;
                           
  /// getInitialFrameState - Returns a list of machine moves that are assumed
  /// on entry to all functions.  Note that LabelID is ignored (assumed to be
  /// the beginning of the function.)
  virtual void getInitialFrameState(std::vector<MachineMove *> &Moves) const;
};

// This is useful when building DenseMaps keyed on virtual registers
struct VirtReg2IndexFunctor : std::unary_function<unsigned, unsigned> {
  unsigned operator()(unsigned Reg) const {
    return Reg - MRegisterInfo::FirstVirtualRegister;
  }
};

} // End llvm namespace

#endif
