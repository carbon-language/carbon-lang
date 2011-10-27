//=== MC/MCRegisterInfo.h - Target Register Description ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes an abstract interface used to get information about a
// target machines register file.  This information is used for a variety of
// purposed, especially register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCREGISTERINFO_H
#define LLVM_MC_MCREGISTERINFO_H

#include "llvm/ADT/DenseMap.h"
#include <cassert>

namespace llvm {

/// MCRegisterClass - Base class of TargetRegisterClass.
class MCRegisterClass {
public:
  typedef const unsigned* iterator;
  typedef const unsigned* const_iterator;
private:
  unsigned ID;
  const char *Name;
  const unsigned RegSize, Alignment; // Size & Alignment of register in bytes
  const int CopyCost;
  const bool Allocatable;
  const iterator RegsBegin, RegsEnd;
  const unsigned char *const RegSet;
  const unsigned RegSetSize;
public:
  MCRegisterClass(unsigned id, const char *name,
                  unsigned RS, unsigned Al, int CC, bool Allocable,
                  iterator RB, iterator RE, const unsigned char *Bits,
                  unsigned NumBytes)
    : ID(id), Name(name), RegSize(RS), Alignment(Al), CopyCost(CC),
      Allocatable(Allocable), RegsBegin(RB), RegsEnd(RE), RegSet(Bits),
      RegSetSize(NumBytes) {
    for (iterator i = RegsBegin; i != RegsEnd; ++i)
       assert(contains(*i) && "Bit field corrupted.");
  }

  /// getID() - Return the register class ID number.
  ///
  unsigned getID() const { return ID; }

  /// getName() - Return the register class name for debugging.
  ///
  const char *getName() const { return Name; }

  /// begin/end - Return all of the registers in this class.
  ///
  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsEnd; }

  /// getNumRegs - Return the number of registers in this class.
  ///
  unsigned getNumRegs() const { return (unsigned)(RegsEnd-RegsBegin); }

  /// getRegister - Return the specified register in the class.
  ///
  unsigned getRegister(unsigned i) const {
    assert(i < getNumRegs() && "Register number out of range!");
    return RegsBegin[i];
  }

  /// contains - Return true if the specified register is included in this
  /// register class.  This does not include virtual registers.
  bool contains(unsigned Reg) const {
    unsigned InByte = Reg % 8;
    unsigned Byte = Reg / 8;
    if (Byte >= RegSetSize)
      return false;
    return (RegSet[Byte] & (1 << InByte)) != 0;
  }

  /// contains - Return true if both registers are in this class.
  bool contains(unsigned Reg1, unsigned Reg2) const {
    return contains(Reg1) && contains(Reg2);
  }

  /// getSize - Return the size of the register in bytes, which is also the size
  /// of a stack slot allocated to hold a spilled copy of this register.
  unsigned getSize() const { return RegSize; }

  /// getAlignment - Return the minimum required alignment for a register of
  /// this class.
  unsigned getAlignment() const { return Alignment; }

  /// getCopyCost - Return the cost of copying a value between two registers in
  /// this class. A negative number means the register class is very expensive
  /// to copy e.g. status flag register classes.
  int getCopyCost() const { return CopyCost; }

  /// isAllocatable - Return true if this register class may be used to create
  /// virtual registers.
  bool isAllocatable() const { return Allocatable; }
};

/// MCRegisterDesc - This record contains all of the information known about
/// a particular register.  The Overlaps field contains a pointer to a zero
/// terminated array of registers that this register aliases, starting with
/// itself. This is needed for architectures like X86 which have AL alias AX
/// alias EAX. The SubRegs field is a zero terminated array of registers that
/// are sub-registers of the specific register, e.g. AL, AH are sub-registers of
/// AX. The SuperRegs field is a zero terminated array of registers that are
/// super-registers of the specific register, e.g. RAX, EAX, are super-registers
/// of AX.
///
struct MCRegisterDesc {
  const char     *Name;         // Printable name for the reg (for debugging)
  const unsigned *Overlaps;     // Overlapping registers, described above
  const unsigned *SubRegs;      // Sub-register set, described above
  const unsigned *SuperRegs;    // Super-register set, described above
};

/// MCRegisterInfo base class - We assume that the target defines a static
/// array of MCRegisterDesc objects that represent all of the machine
/// registers that the target has.  As such, we simply have to track a pointer
/// to this array so that we can turn register number into a register
/// descriptor.
///
/// Note this class is designed to be a base class of TargetRegisterInfo, which
/// is the interface used by codegen. However, specific targets *should never*
/// specialize this class. MCRegisterInfo should only contain getters to access
/// TableGen generated physical register data. It must not be extended with
/// virtual methods.
///
class MCRegisterInfo {
public:
  typedef const MCRegisterClass *regclass_iterator;
private:
  const MCRegisterDesc *Desc;                 // Pointer to the descriptor array
  unsigned NumRegs;                           // Number of entries in the array
  unsigned RAReg;                             // Return address register
  const MCRegisterClass *Classes;             // Pointer to the regclass array
  unsigned NumClasses;                        // Number of entries in the array
  DenseMap<unsigned, int> L2DwarfRegs;        // LLVM to Dwarf regs mapping
  DenseMap<unsigned, int> EHL2DwarfRegs;      // LLVM to Dwarf regs mapping EH
  DenseMap<unsigned, unsigned> Dwarf2LRegs;   // Dwarf to LLVM regs mapping
  DenseMap<unsigned, unsigned> EHDwarf2LRegs; // Dwarf to LLVM regs mapping EH
  DenseMap<unsigned, int> L2SEHRegs;          // LLVM to SEH regs mapping

public:
  /// InitMCRegisterInfo - Initialize MCRegisterInfo, called by TableGen
  /// auto-generated routines. *DO NOT USE*.
  void InitMCRegisterInfo(const MCRegisterDesc *D, unsigned NR, unsigned RA,
                          const MCRegisterClass *C, unsigned NC) {
    Desc = D;
    NumRegs = NR;
    RAReg = RA;
    Classes = C;
    NumClasses = NC;
  }

  /// mapLLVMRegToDwarfReg - Used to initialize LLVM register to Dwarf
  /// register number mapping. Called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void mapLLVMRegToDwarfReg(unsigned LLVMReg, int DwarfReg, bool isEH) {
    if (isEH)
      EHL2DwarfRegs[LLVMReg] = DwarfReg;
    else
      L2DwarfRegs[LLVMReg] = DwarfReg;
  }

  /// mapDwarfRegToLLVMReg - Used to initialize Dwarf register to LLVM
  /// register number mapping. Called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void mapDwarfRegToLLVMReg(unsigned DwarfReg, unsigned LLVMReg, bool isEH) {
    if (isEH)
      EHDwarf2LRegs[DwarfReg] = LLVMReg;
    else
      Dwarf2LRegs[DwarfReg] = LLVMReg;
  }

  /// mapLLVMRegToSEHReg - Used to initialize LLVM register to SEH register
  /// number mapping. By default the SEH register number is just the same
  /// as the LLVM register number.
  /// FIXME: TableGen these numbers. Currently this requires target specific
  /// initialization code.
  void mapLLVMRegToSEHReg(unsigned LLVMReg, int SEHReg) {
    L2SEHRegs[LLVMReg] = SEHReg;
  }

  /// getRARegister - This method should return the register where the return
  /// address can be found.
  unsigned getRARegister() const {
    return RAReg;
  }

  const MCRegisterDesc &operator[](unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to access record for invalid register number!");
    return Desc[RegNo];
  }

  /// Provide a get method, equivalent to [], but more useful if we have a
  /// pointer to this object.
  ///
  const MCRegisterDesc &get(unsigned RegNo) const {
    return operator[](RegNo);
  }

  /// getAliasSet - Return the set of registers aliased by the specified
  /// register, or a null list of there are none.  The list returned is zero
  /// terminated.
  ///
  const unsigned *getAliasSet(unsigned RegNo) const {
    // The Overlaps set always begins with Reg itself.
    return get(RegNo).Overlaps + 1;
  }

  /// getOverlaps - Return a list of registers that overlap Reg, including
  /// itself. This is the same as the alias set except Reg is included in the
  /// list.
  /// These are exactly the registers in { x | regsOverlap(x, Reg) }.
  ///
  const unsigned *getOverlaps(unsigned RegNo) const {
    return get(RegNo).Overlaps;
  }

  /// getSubRegisters - Return the list of registers that are sub-registers of
  /// the specified register, or a null list of there are none. The list
  /// returned is zero terminated and sorted according to super-sub register
  /// relations. e.g. X86::RAX's sub-register list is EAX, AX, AL, AH.
  ///
  const unsigned *getSubRegisters(unsigned RegNo) const {
    return get(RegNo).SubRegs;
  }

  /// getSuperRegisters - Return the list of registers that are super-registers
  /// of the specified register, or a null list of there are none. The list
  /// returned is zero terminated and sorted according to super-sub register
  /// relations. e.g. X86::AL's super-register list is AX, EAX, RAX.
  ///
  const unsigned *getSuperRegisters(unsigned RegNo) const {
    return get(RegNo).SuperRegs;
  }

  /// getName - Return the human-readable symbolic target-specific name for the
  /// specified physical register.
  const char *getName(unsigned RegNo) const {
    return get(RegNo).Name;
  }

  /// getNumRegs - Return the number of registers this target has (useful for
  /// sizing arrays holding per register information)
  unsigned getNumRegs() const {
    return NumRegs;
  }

  /// getDwarfRegNum - Map a target register to an equivalent dwarf register
  /// number.  Returns -1 if there is no equivalent value.  The second
  /// parameter allows targets to use different numberings for EH info and
  /// debugging info.
  int getDwarfRegNum(unsigned RegNum, bool isEH) const {
    const DenseMap<unsigned, int> &M = isEH ? EHL2DwarfRegs : L2DwarfRegs;
    const DenseMap<unsigned, int>::const_iterator I = M.find(RegNum);
    if (I == M.end()) return -1;
    return I->second;
  }

  /// getLLVMRegNum - Map a dwarf register back to a target register.
  ///
  int getLLVMRegNum(unsigned RegNum, bool isEH) const {
    const DenseMap<unsigned, unsigned> &M = isEH ? EHDwarf2LRegs : Dwarf2LRegs;
    const DenseMap<unsigned, unsigned>::const_iterator I = M.find(RegNum);
    if (I == M.end()) {
      assert(0 && "Invalid RegNum");
      return -1;
    }
    return I->second;
  }

  /// getSEHRegNum - Map a target register to an equivalent SEH register
  /// number.  Returns LLVM register number if there is no equivalent value.
  int getSEHRegNum(unsigned RegNum) const {
    const DenseMap<unsigned, int>::const_iterator I = L2SEHRegs.find(RegNum);
    if (I == L2SEHRegs.end()) return (int)RegNum;
    return I->second;
  }

  regclass_iterator regclass_begin() const { return Classes; }
  regclass_iterator regclass_end() const { return Classes+NumClasses; }

  unsigned getNumRegClasses() const {
    return (unsigned)(regclass_end()-regclass_begin());
  }

  /// getRegClass - Returns the register class associated with the enumeration
  /// value.  See class MCOperandInfo.
  const MCRegisterClass getRegClass(unsigned i) const {
    assert(i < getNumRegClasses() && "Register Class ID out of range");
    return Classes[i];
  }
};

} // End llvm namespace

#endif
