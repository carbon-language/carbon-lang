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
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace llvm {

/// MCRegisterClass - Base class of TargetRegisterClass.
class MCRegisterClass {
public:
  typedef const uint16_t* iterator;
  typedef const uint16_t* const_iterator;

  const char *Name;
  const iterator RegsBegin;
  const uint8_t *const RegSet;
  const uint16_t RegsSize;
  const uint16_t RegSetSize;
  const uint16_t ID;
  const uint16_t RegSize, Alignment; // Size & Alignment of register in bytes
  const int8_t CopyCost;
  const bool Allocatable;

  /// getID() - Return the register class ID number.
  ///
  unsigned getID() const { return ID; }

  /// getName() - Return the register class name for debugging.
  ///
  const char *getName() const { return Name; }

  /// begin/end - Return all of the registers in this class.
  ///
  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsBegin + RegsSize; }

  /// getNumRegs - Return the number of registers in this class.
  ///
  unsigned getNumRegs() const { return RegsSize; }

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
  const char *Name;         // Printable name for the reg (for debugging)
  uint32_t   Overlaps;      // Overlapping registers, described above
  uint32_t   SubRegs;       // Sub-register set, described above
  uint32_t   SuperRegs;     // Super-register set, described above
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

  /// DwarfLLVMRegPair - Emitted by tablegen so Dwarf<->LLVM reg mappings can be
  /// performed with a binary search.
  struct DwarfLLVMRegPair {
    unsigned FromReg;
    unsigned ToReg;

    bool operator==(unsigned Reg) const { return FromReg == Reg; }
    bool operator<(unsigned Reg) const { return FromReg < Reg; }
  };
private:
  const MCRegisterDesc *Desc;                 // Pointer to the descriptor array
  unsigned NumRegs;                           // Number of entries in the array
  unsigned RAReg;                             // Return address register
  const MCRegisterClass *Classes;             // Pointer to the regclass array
  unsigned NumClasses;                        // Number of entries in the array
  const uint16_t *RegLists;                   // Pointer to the reglists array
  const uint16_t *SubRegIndices;              // Pointer to the subreg lookup
                                              // array.
  unsigned NumSubRegIndices;                  // Number of subreg indices.

  unsigned L2DwarfRegsSize;
  unsigned EHL2DwarfRegsSize;
  unsigned Dwarf2LRegsSize;
  unsigned EHDwarf2LRegsSize;
  const DwarfLLVMRegPair *L2DwarfRegs;        // LLVM to Dwarf regs mapping
  const DwarfLLVMRegPair *EHL2DwarfRegs;      // LLVM to Dwarf regs mapping EH
  const DwarfLLVMRegPair *Dwarf2LRegs;        // Dwarf to LLVM regs mapping
  const DwarfLLVMRegPair *EHDwarf2LRegs;      // Dwarf to LLVM regs mapping EH
  DenseMap<unsigned, int> L2SEHRegs;          // LLVM to SEH regs mapping

public:
  /// InitMCRegisterInfo - Initialize MCRegisterInfo, called by TableGen
  /// auto-generated routines. *DO NOT USE*.
  void InitMCRegisterInfo(const MCRegisterDesc *D, unsigned NR, unsigned RA,
                          const MCRegisterClass *C, unsigned NC,
                          const uint16_t *RL,
                          const uint16_t *SubIndices,
                          unsigned NumIndices) {
    Desc = D;
    NumRegs = NR;
    RAReg = RA;
    Classes = C;
    RegLists = RL;
    NumClasses = NC;
    SubRegIndices = SubIndices;
    NumSubRegIndices = NumIndices;
  }

  /// mapLLVMRegsToDwarfRegs - Used to initialize LLVM register to Dwarf
  /// register number mapping. Called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void mapLLVMRegsToDwarfRegs(const DwarfLLVMRegPair *Map, unsigned Size,
                              bool isEH) {
    if (isEH) {
      EHL2DwarfRegs = Map;
      EHL2DwarfRegsSize = Size;
    } else {
      L2DwarfRegs = Map;
      L2DwarfRegsSize = Size;
    }
  }

  /// mapDwarfRegsToLLVMRegs - Used to initialize Dwarf register to LLVM
  /// register number mapping. Called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void mapDwarfRegsToLLVMRegs(const DwarfLLVMRegPair *Map, unsigned Size,
                              bool isEH) {
    if (isEH) {
      EHDwarf2LRegs = Map;
      EHDwarf2LRegsSize = Size;
    } else {
      Dwarf2LRegs = Map;
      Dwarf2LRegsSize = Size;
    }
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
  const uint16_t *getAliasSet(unsigned RegNo) const {
    // The Overlaps set always begins with Reg itself.
    return RegLists + get(RegNo).Overlaps + 1;
  }

  /// getOverlaps - Return a list of registers that overlap Reg, including
  /// itself. This is the same as the alias set except Reg is included in the
  /// list.
  /// These are exactly the registers in { x | regsOverlap(x, Reg) }.
  ///
  const uint16_t *getOverlaps(unsigned RegNo) const {
    return RegLists + get(RegNo).Overlaps;
  }

  /// getSubRegisters - Return the list of registers that are sub-registers of
  /// the specified register, or a null list of there are none. The list
  /// returned is zero terminated and sorted according to super-sub register
  /// relations. e.g. X86::RAX's sub-register list is EAX, AX, AL, AH.
  ///
  const uint16_t *getSubRegisters(unsigned RegNo) const {
    return RegLists + get(RegNo).SubRegs;
  }

  /// getSubReg - Returns the physical register number of sub-register "Index"
  /// for physical register RegNo. Return zero if the sub-register does not
  /// exist.
  unsigned getSubReg(unsigned Reg, unsigned Idx) const {
    return *(SubRegIndices + (Reg - 1) * NumSubRegIndices + Idx - 1);
  }

  /// getMatchingSuperReg - Return a super-register of the specified register
  /// Reg so its sub-register of index SubIdx is Reg.
  unsigned getMatchingSuperReg(unsigned Reg, unsigned SubIdx,
                               const MCRegisterClass *RC) const {
    for (const uint16_t *SRs = getSuperRegisters(Reg); unsigned SR = *SRs;++SRs)
      if (Reg == getSubReg(SR, SubIdx) && RC->contains(SR))
        return SR;
    return 0;
  }

  /// getSubRegIndex - For a given register pair, return the sub-register index
  /// if the second register is a sub-register of the first. Return zero
  /// otherwise.
  unsigned getSubRegIndex(unsigned RegNo, unsigned SubRegNo) const {
    for (unsigned I = 1; I <= NumSubRegIndices; ++I)
      if (getSubReg(RegNo, I) == SubRegNo)
        return I;
    return 0;
  }

  /// getSuperRegisters - Return the list of registers that are super-registers
  /// of the specified register, or a null list of there are none. The list
  /// returned is zero terminated and sorted according to super-sub register
  /// relations. e.g. X86::AL's super-register list is AX, EAX, RAX.
  ///
  const uint16_t *getSuperRegisters(unsigned RegNo) const {
    return RegLists + get(RegNo).SuperRegs;
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
    const DwarfLLVMRegPair *M = isEH ? EHL2DwarfRegs : L2DwarfRegs;
    unsigned Size = isEH ? EHL2DwarfRegsSize : L2DwarfRegsSize;

    const DwarfLLVMRegPair *I = std::lower_bound(M, M+Size, RegNum);
    if (I == M+Size || I->FromReg != RegNum)
      return -1;
    return I->ToReg;
  }

  /// getLLVMRegNum - Map a dwarf register back to a target register.
  ///
  int getLLVMRegNum(unsigned RegNum, bool isEH) const {
    const DwarfLLVMRegPair *M = isEH ? EHDwarf2LRegs : Dwarf2LRegs;
    unsigned Size = isEH ? EHDwarf2LRegsSize : Dwarf2LRegsSize;

    const DwarfLLVMRegPair *I = std::lower_bound(M, M+Size, RegNum);
    assert(I != M+Size && I->FromReg == RegNum && "Invalid RegNum");
    return I->ToReg;
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
