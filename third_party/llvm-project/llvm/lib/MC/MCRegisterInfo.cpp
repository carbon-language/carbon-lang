//===- MC/MCRegisterInfo.cpp - Target Register Description ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MCRegisterInfo functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace llvm;

MCRegister
MCRegisterInfo::getMatchingSuperReg(MCRegister Reg, unsigned SubIdx,
                                    const MCRegisterClass *RC) const {
  for (MCSuperRegIterator Supers(Reg, this); Supers.isValid(); ++Supers)
    if (RC->contains(*Supers) && Reg == getSubReg(*Supers, SubIdx))
      return *Supers;
  return 0;
}

MCRegister MCRegisterInfo::getSubReg(MCRegister Reg, unsigned Idx) const {
  assert(Idx && Idx < getNumSubRegIndices() &&
         "This is not a subregister index");
  // Get a pointer to the corresponding SubRegIndices list. This list has the
  // name of each sub-register in the same order as MCSubRegIterator.
  const uint16_t *SRI = SubRegIndices + get(Reg).SubRegIndices;
  for (MCSubRegIterator Subs(Reg, this); Subs.isValid(); ++Subs, ++SRI)
    if (*SRI == Idx)
      return *Subs;
  return 0;
}

unsigned MCRegisterInfo::getSubRegIndex(MCRegister Reg,
                                        MCRegister SubReg) const {
  assert(SubReg && SubReg < getNumRegs() && "This is not a register");
  // Get a pointer to the corresponding SubRegIndices list. This list has the
  // name of each sub-register in the same order as MCSubRegIterator.
  const uint16_t *SRI = SubRegIndices + get(Reg).SubRegIndices;
  for (MCSubRegIterator Subs(Reg, this); Subs.isValid(); ++Subs, ++SRI)
    if (*Subs == SubReg)
      return *SRI;
  return 0;
}

unsigned MCRegisterInfo::getSubRegIdxSize(unsigned Idx) const {
  assert(Idx && Idx < getNumSubRegIndices() &&
         "This is not a subregister index");
  return SubRegIdxRanges[Idx].Size;
}

unsigned MCRegisterInfo::getSubRegIdxOffset(unsigned Idx) const {
  assert(Idx && Idx < getNumSubRegIndices() &&
         "This is not a subregister index");
  return SubRegIdxRanges[Idx].Offset;
}

int MCRegisterInfo::getDwarfRegNum(MCRegister RegNum, bool isEH) const {
  const DwarfLLVMRegPair *M = isEH ? EHL2DwarfRegs : L2DwarfRegs;
  unsigned Size = isEH ? EHL2DwarfRegsSize : L2DwarfRegsSize;

  if (!M)
    return -1;
  DwarfLLVMRegPair Key = { RegNum, 0 };
  const DwarfLLVMRegPair *I = std::lower_bound(M, M+Size, Key);
  if (I == M+Size || I->FromReg != RegNum)
    return -1;
  return I->ToReg;
}

Optional<unsigned> MCRegisterInfo::getLLVMRegNum(unsigned RegNum,
                                                 bool isEH) const {
  const DwarfLLVMRegPair *M = isEH ? EHDwarf2LRegs : Dwarf2LRegs;
  unsigned Size = isEH ? EHDwarf2LRegsSize : Dwarf2LRegsSize;

  if (!M)
    return None;
  DwarfLLVMRegPair Key = { RegNum, 0 };
  const DwarfLLVMRegPair *I = std::lower_bound(M, M+Size, Key);
  if (I != M + Size && I->FromReg == RegNum)
    return I->ToReg;
  return None;
}

int MCRegisterInfo::getDwarfRegNumFromDwarfEHRegNum(unsigned RegNum) const {
  // On ELF platforms, DWARF EH register numbers are the same as DWARF
  // other register numbers.  On Darwin x86, they differ and so need to be
  // mapped.  The .cfi_* directives accept integer literals as well as
  // register names and should generate exactly what the assembly code
  // asked for, so there might be DWARF/EH register numbers that don't have
  // a corresponding LLVM register number at all.  So if we can't map the
  // EH register number to an LLVM register number, assume it's just a
  // valid DWARF register number as is.
  if (Optional<unsigned> LRegNum = getLLVMRegNum(RegNum, true))
    return getDwarfRegNum(*LRegNum, false);
  return RegNum;
}

int MCRegisterInfo::getSEHRegNum(MCRegister RegNum) const {
  const DenseMap<MCRegister, int>::const_iterator I = L2SEHRegs.find(RegNum);
  if (I == L2SEHRegs.end()) return (int)RegNum;
  return I->second;
}

int MCRegisterInfo::getCodeViewRegNum(MCRegister RegNum) const {
  if (L2CVRegs.empty())
    report_fatal_error("target does not implement codeview register mapping");
  const DenseMap<MCRegister, int>::const_iterator I = L2CVRegs.find(RegNum);
  if (I == L2CVRegs.end())
    report_fatal_error("unknown codeview register " + (RegNum < getNumRegs()
                                                           ? getName(RegNum)
                                                           : Twine(RegNum)));
  return I->second;
}
