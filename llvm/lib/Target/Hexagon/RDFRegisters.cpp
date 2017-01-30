//===--- RDFRegisters.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RDFRegisters.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;
using namespace rdf;

PhysicalRegisterInfo::PhysicalRegisterInfo(const TargetRegisterInfo &tri,
      const MachineFunction &mf)
    : TRI(tri) {
  RegInfos.resize(TRI.getNumRegs());

  BitVector BadRC(TRI.getNumRegs());
  for (const TargetRegisterClass *RC : TRI.regclasses()) {
    for (MCPhysReg R : *RC) {
      RegInfo &RI = RegInfos[R];
      if (RI.RegClass != nullptr && !BadRC[R]) {
        if (RC->LaneMask != RI.RegClass->LaneMask) {
          BadRC.set(R);
          RI.RegClass = nullptr;
        }
      } else
        RI.RegClass = RC;
    }
  }

  for (MCPhysReg R = 1, NR = TRI.getNumRegs(); R != NR; ++R) {
    MCPhysReg SuperR = R;
    for (MCSuperRegIterator S(R, &TRI, false); S.isValid(); ++S)
      SuperR = *S;
    RegInfos[R].MaxSuper = SuperR;
  }
}

bool PhysicalRegisterInfo::alias(RegisterRef RA, RegisterRef RB) const {
  assert(TargetRegisterInfo::isPhysicalRegister(RA.Reg));
  assert(TargetRegisterInfo::isPhysicalRegister(RB.Reg));
  MCRegUnitMaskIterator UMA(RA.Reg, &TRI);
  MCRegUnitMaskIterator UMB(RB.Reg, &TRI);
  // Reg units are returned in the numerical order.
  while (UMA.isValid() && UMB.isValid()) {
    std::pair<uint32_t,LaneBitmask> PA = *UMA;
    std::pair<uint32_t,LaneBitmask> PB = *UMB;
    if (PA.first == PB.first) {
      // Lane mask of 0 (given by the iterator) should be treated as "full".
      // This can happen when the register has only one unit, or when the
      // unit corresponds to explicit aliasing. In such cases, the lane mask
      // from RegisterRef should be ignored.
      if (PA.second.none() || PB.second.none())
        return true;

      // At this point the common unit corresponds to a subregister. The lane
      // masks correspond to the lane mask of that unit within the original
      // register, for example assuming register quadruple q0 = r3:0, and
      // a register pair d1 = r3:2, the lane mask of r2 in q0 may be 0b0100,
      // while the lane mask of r2 in d1 may be 0b0001.
      LaneBitmask LA = PA.second & RA.Mask;
      LaneBitmask LB = PB.second & RB.Mask;
      if (LA.any() && LB.any()) {
        unsigned Root = *MCRegUnitRootIterator(PA.first, &TRI);
        // If register units were guaranteed to only have 1 bit in any lane
        // mask, the code below would not be necessary. This is because LA
        // and LB would have at most 1 bit set each, and that bit would be
        // guaranteed to correspond to the given register unit.
        uint32_t SubA = TRI.getSubRegIndex(RA.Reg, Root);
        uint32_t SubB = TRI.getSubRegIndex(RB.Reg, Root);
        const TargetRegisterClass *RC = RegInfos[Root].RegClass;
        LaneBitmask RCMask = RC != nullptr ? RC->LaneMask : LaneBitmask(0x1);
        LaneBitmask MaskA = TRI.reverseComposeSubRegIndexLaneMask(SubA, LA);
        LaneBitmask MaskB = TRI.reverseComposeSubRegIndexLaneMask(SubB, LB);
        if ((MaskA & MaskB & RCMask).any())
          return true;
      }

      ++UMA;
      ++UMB;
      continue;
    }
    if (PA.first < PB.first)
      ++UMA;
    else if (PB.first < PA.first)
      ++UMB;
  }
  return false;
}

RegisterRef RegisterAggr::normalize(RegisterRef RR) const {
  const TargetRegisterClass *RC = PRI.RegInfos[RR.Reg].RegClass;
  LaneBitmask RCMask = RC != nullptr ? RC->LaneMask : LaneBitmask(0x00000001);
  LaneBitmask Common = RR.Mask & RCMask;

  RegisterId SuperReg = PRI.RegInfos[RR.Reg].MaxSuper;
// Ex: IP/EIP/RIP
//  assert(RC != nullptr || RR.Reg == SuperReg);
  uint32_t Sub = PRI.getTRI().getSubRegIndex(SuperReg, RR.Reg);
  LaneBitmask SuperMask = PRI.getTRI().composeSubRegIndexLaneMask(Sub, Common);
  return RegisterRef(SuperReg, SuperMask);
}

bool RegisterAggr::hasAliasOf(RegisterRef RR) const {
  RegisterRef NR = normalize(RR);
  auto F = Masks.find(NR.Reg);
  if (F != Masks.end()) {
    if ((F->second & NR.Mask).any())
      return true;
  }
  if (CheckUnits) {
    for (MCRegUnitIterator U(RR.Reg, &PRI.getTRI()); U.isValid(); ++U)
      if (ExpAliasUnits.test(*U))
        return true;
  }
  return false;
}

bool RegisterAggr::hasCoverOf(RegisterRef RR) const {
  // Always have a cover for empty lane mask.
  RegisterRef NR = normalize(RR);
  if (NR.Mask.none())
    return true;
  auto F = Masks.find(NR.Reg);
  if (F == Masks.end())
    return false;
  return (NR.Mask & F->second) == NR.Mask;
}

RegisterAggr &RegisterAggr::insert(RegisterRef RR) {
  RegisterRef NR = normalize(RR);
  auto F = Masks.find(NR.Reg);
  if (F == Masks.end())
    Masks.insert({NR.Reg, NR.Mask});
  else
    F->second |= NR.Mask;

  // Visit all register units to see if there are any that were created
  // by explicit aliases. Add those that were to the bit vector.
  const TargetRegisterInfo &TRI = PRI.getTRI();
  for (MCRegUnitIterator U(RR.Reg, &TRI); U.isValid(); ++U) {
    MCRegUnitRootIterator R(*U, &TRI);
    ++R;
    if (!R.isValid())
      continue;
    ExpAliasUnits.set(*U);
    CheckUnits = true;
  }
  return *this;
}

RegisterAggr &RegisterAggr::insert(const RegisterAggr &RG) {
  for (std::pair<RegisterId,LaneBitmask> P : RG.Masks)
    insert(RegisterRef(P.first, P.second));
  return *this;
}

RegisterAggr &RegisterAggr::clear(RegisterRef RR) {
  RegisterRef NR = normalize(RR);
  auto F = Masks.find(NR.Reg);
  if (F == Masks.end())
    return *this;
  LaneBitmask NewM = F->second & ~NR.Mask;
  if (NewM.none())
    Masks.erase(F);
  else
    F->second = NewM;
  return *this;
}

RegisterAggr &RegisterAggr::clear(const RegisterAggr &RG) {
  for (std::pair<RegisterId,LaneBitmask> P : RG.Masks)
    clear(RegisterRef(P.first, P.second));
  return *this;
}

RegisterRef RegisterAggr::clearIn(RegisterRef RR) const {
  RegisterAggr T(PRI);
  T.insert(RR).clear(*this);
  if (T.empty())
    return RegisterRef();
  return RegisterRef(T.begin()->first, T.begin()->second);
}

void RegisterAggr::print(raw_ostream &OS) const {
  OS << '{';
  for (auto I : Masks)
    OS << ' ' << PrintReg(I.first, &PRI.getTRI())
       << PrintLaneMaskOpt(I.second);
  OS << " }";
}

