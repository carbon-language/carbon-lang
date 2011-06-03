//===-- RegisterClassInfo.cpp - Dynamic Register Class Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegisterClassInfo class which provides dynamic
// information about target register classes. Callee saved and reserved
// registers depends on calling conventions and other dynamic information, so
// some things cannot be determined statically.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "RegisterClassInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

RegisterClassInfo::RegisterClassInfo() : Tag(0), MF(0), TRI(0), CalleeSaved(0)
{}

void RegisterClassInfo::runOnMachineFunction(const MachineFunction &mf) {
  bool Update = false;
  MF = &mf;

  // Allocate new array the first time we see a new target.
  if (MF->getTarget().getRegisterInfo() != TRI) {
    TRI = MF->getTarget().getRegisterInfo();
    RegClass.reset(new RCInfo[TRI->getNumRegClasses()]);
    Update = true;
  }

  // Does this MF have different CSRs?
  const unsigned *CSR = TRI->getCalleeSavedRegs(MF);
  if (Update || CSR != CalleeSaved) {
    // Build a CSRNum map. Every CSR alias gets an entry pointing to the last
    // overlapping CSR.
    CSRNum.clear();
    CSRNum.resize(TRI->getNumRegs(), 0);
    for (unsigned N = 0; unsigned Reg = CSR[N]; ++N)
      for (const unsigned *AS = TRI->getOverlaps(Reg);
           unsigned Alias = *AS; ++AS)
        CSRNum[Alias] = N + 1; // 0 means no CSR, 1 means CalleeSaved[0], ...
    Update = true;
  }
  CalleeSaved = CSR;

  // Different reserved registers?
  BitVector RR = TRI->getReservedRegs(*MF);
  if (RR != Reserved)
    Update = true;
  Reserved = RR;

  // Invalidate cached information from previous function.
  if (Update)
    ++Tag;
}

/// compute - Compute the preferred allocation order for RC with reserved
/// registers filtered out. Volatile registers come first followed by CSR
/// aliases ordered according to the CSR order specified by the target.
void RegisterClassInfo::compute(const TargetRegisterClass *RC) const {
  RCInfo &RCI = RegClass[RC->getID()];

  // Raw register count, including all reserved regs.
  unsigned NumRegs = RC->getNumRegs();

  if (!RCI.Order)
    RCI.Order.reset(new unsigned[NumRegs]);

  unsigned N = 0;
  SmallVector<std::pair<unsigned, unsigned>, 8> CSRAlias;

  // FIXME: Once targets reserve registers instead of removing them from the
  // allocation order, we can simply use begin/end here.
  TargetRegisterClass::iterator AOB = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator AOE = RC->allocation_order_end(*MF);

  for (TargetRegisterClass::iterator I = AOB; I != AOE; ++I) {
    unsigned PhysReg = *I;
    // Remove reserved registers from the allocation order.
    if (Reserved.test(PhysReg))
      continue;
    if (unsigned CSR = CSRNum[PhysReg])
      // PhysReg aliases a CSR, save it for later.  Provide a (CSR, N) sort key
      // to preserve the original ordering of multiple aliases of the same CSR.
      CSRAlias.push_back(std::make_pair((CSR << 16) + (I - AOB), PhysReg));
    else
      RCI.Order[N++] = PhysReg;
  }
  RCI.NumRegs = N + CSRAlias.size();
  assert (RCI.NumRegs <= NumRegs && "Allocation order larger than regclass");

  // Sort CSR aliases acording to the CSR ordering.
  if (CSRAlias.size() >= 2)
    array_pod_sort(CSRAlias.begin(), CSRAlias.end());

  for (unsigned i = 0, e = CSRAlias.size(); i != e; ++i)
      RCI.Order[N++] = CSRAlias[i].second;

  DEBUG({
    dbgs() << "AllocationOrder(" << RC->getName() << ") = [";
    for (unsigned I = 0; I != N; ++I)
      dbgs() << ' ' << PrintReg(RCI.Order[I], TRI);
    dbgs() << " ]\n";
  });

  // RCI is now up-to-date.
  RCI.Tag = Tag;
}

