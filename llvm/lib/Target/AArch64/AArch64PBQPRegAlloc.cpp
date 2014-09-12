//===-- AArch64PBQPRegAlloc.cpp - AArch64 specific PBQP constraints -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file contains the AArch64 / Cortex-A57 specific register allocation
// constraints for use by the PBQP register allocator.
//
// It is essentially a transcription of what is contained in
// AArch64A57FPLoadBalancing, which tries to use a balanced
// mix of odd and even D-registers when performing a critical sequence of
// independent, non-quadword FP/ASIMD floating-point multiply-accumulates.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "aarch64-pbqp"

#include "AArch64.h"
#include "AArch64RegisterInfo.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocPBQP.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PBQP_BUILDER PBQPBuilderWithCoalescing

using namespace llvm;

namespace {

#ifndef NDEBUG
bool isFPReg(unsigned reg) {
  return AArch64::FPR32RegClass.contains(reg) ||
         AArch64::FPR64RegClass.contains(reg) ||
         AArch64::FPR128RegClass.contains(reg);
}
#endif

bool isOdd(unsigned reg) {
  switch (reg) {
  default:
    llvm_unreachable("Register is not from the expected class !");
  case AArch64::S1:
  case AArch64::S3:
  case AArch64::S5:
  case AArch64::S7:
  case AArch64::S9:
  case AArch64::S11:
  case AArch64::S13:
  case AArch64::S15:
  case AArch64::S17:
  case AArch64::S19:
  case AArch64::S21:
  case AArch64::S23:
  case AArch64::S25:
  case AArch64::S27:
  case AArch64::S29:
  case AArch64::S31:
  case AArch64::D1:
  case AArch64::D3:
  case AArch64::D5:
  case AArch64::D7:
  case AArch64::D9:
  case AArch64::D11:
  case AArch64::D13:
  case AArch64::D15:
  case AArch64::D17:
  case AArch64::D19:
  case AArch64::D21:
  case AArch64::D23:
  case AArch64::D25:
  case AArch64::D27:
  case AArch64::D29:
  case AArch64::D31:
  case AArch64::Q1:
  case AArch64::Q3:
  case AArch64::Q5:
  case AArch64::Q7:
  case AArch64::Q9:
  case AArch64::Q11:
  case AArch64::Q13:
  case AArch64::Q15:
  case AArch64::Q17:
  case AArch64::Q19:
  case AArch64::Q21:
  case AArch64::Q23:
  case AArch64::Q25:
  case AArch64::Q27:
  case AArch64::Q29:
  case AArch64::Q31:
    return true;
  case AArch64::S0:
  case AArch64::S2:
  case AArch64::S4:
  case AArch64::S6:
  case AArch64::S8:
  case AArch64::S10:
  case AArch64::S12:
  case AArch64::S14:
  case AArch64::S16:
  case AArch64::S18:
  case AArch64::S20:
  case AArch64::S22:
  case AArch64::S24:
  case AArch64::S26:
  case AArch64::S28:
  case AArch64::S30:
  case AArch64::D0:
  case AArch64::D2:
  case AArch64::D4:
  case AArch64::D6:
  case AArch64::D8:
  case AArch64::D10:
  case AArch64::D12:
  case AArch64::D14:
  case AArch64::D16:
  case AArch64::D18:
  case AArch64::D20:
  case AArch64::D22:
  case AArch64::D24:
  case AArch64::D26:
  case AArch64::D28:
  case AArch64::D30:
  case AArch64::Q0:
  case AArch64::Q2:
  case AArch64::Q4:
  case AArch64::Q6:
  case AArch64::Q8:
  case AArch64::Q10:
  case AArch64::Q12:
  case AArch64::Q14:
  case AArch64::Q16:
  case AArch64::Q18:
  case AArch64::Q20:
  case AArch64::Q22:
  case AArch64::Q24:
  case AArch64::Q26:
  case AArch64::Q28:
  case AArch64::Q30:
    return false;

  }
}

bool haveSameParity(unsigned reg1, unsigned reg2) {
  assert(isFPReg(reg1) && "Expecting an FP register for reg1");
  assert(isFPReg(reg2) && "Expecting an FP register for reg2");

  return isOdd(reg1) == isOdd(reg2);
}

class A57PBQPBuilder : public PBQP_BUILDER {
public:
  A57PBQPBuilder() : PBQP_BUILDER(), TRI(nullptr), LIs(nullptr), Chains() {}

  // Build a PBQP instance to represent the register allocation problem for
  // the given MachineFunction.
  std::unique_ptr<PBQPRAProblem>
  build(MachineFunction *MF, const LiveIntervals *LI,
        const MachineBlockFrequencyInfo *blockInfo,
        const RegSet &VRegs) override;

private:
  const AArch64RegisterInfo *TRI;
  const LiveIntervals *LIs;
  SmallSetVector<unsigned, 32> Chains;

  // Return true if reg is a physical register
  bool isPhysicalReg(unsigned reg) const {
    return TRI->isPhysicalRegister(reg);
  }

  // Add the accumulator chaining constraint, inside the chain, i.e. so that
  // parity(Rd) == parity(Ra).
  // \return true if a constraint was added
  bool addIntraChainConstraint(PBQPRAProblem *p, unsigned Rd, unsigned Ra);

  // Add constraints between existing chains
  void addInterChainConstraint(PBQPRAProblem *p, unsigned Rd, unsigned Ra);
};
} // Anonymous namespace

bool A57PBQPBuilder::addIntraChainConstraint(PBQPRAProblem *p, unsigned Rd,
                                             unsigned Ra) {
  if (Rd == Ra)
    return false;

  if (isPhysicalReg(Rd) || isPhysicalReg(Ra)) {
    DEBUG(dbgs() << "Rd is a physical reg:" << isPhysicalReg(Rd) << '\n');
    DEBUG(dbgs() << "Ra is a physical reg:" << isPhysicalReg(Ra) << '\n');
    return false;
  }

  const PBQPRAProblem::AllowedSet *vRdAllowed = &p->getAllowedSet(Rd);
  const PBQPRAProblem::AllowedSet *vRaAllowed = &p->getAllowedSet(Ra);

  PBQPRAGraph &g = p->getGraph();
  PBQPRAGraph::NodeId node1 = p->getNodeForVReg(Rd);
  PBQPRAGraph::NodeId node2 = p->getNodeForVReg(Ra);
  PBQPRAGraph::EdgeId edge = g.findEdge(node1, node2);

  // The edge does not exist. Create one with the appropriate interference
  // costs.
  if (edge == g.invalidEdgeId()) {
    const LiveInterval &ld = LIs->getInterval(Rd);
    const LiveInterval &la = LIs->getInterval(Ra);
    bool livesOverlap = ld.overlaps(la);

    PBQP::Matrix costs(vRdAllowed->size() + 1, vRaAllowed->size() + 1, 0);
    for (unsigned i = 0, ie = vRdAllowed->size(); i != ie; ++i) {
      unsigned pRd = (*vRdAllowed)[i];
      for (unsigned j = 0, je = vRaAllowed->size(); j != je; ++j) {
        unsigned pRa = (*vRaAllowed)[j];
        if (livesOverlap && TRI->regsOverlap(pRd, pRa))
          costs[i + 1][j + 1] = std::numeric_limits<PBQP::PBQPNum>::infinity();
        else
          costs[i + 1][j + 1] = haveSameParity(pRd, pRa) ? 0.0 : 1.0;
      }
    }
    g.addEdge(node1, node2, std::move(costs));
    return true;
  }

  if (g.getEdgeNode1Id(edge) == node2) {
    std::swap(node1, node2);
    std::swap(vRdAllowed, vRaAllowed);
  }

  // Enforce minCost(sameParity(RaClass)) > maxCost(otherParity(RdClass))
  PBQP::Matrix costs(g.getEdgeCosts(edge));
  for (unsigned i = 0, ie = vRdAllowed->size(); i != ie; ++i) {
    unsigned pRd = (*vRdAllowed)[i];

    // Get the maximum cost (excluding unallocatable reg) for same parity
    // registers
    PBQP::PBQPNum sameParityMax = std::numeric_limits<PBQP::PBQPNum>::min();
    for (unsigned j = 0, je = vRaAllowed->size(); j != je; ++j) {
      unsigned pRa = (*vRaAllowed)[j];
      if (haveSameParity(pRd, pRa))
        if (costs[i + 1][j + 1] !=
                std::numeric_limits<PBQP::PBQPNum>::infinity() &&
            costs[i + 1][j + 1] > sameParityMax)
          sameParityMax = costs[i + 1][j + 1];
    }

    // Ensure all registers with a different parity have a higher cost
    // than sameParityMax
    for (unsigned j = 0, je = vRaAllowed->size(); j != je; ++j) {
      unsigned pRa = (*vRaAllowed)[j];
      if (!haveSameParity(pRd, pRa))
        if (sameParityMax > costs[i + 1][j + 1])
          costs[i + 1][j + 1] = sameParityMax + 1.0;
    }
  }
  g.setEdgeCosts(edge, costs);

  return true;
}

void
A57PBQPBuilder::addInterChainConstraint(PBQPRAProblem *p, unsigned Rd,
                                        unsigned Ra) {
  // Do some Chain management
  if (Chains.count(Ra)) {
    if (Rd != Ra) {
      DEBUG(dbgs() << "Moving acc chain from " << PrintReg(Ra, TRI) << " to "
                   << PrintReg(Rd, TRI) << '\n';);
      Chains.remove(Ra);
      Chains.insert(Rd);
    }
  } else {
    DEBUG(dbgs() << "Creating new acc chain for " << PrintReg(Rd, TRI)
                 << '\n';);
    Chains.insert(Rd);
  }

  const LiveInterval &ld = LIs->getInterval(Rd);
  for (auto r : Chains) {
    // Skip self
    if (r == Rd)
      continue;

    const LiveInterval &lr = LIs->getInterval(r);
    if (ld.overlaps(lr)) {
      const PBQPRAProblem::AllowedSet *vRdAllowed = &p->getAllowedSet(Rd);
      const PBQPRAProblem::AllowedSet *vRrAllowed = &p->getAllowedSet(r);

      PBQPRAGraph &g = p->getGraph();
      PBQPRAGraph::NodeId node1 = p->getNodeForVReg(Rd);
      PBQPRAGraph::NodeId node2 = p->getNodeForVReg(r);
      PBQPRAGraph::EdgeId edge = g.findEdge(node1, node2);
      assert(edge != g.invalidEdgeId() &&
             "PBQP error ! The edge should exist !");

      DEBUG(dbgs() << "Refining constraint !\n";);

      if (g.getEdgeNode1Id(edge) == node2) {
        std::swap(node1, node2);
        std::swap(vRdAllowed, vRrAllowed);
      }

      // Enforce that cost is higher with all other Chains of the same parity
      PBQP::Matrix costs(g.getEdgeCosts(edge));
      for (unsigned i = 0, ie = vRdAllowed->size(); i != ie; ++i) {
        unsigned pRd = (*vRdAllowed)[i];

        // Get the maximum cost (excluding unallocatable reg) for all other
        // parity registers
        PBQP::PBQPNum sameParityMax = std::numeric_limits<PBQP::PBQPNum>::min();
        for (unsigned j = 0, je = vRrAllowed->size(); j != je; ++j) {
          unsigned pRa = (*vRrAllowed)[j];
          if (!haveSameParity(pRd, pRa))
            if (costs[i + 1][j + 1] !=
                    std::numeric_limits<PBQP::PBQPNum>::infinity() &&
                costs[i + 1][j + 1] > sameParityMax)
              sameParityMax = costs[i + 1][j + 1];
        }

        // Ensure all registers with same parity have a higher cost
        // than sameParityMax
        for (unsigned j = 0, je = vRrAllowed->size(); j != je; ++j) {
          unsigned pRa = (*vRrAllowed)[j];
          if (haveSameParity(pRd, pRa))
            if (sameParityMax > costs[i + 1][j + 1])
              costs[i + 1][j + 1] = sameParityMax + 1.0;
        }
      }
      g.setEdgeCosts(edge, costs);
    }
  }
}

std::unique_ptr<PBQPRAProblem>
A57PBQPBuilder::build(MachineFunction *MF, const LiveIntervals *LI,
                      const MachineBlockFrequencyInfo *blockInfo,
                      const RegSet &VRegs) {
  std::unique_ptr<PBQPRAProblem> p =
      PBQP_BUILDER::build(MF, LI, blockInfo, VRegs);

  TRI = static_cast<const AArch64RegisterInfo *>(
      MF->getTarget().getSubtargetImpl()->getRegisterInfo());
  LIs = LI;

  DEBUG(MF->dump(););

  for (MachineFunction::const_iterator mbbItr = MF->begin(), mbbEnd = MF->end();
       mbbItr != mbbEnd; ++mbbItr) {
    const MachineBasicBlock *MBB = &*mbbItr;
    Chains.clear(); // FIXME: really needed ? Could not work at MF level ?

    for (MachineBasicBlock::const_iterator miItr = MBB->begin(),
                                           miEnd = MBB->end();
         miItr != miEnd; ++miItr) {
      const MachineInstr *MI = &*miItr;
      switch (MI->getOpcode()) {
      case AArch64::FMSUBSrrr:
      case AArch64::FMADDSrrr:
      case AArch64::FNMSUBSrrr:
      case AArch64::FNMADDSrrr:
      case AArch64::FMSUBDrrr:
      case AArch64::FMADDDrrr:
      case AArch64::FNMSUBDrrr:
      case AArch64::FNMADDDrrr: {
        unsigned Rd = MI->getOperand(0).getReg();
        unsigned Ra = MI->getOperand(3).getReg();

        if (addIntraChainConstraint(p.get(), Rd, Ra))
          addInterChainConstraint(p.get(), Rd, Ra);
        break;
      }

      case AArch64::FMLAv2f32:
      case AArch64::FMLSv2f32: {
        unsigned Rd = MI->getOperand(0).getReg();
        addInterChainConstraint(p.get(), Rd, Rd);
        break;
      }

      default:
        // Forget Chains which have been killed
        for (auto r : Chains) {
          SmallVector<unsigned, 8> toDel;
          if (MI->killsRegister(r)) {
            DEBUG(dbgs() << "Killing chain " << PrintReg(r, TRI) << " at ";
                  MI->print(dbgs()););
            toDel.push_back(r);
          }

          while (!toDel.empty()) {
            Chains.remove(toDel.back());
            toDel.pop_back();
          }
        }
      }
    }
  }

  return p;
}

// Factory function used by AArch64TargetMachine to add the pass to the
// passmanager.
FunctionPass *llvm::createAArch64A57PBQPRegAlloc() {
  std::unique_ptr<PBQP_BUILDER> builder = llvm::make_unique<A57PBQPBuilder>();
  return createPBQPRegisterAllocator(std::move(builder), nullptr);
}
