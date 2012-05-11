//===---- HexagonOptimizeConstExt.cpp - Optimize Constant Extender Use ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass traverses through all the basic blocks in a functions and replaces
// constant extended instruction with their register equivalent if the same
// constant is being used by more than two instructions.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "xfer"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "HexagonTargetMachine.h"
#include "HexagonConstExtInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/Support/CommandLine.h"
#define DEBUG_TYPE "xfer"

using namespace llvm;

namespace {

class HexagonOptimizeConstExt : public MachineFunctionPass {
  HexagonTargetMachine& QTM;
  const HexagonSubtarget &QST;

public:
  static char ID;
  HexagonOptimizeConstExt(HexagonTargetMachine& TM)
    : MachineFunctionPass(ID), QTM(TM), QST(*TM.getSubtargetImpl()) {}

  const char *getPassName() const {
    return "Remove sub-optimal uses of constant extenders";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
  }

  bool runOnMachineFunction(MachineFunction &Fn);
  void removeConstExtFromMI (const HexagonInstrInfo *TII, MachineInstr* oldMI,
                             unsigned DestReg);
};

char HexagonOptimizeConstExt::ID = 0;

// Remove constant extended instructions with the corresponding non-extended
// instruction.
void HexagonOptimizeConstExt::removeConstExtFromMI (const HexagonInstrInfo *TII,
                                                    MachineInstr* oldMI,
                                                    unsigned DestReg) {
  assert(HexagonConstExt::NonExtEquivalentExists(oldMI->getOpcode()) &&
         "Non-extended equivalent instruction doesn't exist");
  MachineBasicBlock *MBB = oldMI->getParent ();
  int oldOpCode = oldMI->getOpcode();
  unsigned short CExtOpNum = HexagonConstExt::getCExtOpNum(oldOpCode);
  unsigned numOperands = oldMI->getNumOperands();
  MachineInstrBuilder MIB = BuildMI(*MBB, oldMI, oldMI->getDebugLoc(),
                TII->get(HexagonConstExt::getNonExtOpcode(oldMI->getOpcode())));

  for (unsigned i = 0; i < numOperands; ++i) {
    if (i == CExtOpNum) {
      MIB.addReg(DestReg);
      if (oldMI->getDesc().mayLoad()) {
        // As of now, only absolute addressing mode instructions can load from
        // global addresses. Other addressing modes allow only constant
        // literals. Load with absolute addressing mode gets replaced with the
        // corresponding base+offset load.
        if (oldMI->getOperand(i).isGlobal()) {
          MIB.addImm(oldMI->getOperand(i).getOffset());
        }
        else
          MIB.addImm(0);
      }
      else if (oldMI->getDesc().mayStore()){
        if (oldMI->getOperand(i).isGlobal()) {
          // If stored value is a global address and is extended, it is required
          // to have 0 offset.
          if (CExtOpNum == (numOperands-1))
            assert((oldMI->getOperand(i).getOffset()==0) && "Invalid Offset");
          else
            MIB.addImm(oldMI->getOperand(i).getOffset());
        }
        else if (CExtOpNum != (numOperands-1))
          MIB.addImm(0);
      }
    }
    else {
      const MachineOperand &op = oldMI->getOperand(i);
      MIB.addOperand(op);
    }
  }
  DEBUG(dbgs () << "Removing old instr: " << *oldMI << "\n");
  DEBUG(dbgs() << "New instr: " << (*MIB) << "\n");
  oldMI->eraseFromParent();
}

// Returns false for the following instructions, since it may not be profitable
// to convert these instructions into a non-extended instruction if the offset
// is non-zero.
static bool canHaveAnyOffset(MachineInstr* MI) {
  switch (MI->getOpcode()) {
    case Hexagon::STriw_offset_ext_V4:
    case Hexagon::STrih_offset_ext_V4:
      return false;
    default:
      return true;
  }
}

bool HexagonOptimizeConstExt::runOnMachineFunction(MachineFunction &Fn) {

  const HexagonInstrInfo *TII = QTM.getInstrInfo();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();

  // CExtMap maintains a list of instructions for each constant extended value.
  // It also keeps a flag for the value to indicate if it's a global address
  // or a constant literal.
  StringMap<std::pair<SmallVector<MachineInstr*, 8>, bool > > CExtMap;

  // Loop over all the basic blocks
  for (MachineFunction::iterator MBBb = Fn.begin(), MBBe = Fn.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;

    // Traverse the basic block and update a map of (ImmValue->MI)
    MachineBasicBlock::iterator MII = MBB->begin();
    MachineBasicBlock::iterator MIE = MBB->end ();

    while (MII != MIE) {
      MachineInstr *MI = MII;
      // Check if the instruction has any constant extended operand and also has
      //  a non-extended equivalent.
      if (TII->isConstExtended(MI) &&
          HexagonConstExt::NonExtEquivalentExists(MI->getOpcode())) {
        short ExtOpNum = HexagonConstExt::getCExtOpNum(MI->getOpcode());
        SmallString<256> TmpData;
        if (MI->getOperand(ExtOpNum).isImm()) {
          DEBUG(dbgs() << "Selected for replacement : " << *MI << "\n");
          int ImmValue = MI->getOperand(ExtOpNum).getImm();
          StringRef ExtValue = Twine(ImmValue).toStringRef(TmpData);
          CExtMap[ExtValue].first.push_back(MI);
          CExtMap[ExtValue].second = false;
        }
        else if (MI->getOperand(ExtOpNum).isGlobal()) {
          StringRef ExtValue = MI->getOperand(ExtOpNum).getGlobal()->getName();
          // If stored value is constant extended and has an offset, it's not
          // profitable to replace these instructions with the non-extended
          // version.
          if (MI->getOperand(ExtOpNum).getOffset() == 0
             || canHaveAnyOffset(MI)) {
            DEBUG(dbgs() << "Selected for replacement : " << *MI << "\n");
            CExtMap[ExtValue].first.push_back(MI);
            CExtMap[ExtValue].second = true;
          }
        }
      }
      ++MII;
    } // While ends
  }

  enum OpType {imm, GlobalAddr};
  // Process the constants that have been extended.
  for (StringMap<std::pair<SmallVector<MachineInstr*, 8>, bool> >::iterator II=
         CExtMap.begin(), IE = CExtMap.end(); II != IE; ++II) {

    SmallVector<MachineInstr*, 8> &MIList = (*II).second.first;

    // Replace the constant extended instructions with the non-extended
    // equivalent if more than 2 instructions extend the same constant value.
    if (MIList.size() <= 2)
      continue;

    bool ExtOpType = (*II).second.second;
    StringRef ExtValue = (*II).getKeyData();
    const GlobalValue *GV = NULL;
    unsigned char TargetFlags=0;
    int ExtOpNum = HexagonConstExt::getCExtOpNum(MIList[0]->getOpcode());
    SmallVector<MachineBasicBlock*, 8> MachineBlocks;

    if (ExtOpType == GlobalAddr) {
      GV = MIList[0]->getOperand(ExtOpNum).getGlobal();
      TargetFlags = MIList[0]->getOperand(ExtOpNum).getTargetFlags();
    }

    // For each instruction in the list, record the block it belongs to.
    for (SmallVector<MachineInstr*, 8>::iterator LB = MIList.begin(),
           LE = MIList.end(); LB != LE; ++LB) {
      MachineInstr *MI = (*LB);
      MachineBlocks.push_back (MI->getParent());
    }

    MachineBasicBlock* CommDomBlock = MachineBlocks[0];
    MachineBasicBlock* oldCommDomBlock = NULL;
    // replaceMIs is the list of instructions to be replaced with a
    // non-extended equivalent instruction.
    // The idea here is that not all the instructions in the MIList will
    // be replaced with a register.
    SmallVector<MachineInstr*, 8> replaceMIs;
    replaceMIs.push_back(MIList[0]);

    for (unsigned i= 1; i < MachineBlocks.size(); ++i) {
      oldCommDomBlock = CommDomBlock;
      MachineBasicBlock *BB = MachineBlocks[i];
      CommDomBlock = MDT.findNearestCommonDominator(&(*CommDomBlock),
                                                    &(*BB));
      if (!CommDomBlock) {
        CommDomBlock = oldCommDomBlock;
        break;
      }
      replaceMIs.push_back(MIList[i]);
    }

    // Insert into CommDomBlock.
    if (CommDomBlock) {
      unsigned DestReg = TII->createVR (CommDomBlock->getParent(), MVT::i32);
      MachineInstr *firstMI = CommDomBlock->getFirstNonPHI();
      if (ExtOpType == imm) {
        int ImmValue = 0;
        ExtValue.getAsInteger(10,ImmValue);
        BuildMI (*CommDomBlock, firstMI, firstMI->getDebugLoc(),
                                     TII->get(Hexagon::TFRI), DestReg)
                                     .addImm(ImmValue);
      }
      else {
        BuildMI (*CommDomBlock, firstMI, firstMI->getDebugLoc(),
                                     TII->get(Hexagon::TFRI_V4), DestReg)
                                     .addGlobalAddress(GV, 0, TargetFlags);
      }
      for (unsigned i= 0; i < replaceMIs.size(); i++) {
        MachineInstr *oldMI = replaceMIs[i];
        removeConstExtFromMI(TII, oldMI, DestReg);
      }
      replaceMIs.clear();
    }
  }
  return true;
}
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *
llvm::createHexagonOptimizeConstExt(HexagonTargetMachine &TM) {
  return new HexagonOptimizeConstExt(TM);
}

