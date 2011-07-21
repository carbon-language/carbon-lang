//===-- Mips/MipsCodeEmitter.cpp - Convert Mips code to machine code -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file contains the pass that transforms the Mips machine instructions
// into relocatable machine code.
//
//===---------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "Mips.h"
#include "MipsInstrInfo.h"
#include "MipsRelocations.h"
#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#ifndef NDEBUG
#include <iomanip>
#endif

#include "llvm/CodeGen/MachineOperand.h"

using namespace llvm;

namespace {

class MipsCodeEmitter : public MachineFunctionPass {
  MipsJITInfo *JTI;
  const MipsInstrInfo *II;
  const TargetData *TD;
  const MipsSubtarget *Subtarget;
  TargetMachine &TM;
  JITCodeEmitter &MCE;
  const std::vector<MachineConstantPoolEntry> *MCPEs;
  const std::vector<MachineJumpTableEntry> *MJTEs;
  bool IsPIC;

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<MachineModuleInfo> ();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  static char ID;

  public:
    MipsCodeEmitter(TargetMachine &tm, JITCodeEmitter &mce) :
      MachineFunctionPass(ID), JTI(0),
        II((const MipsInstrInfo *) tm.getInstrInfo()),
        TD(tm.getTargetData()), TM(tm), MCE(mce), MCPEs(0), MJTEs(0),
        IsPIC(TM.getRelocationModel() == Reloc::PIC_) {
    }

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "Mips Machine Code Emitter";
    }

    void emitInstruction(const MachineInstr &MI);

    unsigned getOperandValue(const MachineOperand &MO,
        unsigned relocType = -1);

    void emitGlobalAddress(const GlobalValue *GV, unsigned Reloc,
        bool MayNeedFarStub = true);

    void emitMachineBasicBlock(MachineBasicBlock *BB, unsigned Reloc,
        intptr_t JTBase = 0);

    void emitExternalSymbolAddress(const char *ES, unsigned Reloc);
    void emitJumpTableAddress(unsigned JTIndex, unsigned Reloc) const;
    void emitConstPoolAddress(unsigned CPI, unsigned Reloc);
};
}

void MipsCodeEmitter::emitGlobalAddress(const GlobalValue *GV, unsigned Reloc,
    bool mayNeedFarStub) {
  MachineRelocation MR = MachineRelocation::getGV(MCE.getCurrentPCOffset(),
                  Reloc, const_cast<GlobalValue *> (GV), 0, mayNeedFarStub);
  MCE.addRelocation(MR);
}

/// emitMachineBasicBlock - Emit the specified address basic block.
void MipsCodeEmitter::emitMachineBasicBlock(MachineBasicBlock *BB,
    unsigned Reloc, intptr_t JTBase) {
  MCE.addRelocation(
      MachineRelocation::getBB(MCE.getCurrentPCOffset(), Reloc, BB, JTBase));
}

void MipsCodeEmitter::emitExternalSymbolAddress(const char *ES,
    unsigned Reloc) {
  MCE.addRelocation(
      MachineRelocation::getExtSym(MCE.getCurrentPCOffset(), Reloc, ES, 0, 0,
          false));
}

void MipsCodeEmitter::emitJumpTableAddress(unsigned JTIndex, unsigned Reloc)
    const {
  MCE.addRelocation(
      MachineRelocation::getJumpTable(MCE.getCurrentPCOffset(), Reloc, JTIndex,
          0, false));
}

void MipsCodeEmitter::emitConstPoolAddress(unsigned CPI, unsigned Reloc) {
  MCE.addRelocation(
      MachineRelocation::getConstPool
        (MCE.getCurrentPCOffset(), Reloc, CPI, 0));
}

/// createMipsJITCodeEmitterPass - Return a pass that emits the collected Mips
/// code to the specified MCE object.
FunctionPass *llvm::createMipsJITCodeEmitterPass(MipsTargetMachine &TM,
    JITCodeEmitter &JCE) {
  return new MipsCodeEmitter(TM, JCE);
}

char MipsCodeEmitter::ID = 10;

bool MipsCodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  JTI = ((MipsTargetMachine&) MF.getTarget()).getJITInfo();
  II = ((const MipsTargetMachine&) MF.getTarget()).getInstrInfo();
  TD = ((const MipsTargetMachine&) MF.getTarget()).getTargetData();
  Subtarget = &TM.getSubtarget<MipsSubtarget> ();
  MCPEs = &MF.getConstantPool()->getConstants();
  MJTEs = 0;
  if (MF.getJumpTableInfo()) MJTEs = &MF.getJumpTableInfo()->getJumpTables();
  JTI->Initialize(MF, IsPIC);
  MCE.setModuleInfo(&getAnalysis<MachineModuleInfo> ());

  do {
    DEBUG(errs() << "JITTing function '"
        << MF.getFunction()->getName() << "'\n");
    MCE.startFunction(MF);

    for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
        MBB != E; ++MBB){
      MCE.StartMachineBasicBlock(MBB);
      for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
          I != E; ++I)
        emitInstruction(*I);
    }
  } while (MCE.finishFunction(MF));

  return false;
}

void MipsCodeEmitter::emitInstruction(const MachineInstr &MI) {}

unsigned MipsCodeEmitter::getOperandValue(const MachineOperand &MO,
    unsigned relocType) {
  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    return MO.getImm();
  case MachineOperand::MO_GlobalAddress:
    emitGlobalAddress(MO.getGlobal(), relocType, false);
    return 0;
  case MachineOperand::MO_ExternalSymbol:
    emitExternalSymbolAddress(MO.getSymbolName(), relocType);
    return 0;
  case MachineOperand::MO_MachineBasicBlock:
    emitMachineBasicBlock(MO.getMBB(), relocType, MCE.getCurrentPCValue());
    return 0;
  case MachineOperand::MO_Register:
    return MipsRegisterInfo::getRegisterNumbering(MO.getReg());
  case MachineOperand::MO_JumpTableIndex:
    emitJumpTableAddress(MO.getIndex(), relocType);
    return 0;
  case MachineOperand::MO_ConstantPoolIndex:
    emitConstPoolAddress(MO.getIndex(), relocType);
    return 0;
  default: return 0;
  }
}

