//===-- MachOWriter.cpp - Target-independent Mach-O Writer code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent Mach-O writer.  This file writes
// out the Mach-O file in the following order:
//
//  #1 FatHeader (universal-only)
//  #2 FatArch (universal-only, 1 per universal arch)
//  Per arch:
//    #3 Header
//    #4 Load Commands
//    #5 Sections
//    #6 Relocations
//    #7 Symbols
//    #8 Strings
//
//===----------------------------------------------------------------------===//

#include "MachOWriter.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
using namespace llvm;

namespace llvm { 
MachineFunctionPass *createMachOWriter(formatted_raw_ostream &O,
                                       TargetMachine &TM,
                                       const MCAsmInfo *T, 
                                       MCCodeEmitter *MCE) { 
  return new MachOWriter(O, TM, T, MCE);
}
}

//===----------------------------------------------------------------------===//
//                          MachOWriter Implementation
//===----------------------------------------------------------------------===//

char MachOWriter::ID = 0;

MachOWriter::MachOWriter(formatted_raw_ostream &o, TargetMachine &tm,
                         const MCAsmInfo *T, MCCodeEmitter *MCE)
  : MachineFunctionPass(&ID), O(o), TM(tm), MAI(T), MCCE(MCE),
    OutContext(*new MCContext()),
    OutStreamer(*createMachOStreamer(OutContext, O, MCCE)) { 
}

MachOWriter::~MachOWriter() {
  delete &OutStreamer;
  delete &OutContext;
  delete MCCE;
}

bool MachOWriter::doInitialization(Module &M) {
  // Initialize TargetLoweringObjectFile.
  TM.getTargetLowering()->getObjFileLowering().Initialize(OutContext, TM);

  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the Mach-O file to 'O'.
bool MachOWriter::doFinalization(Module &M) {
  OutStreamer.Finish();
  return false;
}

bool MachOWriter::runOnMachineFunction(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  TargetLoweringObjectFile &TLOF = TM.getTargetLowering()->getObjFileLowering();
  const MCSection *S = TLOF.SectionForGlobal(F, Mang, TM);
  OutStreamer.SwitchSection(S);

  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block.
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MI = II;
      MCInst OutMI;
      OutMI.setOpcode(MI->getOpcode());

      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        MCOperand MCOp;

        switch (MO.getType()) {
          default:
            MI->dump();
            llvm_unreachable("unknown operand type");
          case MachineOperand::MO_Register:
            // Ignore all implicit register operands.
            if (MO.isImplicit()) continue;
            MCOp = MCOperand::CreateReg(MO.getReg());
            break;
          case MachineOperand::MO_Immediate:
            MCOp = MCOperand::CreateImm(MO.getImm());
            break;
        }
        OutMI.addOperand(MCOp);
      }
      
      OutStreamer.EmitInstruction(OutMI);
    }
  }

  return false;
}
