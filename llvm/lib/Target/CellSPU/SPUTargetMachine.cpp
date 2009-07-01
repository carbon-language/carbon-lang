//===-- SPUTargetMachine.cpp - Define TargetMachine for Cell SPU ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the Cell SPU target.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPURegisterNames.h"
#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"

using namespace llvm;

namespace {
  // Register the targets
  RegisterTarget<SPUTargetMachine>
  CELLSPU("cellspu", "STI CBEA Cell SPU [experimental]");
}

// No assembler printer by default
SPUTargetMachine::AsmPrinterCtorFn SPUTargetMachine::AsmPrinterCtor = 0;

// Force static initialization.
extern "C" void LLVMInitializeCellSPUTarget() { }

const std::pair<unsigned, int> *
SPUFrameInfo::getCalleeSaveSpillSlots(unsigned &NumEntries) const {
  NumEntries = 1;
  return &LR[0];
}

const TargetAsmInfo *
SPUTargetMachine::createTargetAsmInfo() const
{
  return new SPULinuxTargetAsmInfo(*this);
}

unsigned
SPUTargetMachine::getModuleMatchQuality(const Module &M)
{
  // We strongly match "spu-*" or "cellspu-*".
  std::string TT = M.getTargetTriple();
  if ((TT.size() == 3 && std::string(TT.begin(), TT.begin()+3) == "spu")
      || (TT.size() == 7 && std::string(TT.begin(), TT.begin()+7) == "cellspu")
      || (TT.size() >= 4 && std::string(TT.begin(), TT.begin()+4) == "spu-")
      || (TT.size() >= 8 && std::string(TT.begin(), TT.begin()+8) == "cellspu-"))
    return 20;
  
  return 0;                     // No match at all...
}

SPUTargetMachine::SPUTargetMachine(const Module &M, const std::string &FS)
  : Subtarget(*this, M, FS),
    DataLayout(Subtarget.getTargetDataString()),
    InstrInfo(*this),
    FrameInfo(*this),
    TLInfo(*this),
    InstrItins(Subtarget.getInstrItineraryData())
{
  // For the time being, use static relocations, since there's really no
  // support for PIC yet.
  setRelocationModel(Reloc::Static);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool
SPUTargetMachine::addInstSelector(PassManagerBase &PM,
                                  CodeGenOpt::Level OptLevel)
{
  // Install an instruction selector.
  PM.add(createSPUISelDag(*this));
  return false;
}

bool SPUTargetMachine::addAssemblyEmitter(PassManagerBase &PM,
                                          CodeGenOpt::Level OptLevel,
                                          bool Verbose,
                                          raw_ostream &Out) {
  // Output assembly language.
  assert(AsmPrinterCtor && "AsmPrinter was not linked in");
  if (AsmPrinterCtor)
    PM.add(AsmPrinterCtor(Out, *this, Verbose));
  return false;
}
