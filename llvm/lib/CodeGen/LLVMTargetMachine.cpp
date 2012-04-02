//===-- LLVMTargetMachine.cpp - Implement the LLVMTargetMachine class -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMTargetMachine class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

// Enable or disable FastISel. Both options are needed, because
// FastISel is enabled by default with -fast, and we wish to be
// able to enable or disable fast-isel independently from -O0.
static cl::opt<cl::boolOrDefault>
EnableFastISelOption("fast-isel", cl::Hidden,
  cl::desc("Enable the \"fast\" instruction selector"));

static cl::opt<bool> ShowMCEncoding("show-mc-encoding", cl::Hidden,
    cl::desc("Show encoding in .s output"));
static cl::opt<bool> ShowMCInst("show-mc-inst", cl::Hidden,
    cl::desc("Show instruction structure in .s output"));

static cl::opt<cl::boolOrDefault>
AsmVerbose("asm-verbose", cl::desc("Add comments to directives."),
           cl::init(cl::BOU_UNSET));

static bool getVerboseAsm() {
  switch (AsmVerbose) {
  case cl::BOU_UNSET: return TargetMachine::getAsmVerbosityDefault();
  case cl::BOU_TRUE:  return true;
  case cl::BOU_FALSE: return false;
  }
  llvm_unreachable("Invalid verbose asm state");
}

LLVMTargetMachine::LLVMTargetMachine(const Target &T, StringRef Triple,
                                     StringRef CPU, StringRef FS,
                                     TargetOptions Options,
                                     Reloc::Model RM, CodeModel::Model CM,
                                     CodeGenOpt::Level OL)
  : TargetMachine(T, Triple, CPU, FS, Options) {
  CodeGenInfo = T.createMCCodeGenInfo(Triple, RM, CM, OL);
  AsmInfo = T.createMCAsmInfo(Triple);
  // TargetSelect.h moved to a different directory between LLVM 2.9 and 3.0,
  // and if the old one gets included then MCAsmInfo will be NULL and
  // we'll crash later.
  // Provide the user with a useful error message about what's wrong.
  assert(AsmInfo && "MCAsmInfo not initialized."
         "Make sure you include the correct TargetSelect.h"
         "and that InitializeAllTargetMCs() is being invoked!");
}

/// Turn exception handling constructs into something the code generators can
/// handle.
static void addPassesToHandleExceptions(TargetMachine *TM,
                                        PassManagerBase &PM) {
  switch (TM->getMCAsmInfo()->getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    PM.add(createSjLjEHPreparePass(TM->getTargetLowering()));
    // FALLTHROUGH
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::Win64:
    PM.add(createDwarfEHPass(TM));
    break;
  case ExceptionHandling::None:
    PM.add(createLowerInvokePass(TM->getTargetLowering()));

    // The lower invoke pass may create unreachable code. Remove it.
    PM.add(createUnreachableBlockEliminationPass());
    break;
  }
}

/// addPassesToX helper drives creation and initialization of TargetPassConfig.
static MCContext *addPassesToGenerateCode(LLVMTargetMachine *TM,
                                          PassManagerBase &PM,
                                          bool DisableVerify) {
  // Targets may override createPassConfig to provide a target-specific sublass.
  TargetPassConfig *PassConfig = TM->createPassConfig(PM);

  // Set PassConfig options provided by TargetMachine.
  PassConfig->setDisableVerify(DisableVerify);

  PM.add(PassConfig);

  PassConfig->addIRPasses();

  addPassesToHandleExceptions(TM, PM);

  PassConfig->addISelPrepare();

  // Install a MachineModuleInfo class, which is an immutable pass that holds
  // all the per-module stuff we're generating, including MCContext.
  MachineModuleInfo *MMI =
    new MachineModuleInfo(*TM->getMCAsmInfo(), *TM->getRegisterInfo(),
                          &TM->getTargetLowering()->getObjFileLowering());
  PM.add(MMI);
  MCContext *Context = &MMI->getContext(); // Return the MCContext by-ref.

  // Set up a MachineFunction for the rest of CodeGen to work on.
  PM.add(new MachineFunctionAnalysis(*TM));

  // Enable FastISel with -fast, but allow that to be overridden.
  if (EnableFastISelOption == cl::BOU_TRUE ||
      (TM->getOptLevel() == CodeGenOpt::None &&
       EnableFastISelOption != cl::BOU_FALSE))
    TM->setFastISel(true);

  // Ask the target for an isel.
  if (PassConfig->addInstSelector())
    return NULL;

  PassConfig->addMachinePasses();

  PassConfig->setInitialized();

  return Context;
}

bool LLVMTargetMachine::addPassesToEmitFile(PassManagerBase &PM,
                                            formatted_raw_ostream &Out,
                                            CodeGenFileType FileType,
                                            bool DisableVerify) {
  // Add common CodeGen passes.
  MCContext *Context = addPassesToGenerateCode(this, PM, DisableVerify);
  if (!Context)
    return true;

  if (hasMCSaveTempLabels())
    Context->setAllowTemporaryLabels(false);

  const MCAsmInfo &MAI = *getMCAsmInfo();
  const MCSubtargetInfo &STI = getSubtarget<MCSubtargetInfo>();
  OwningPtr<MCStreamer> AsmStreamer;

  switch (FileType) {
  case CGFT_AssemblyFile: {
    MCInstPrinter *InstPrinter =
      getTarget().createMCInstPrinter(MAI.getAssemblerDialect(), MAI,
                                      *getInstrInfo(),
                                      Context->getRegisterInfo(), STI);

    // Create a code emitter if asked to show the encoding.
    MCCodeEmitter *MCE = 0;
    MCAsmBackend *MAB = 0;
    if (ShowMCEncoding) {
      const MCSubtargetInfo &STI = getSubtarget<MCSubtargetInfo>();
      MCE = getTarget().createMCCodeEmitter(*getInstrInfo(), STI, *Context);
      MAB = getTarget().createMCAsmBackend(getTargetTriple());
    }

    MCStreamer *S = getTarget().createAsmStreamer(*Context, Out,
                                                  getVerboseAsm(),
                                                  hasMCUseLoc(),
                                                  hasMCUseCFI(),
                                                  hasMCUseDwarfDirectory(),
                                                  InstPrinter,
                                                  MCE, MAB,
                                                  ShowMCInst);
    AsmStreamer.reset(S);
    break;
  }
  case CGFT_ObjectFile: {
    // Create the code emitter for the target if it exists.  If not, .o file
    // emission fails.
    MCCodeEmitter *MCE = getTarget().createMCCodeEmitter(*getInstrInfo(), STI,
                                                         *Context);
    MCAsmBackend *MAB = getTarget().createMCAsmBackend(getTargetTriple());
    if (MCE == 0 || MAB == 0)
      return true;

    AsmStreamer.reset(getTarget().createMCObjectStreamer(getTargetTriple(),
                                                         *Context, *MAB, Out,
                                                         MCE, hasMCRelaxAll(),
                                                         hasMCNoExecStack()));
    AsmStreamer.get()->InitSections();
    break;
  }
  case CGFT_Null:
    // The Null output is intended for use for performance analysis and testing,
    // not real users.
    AsmStreamer.reset(createNullStreamer(*Context));
    break;
  }

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer = getTarget().createAsmPrinter(*this, *AsmStreamer);
  if (Printer == 0)
    return true;

  // If successful, createAsmPrinter took ownership of AsmStreamer.
  AsmStreamer.take();

  PM.add(Printer);

  PM.add(createGCInfoDeleter());
  return false;
}

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to
/// get machine code emitted.  This uses a JITCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool LLVMTargetMachine::addPassesToEmitMachineCode(PassManagerBase &PM,
                                                   JITCodeEmitter &JCE,
                                                   bool DisableVerify) {
  // Add common CodeGen passes.
  MCContext *Context = addPassesToGenerateCode(this, PM, DisableVerify);
  if (!Context)
    return true;

  addCodeEmitter(PM, JCE);
  PM.add(createGCInfoDeleter());

  return false; // success!
}

/// addPassesToEmitMC - Add passes to the specified pass manager to get
/// machine code emitted with the MCJIT. This method returns true if machine
/// code is not supported. It fills the MCContext Ctx pointer which can be
/// used to build custom MCStreamer.
///
bool LLVMTargetMachine::addPassesToEmitMC(PassManagerBase &PM,
                                          MCContext *&Ctx,
                                          raw_ostream &Out,
                                          bool DisableVerify) {
  // Add common CodeGen passes.
  Ctx = addPassesToGenerateCode(this, PM, DisableVerify);
  if (!Ctx)
    return true;

  if (hasMCSaveTempLabels())
    Ctx->setAllowTemporaryLabels(false);

  // Create the code emitter for the target if it exists.  If not, .o file
  // emission fails.
  const MCSubtargetInfo &STI = getSubtarget<MCSubtargetInfo>();
  MCCodeEmitter *MCE = getTarget().createMCCodeEmitter(*getInstrInfo(),STI,
                                                       *Ctx);
  MCAsmBackend *MAB = getTarget().createMCAsmBackend(getTargetTriple());
  if (MCE == 0 || MAB == 0)
    return true;

  OwningPtr<MCStreamer> AsmStreamer;
  AsmStreamer.reset(getTarget().createMCObjectStreamer(getTargetTriple(), *Ctx,
                                                       *MAB, Out, MCE,
                                                       hasMCRelaxAll(),
                                                       hasMCNoExecStack()));
  AsmStreamer.get()->InitSections();

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer = getTarget().createAsmPrinter(*this, *AsmStreamer);
  if (Printer == 0)
    return true;

  // If successful, createAsmPrinter took ownership of AsmStreamer.
  AsmStreamer.take();

  PM.add(Printer);

  return false; // success!
}
