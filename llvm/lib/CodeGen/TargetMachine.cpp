//===-- TargetMachine.cpp - Implement the TargetMachine class -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Implements the TargetMachine class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"

#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

TargetMachine::TargetMachine(const Target &T, StringRef DataLayoutString,
                             const Triple &TT, StringRef CPU, StringRef FS,
                             const TargetOptions &Options, Reloc::Model RM,
                             CodeModel::Model CM, CodeGenOpt::Level OL)
    : TheTarget(T), DL(DataLayoutString), TargetTriple(TT), TargetCPU(CPU),
      TargetFS(FS), RM(RM), CMModel(CM), OptLevel(OL), AsmInfo(nullptr),
      MRI(nullptr), MII(nullptr), STI(nullptr), RequireStructuredCFG(false),
      DefaultOptions(Options), Options(Options) {
}

TargetMachine::~TargetMachine() {
  delete AsmInfo;
  delete MRI;
  delete MII;
  delete STI;
}

bool TargetMachine::isPositionIndependent() const {
  return getRelocationModel() == Reloc::PIC_;
}

// FIXME: This function needs to go away for a number of reasons:
// a) global state on the TargetMachine is terrible in general,
// b) these target options should be passed only on the function
//    and not on the TargetMachine (via TargetOptions) at all.
void TargetMachine::resetTargetOptions(const Function &F) const {
#define RESET_OPTION(X, Y)                                                     \
  do {                                                                         \
    if (F.hasFnAttribute(Y))                                                   \
      Options.X = (F.getFnAttribute(Y).getValueAsString() == "true");          \
    else                                                                       \
      Options.X = DefaultOptions.X;                                            \
  } while (0)

  RESET_OPTION(UnsafeFPMath, "unsafe-fp-math");
  RESET_OPTION(NoInfsFPMath, "no-infs-fp-math");
  RESET_OPTION(NoNaNsFPMath, "no-nans-fp-math");
  RESET_OPTION(NoSignedZerosFPMath, "no-signed-zeros-fp-math");
  RESET_OPTION(NoTrappingFPMath, "no-trapping-math");

  StringRef Denormal =
    F.getFnAttribute("denormal-fp-math").getValueAsString();
  if (Denormal == "ieee")
    Options.FPDenormalMode = FPDenormal::IEEE;
  else if (Denormal == "preserve-sign")
    Options.FPDenormalMode = FPDenormal::PreserveSign;
  else if (Denormal == "positive-zero")
    Options.FPDenormalMode = FPDenormal::PositiveZero;
  else
    Options.FPDenormalMode = DefaultOptions.FPDenormalMode;
}

Reloc::Model TargetMachine::getRelocationModel() const { return RM; }

CodeModel::Model TargetMachine::getCodeModel() const { return CMModel; }

/// Get the IR-specified TLS model for Var.
static TLSModel::Model getSelectedTLSModel(const GlobalValue *GV) {
  switch (GV->getThreadLocalMode()) {
  case GlobalVariable::NotThreadLocal:
    llvm_unreachable("getSelectedTLSModel for non-TLS variable");
    break;
  case GlobalVariable::GeneralDynamicTLSModel:
    return TLSModel::GeneralDynamic;
  case GlobalVariable::LocalDynamicTLSModel:
    return TLSModel::LocalDynamic;
  case GlobalVariable::InitialExecTLSModel:
    return TLSModel::InitialExec;
  case GlobalVariable::LocalExecTLSModel:
    return TLSModel::LocalExec;
  }
  llvm_unreachable("invalid TLS model");
}

bool TargetMachine::shouldAssumeDSOLocal(const Module &M,
                                         const GlobalValue *GV) const {
  Reloc::Model RM = getRelocationModel();
  const Triple &TT = getTargetTriple();

  // DLLImport explicitly marks the GV as external.
  if (GV && GV->hasDLLImportStorageClass())
    return false;

  // Every other GV is local on COFF.
  // Make an exception for windows OS in the triple: Some firmwares builds use
  // *-win32-macho triples. This (accidentally?) produced windows relocations
  // without GOT tables in older clang versions; Keep this behaviour.
  if (TT.isOSBinFormatCOFF() || (TT.isOSWindows() && TT.isOSBinFormatMachO()))
    return true;

  if (GV && (GV->hasLocalLinkage() || !GV->hasDefaultVisibility()))
    return true;

  if (TT.isOSBinFormatMachO()) {
    if (RM == Reloc::Static)
      return true;
    return GV && GV->isStrongDefinitionForLinker();
  }

  assert(TT.isOSBinFormatELF());
  assert(RM != Reloc::DynamicNoPIC);

  bool IsExecutable =
      RM == Reloc::Static || M.getPIELevel() != PIELevel::Default;
  if (IsExecutable) {
    // If the symbol is defined, it cannot be preempted.
    if (GV && !GV->isDeclarationForLinker())
      return true;

    bool IsTLS = GV && GV->isThreadLocal();
    bool IsAccessViaCopyRelocs = Options.MCOptions.MCPIECopyRelocations && GV &&
                                 isa<GlobalVariable>(GV) &&
                                 !GV->hasExternalWeakLinkage();
    Triple::ArchType Arch = TT.getArch();
    bool IsPPC =
        Arch == Triple::ppc || Arch == Triple::ppc64 || Arch == Triple::ppc64le;
    // Check if we can use copy relocations. PowerPC has no copy relocations.
    if (!IsTLS && !IsPPC && (RM == Reloc::Static || IsAccessViaCopyRelocs))
      return true;
  }

  // ELF supports preemption of other symbols.
  return false;
}

TLSModel::Model TargetMachine::getTLSModel(const GlobalValue *GV) const {
  bool IsPIE = GV->getParent()->getPIELevel() != PIELevel::Default;
  Reloc::Model RM = getRelocationModel();
  bool IsSharedLibrary = RM == Reloc::PIC_ && !IsPIE;
  bool IsLocal = shouldAssumeDSOLocal(*GV->getParent(), GV);

  TLSModel::Model Model;
  if (IsSharedLibrary) {
    if (IsLocal)
      Model = TLSModel::LocalDynamic;
    else
      Model = TLSModel::GeneralDynamic;
  } else {
    if (IsLocal)
      Model = TLSModel::LocalExec;
    else
      Model = TLSModel::InitialExec;
  }

  // If the user specified a more specific model, use that.
  TLSModel::Model SelectedModel = getSelectedTLSModel(GV);
  if (SelectedModel > Model)
    return SelectedModel;

  return Model;
}

CodeGenOpt::Level TargetMachine::getOptLevel() const { return OptLevel; }

void TargetMachine::setOptLevel(CodeGenOpt::Level Level) { OptLevel = Level; }

void TargetMachine::getNameWithPrefix(SmallVectorImpl<char> &Name,
                                      const GlobalValue *GV, Mangler &Mang,
                                      bool MayAlwaysUsePrivate) const {
  if (MayAlwaysUsePrivate || !GV->hasPrivateLinkage()) {
    // Simple case: If GV is not private, it is not important to find out if
    // private labels are legal in this case or not.
    Mang.getNameWithPrefix(Name, GV, false);
    return;
  }
  const TargetLoweringObjectFile *TLOF = getObjFileLowering();
  TLOF->getNameWithPrefix(Name, GV, *this);
}

MCSymbol *TargetMachine::getSymbol(const GlobalValue *GV) const {
  const TargetLoweringObjectFile *TLOF = getObjFileLowering();
  SmallString<128> NameStr;
  getNameWithPrefix(NameStr, GV, TLOF->getMangler());
  return TLOF->getContext().getOrCreateSymbol(NameStr);
}

void TargetMachine::initAsmInfo() {
  MRI = TheTarget.createMCRegInfo(getTargetTriple().str());
  MII = TheTarget.createMCInstrInfo();
  // FIXME: Having an MCSubtargetInfo on the target machine is a hack due
  // to some backends having subtarget feature dependent module level
  // code generation. This is similar to the hack in the AsmPrinter for
  // module level assembly etc.
  STI = TheTarget.createMCSubtargetInfo(getTargetTriple().str(), getTargetCPU(),
                                        getTargetFeatureString());

  MCAsmInfo *TmpAsmInfo =
      TheTarget.createMCAsmInfo(*MRI, getTargetTriple().str());
  // TargetSelect.h moved to a different directory between LLVM 2.9 and 3.0,
  // and if the old one gets included then MCAsmInfo will be NULL and
  // we'll crash later.
  // Provide the user with a useful error message about what's wrong.
  assert(TmpAsmInfo && "MCAsmInfo not initialized. "
         "Make sure you include the correct TargetSelect.h"
         "and that InitializeAllTargetMCs() is being invoked!");

  if (Options.DisableIntegratedAS)
    TmpAsmInfo->setUseIntegratedAssembler(false);

  TmpAsmInfo->setPreserveAsmComments(Options.MCOptions.PreserveAsmComments);

  TmpAsmInfo->setCompressDebugSections(Options.CompressDebugSections);

  TmpAsmInfo->setRelaxELFRelocations(Options.RelaxELFRelocations);

  if (Options.ExceptionModel != ExceptionHandling::None)
    TmpAsmInfo->setExceptionsType(Options.ExceptionModel);

  AsmInfo = TmpAsmInfo;
}

TargetIRAnalysis TargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(BasicTTIImpl(this, F));
  });
}

/// addPassesToX helper drives creation and initialization of TargetPassConfig.
static MCContext *
addPassesToGenerateCode(TargetMachine *TM, PassManagerBase &PM,
                        bool DisableVerify, bool &WillCompleteCodeGenPipeline,
                        raw_pwrite_stream &Out, MachineModuleInfo *MMI) {
  // Targets may override createPassConfig to provide a target-specific
  // subclass.
  TargetPassConfig *PassConfig = TM->createPassConfig(PM);
  // Set PassConfig options provided by TargetMachine.
  PassConfig->setDisableVerify(DisableVerify);
  WillCompleteCodeGenPipeline = PassConfig->willCompleteCodeGenPipeline();
  PM.add(PassConfig);
  if (!MMI)
    MMI = new MachineModuleInfo(TM);
  PM.add(MMI);

  if (PassConfig->addISelPasses())
    return nullptr;
  PassConfig->addMachinePasses();
  PassConfig->setInitialized();
  if (!WillCompleteCodeGenPipeline)
    PM.add(createPrintMIRPass(Out));

  return &MMI->getContext();
}

bool TargetMachine::addAsmPrinter(PassManagerBase &PM,
    raw_pwrite_stream &Out, CodeGenFileType FileType,
    MCContext &Context) {
  if (Options.MCOptions.MCSaveTempLabels)
    Context.setAllowTemporaryLabels(false);

  const MCSubtargetInfo &STI = *getMCSubtargetInfo();
  const MCAsmInfo &MAI = *getMCAsmInfo();
  const MCRegisterInfo &MRI = *getMCRegisterInfo();
  const MCInstrInfo &MII = *getMCInstrInfo();

  std::unique_ptr<MCStreamer> AsmStreamer;

  switch (FileType) {
  case CGFT_AssemblyFile: {
    MCInstPrinter *InstPrinter = getTarget().createMCInstPrinter(
        getTargetTriple(), MAI.getAssemblerDialect(), MAI, MII, MRI);

    // Create a code emitter if asked to show the encoding.
    MCCodeEmitter *MCE = nullptr;
    if (Options.MCOptions.ShowMCEncoding)
      MCE = getTarget().createMCCodeEmitter(MII, MRI, Context);

    MCAsmBackend *MAB =
        getTarget().createMCAsmBackend(MRI, getTargetTriple().str(), TargetCPU,
                                       Options.MCOptions);
    auto FOut = llvm::make_unique<formatted_raw_ostream>(Out);
    MCStreamer *S = getTarget().createAsmStreamer(
        Context, std::move(FOut), Options.MCOptions.AsmVerbose,
        Options.MCOptions.MCUseDwarfDirectory, InstPrinter, MCE, MAB,
        Options.MCOptions.ShowMCInst);
    AsmStreamer.reset(S);
    break;
  }
  case CGFT_ObjectFile: {
    // Create the code emitter for the target if it exists.  If not, .o file
    // emission fails.
    MCCodeEmitter *MCE = getTarget().createMCCodeEmitter(MII, MRI, Context);
    MCAsmBackend *MAB =
        getTarget().createMCAsmBackend(MRI, getTargetTriple().str(), TargetCPU,
                                       Options.MCOptions);
    if (!MCE || !MAB)
      return true;

    // Don't waste memory on names of temp labels.
    Context.setUseNamesOnTempLabels(false);

    Triple T(getTargetTriple().str());
    AsmStreamer.reset(getTarget().createMCObjectStreamer(
        T, Context, std::unique_ptr<MCAsmBackend>(MAB), Out,
        std::unique_ptr<MCCodeEmitter>(MCE), STI, Options.MCOptions.MCRelaxAll,
        Options.MCOptions.MCIncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ true));
    break;
  }
  case CGFT_Null:
    // The Null output is intended for use for performance analysis and testing,
    // not real users.
    AsmStreamer.reset(getTarget().createNullStreamer(Context));
    break;
  }

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer =
      getTarget().createAsmPrinter(*this, std::move(AsmStreamer));
  if (!Printer)
    return true;

  PM.add(Printer);
  return false;
}

bool TargetMachine::addPassesToEmitFile(PassManagerBase &PM,
                                        raw_pwrite_stream &Out,
                                        CodeGenFileType FileType,
                                        bool DisableVerify,
                                        MachineModuleInfo *MMI) {
  // Add common CodeGen passes.
  bool WillCompleteCodeGenPipeline = true;
  MCContext *Context = addPassesToGenerateCode(
      this, PM, DisableVerify, WillCompleteCodeGenPipeline, Out, MMI);
  if (!Context)
    return true;

  if (WillCompleteCodeGenPipeline && addAsmPrinter(PM, Out, FileType, *Context))
    return true;

  PM.add(createFreeMachineFunctionPass());
  return false;
}

bool TargetMachine::addPassesToEmitMC(PassManagerBase &PM, MCContext *&Ctx,
                                      raw_pwrite_stream &Out,
                                      bool DisableVerify) {
  // Add common CodeGen passes.
  bool WillCompleteCodeGenPipeline = true;
  Ctx = addPassesToGenerateCode(this, PM, DisableVerify,
                                WillCompleteCodeGenPipeline, Out,
                                /*MachineModuleInfo*/ nullptr);
  if (!Ctx)
    return true;
  assert(WillCompleteCodeGenPipeline && "CodeGen pipeline has been altered");

  if (Options.MCOptions.MCSaveTempLabels)
    Ctx->setAllowTemporaryLabels(false);

  // Create the code emitter for the target if it exists.  If not, .o file
  // emission fails.
  const MCRegisterInfo &MRI = *getMCRegisterInfo();
  MCCodeEmitter *MCE =
      getTarget().createMCCodeEmitter(*getMCInstrInfo(), MRI, *Ctx);
  MCAsmBackend *MAB =
      getTarget().createMCAsmBackend(MRI, getTargetTriple().str(), TargetCPU,
                                     Options.MCOptions);
  if (!MCE || !MAB)
    return true;

  const Triple &T = getTargetTriple();
  const MCSubtargetInfo &STI = *getMCSubtargetInfo();
  std::unique_ptr<MCStreamer> AsmStreamer(getTarget().createMCObjectStreamer(
      T, *Ctx, std::unique_ptr<MCAsmBackend>(MAB), Out,
      std::unique_ptr<MCCodeEmitter>(MCE), STI, Options.MCOptions.MCRelaxAll,
      Options.MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ true));

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer =
      getTarget().createAsmPrinter(*this, std::move(AsmStreamer));
  if (!Printer)
    return true;

  PM.add(Printer);
  PM.add(createFreeMachineFunctionPass());

  return false; // success!
}
