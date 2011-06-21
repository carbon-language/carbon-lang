//===--- BackendUtil.cpp - LLVM Backend Utilities -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/BackendUtil.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/PassManagerBuilder.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Transforms/Instrumentation.h"
using namespace clang;
using namespace llvm;

namespace {

class EmitAssemblyHelper {
  Diagnostic &Diags;
  const CodeGenOptions &CodeGenOpts;
  const TargetOptions &TargetOpts;
  Module *TheModule;

  Timer CodeGenerationTime;

  mutable PassManager *CodeGenPasses;
  mutable PassManager *PerModulePasses;
  mutable FunctionPassManager *PerFunctionPasses;

private:
  PassManager *getCodeGenPasses() const {
    if (!CodeGenPasses) {
      CodeGenPasses = new PassManager();
      CodeGenPasses->add(new TargetData(TheModule));
    }
    return CodeGenPasses;
  }

  PassManager *getPerModulePasses() const {
    if (!PerModulePasses) {
      PerModulePasses = new PassManager();
      PerModulePasses->add(new TargetData(TheModule));
    }
    return PerModulePasses;
  }

  FunctionPassManager *getPerFunctionPasses() const {
    if (!PerFunctionPasses) {
      PerFunctionPasses = new FunctionPassManager(TheModule);
      PerFunctionPasses->add(new TargetData(TheModule));
    }
    return PerFunctionPasses;
  }

  void CreatePasses();

  /// AddEmitPasses - Add passes necessary to emit assembly or LLVM IR.
  ///
  /// \return True on success.
  bool AddEmitPasses(BackendAction Action, formatted_raw_ostream &OS);

public:
  EmitAssemblyHelper(Diagnostic &_Diags,
                     const CodeGenOptions &CGOpts, const TargetOptions &TOpts,
                     Module *M)
    : Diags(_Diags), CodeGenOpts(CGOpts), TargetOpts(TOpts),
      TheModule(M), CodeGenerationTime("Code Generation Time"),
      CodeGenPasses(0), PerModulePasses(0), PerFunctionPasses(0) {}

  ~EmitAssemblyHelper() {
    delete CodeGenPasses;
    delete PerModulePasses;
    delete PerFunctionPasses;
  }

  void EmitAssembly(BackendAction Action, raw_ostream *OS);
};

}

void EmitAssemblyHelper::CreatePasses() {
  unsigned OptLevel = CodeGenOpts.OptimizationLevel;
  CodeGenOptions::InliningMethod Inlining = CodeGenOpts.Inlining;

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (CodeGenOpts.DisableLLVMOpts) {
    OptLevel = 0;
    Inlining = CodeGenOpts.NoInlining;
  }
  
  PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptLevel;
  PMBuilder.SizeLevel = CodeGenOpts.OptimizeSize;

  PMBuilder.DisableSimplifyLibCalls = !CodeGenOpts.SimplifyLibCalls;
  PMBuilder.DisableUnitAtATime = !CodeGenOpts.UnitAtATime;
  PMBuilder.DisableUnrollLoops = !CodeGenOpts.UnrollLoops;
  
  // Figure out TargetLibraryInfo.
  Triple TargetTriple(TheModule->getTargetTriple());
  PMBuilder.LibraryInfo = new TargetLibraryInfo(TargetTriple);
  if (!CodeGenOpts.SimplifyLibCalls)
    PMBuilder.LibraryInfo->disableAllFunctions();
  
  switch (Inlining) {
  case CodeGenOptions::NoInlining: break;
  case CodeGenOptions::NormalInlining: {
    // FIXME: Derive these constants in a principled fashion.
    unsigned Threshold = 225;
    if (CodeGenOpts.OptimizeSize == 1)      // -Os
      Threshold = 75;
    else if (CodeGenOpts.OptimizeSize == 2) // -Oz
      Threshold = 25;
    else if (OptLevel > 2)
      Threshold = 275;
    PMBuilder.Inliner = createFunctionInliningPass(Threshold);
    break;
  }
  case CodeGenOptions::OnlyAlwaysInlining:
    // Respect always_inline.
    PMBuilder.Inliner = createAlwaysInlinerPass();
    break;
  }

 
  // Set up the per-function pass manager.
  FunctionPassManager *FPM = getPerFunctionPasses();
  if (CodeGenOpts.VerifyModule)
    FPM->add(createVerifierPass());
  PMBuilder.populateFunctionPassManager(*FPM);

  // Set up the per-module pass manager.
  PassManager *MPM = getPerModulePasses();

  if (CodeGenOpts.EmitGcovArcs || CodeGenOpts.EmitGcovNotes) {
    MPM->add(createGCOVProfilerPass(CodeGenOpts.EmitGcovNotes,
                                    CodeGenOpts.EmitGcovArcs,
                                    TargetTriple.isMacOSX()));

    if (!CodeGenOpts.DebugInfo)
      MPM->add(createStripSymbolsPass(true));
  }
  
  
  PMBuilder.populateModulePassManager(*MPM);
}

bool EmitAssemblyHelper::AddEmitPasses(BackendAction Action,
                                       formatted_raw_ostream &OS) {
  // Create the TargetMachine for generating code.
  std::string Error;
  std::string Triple = TheModule->getTargetTriple();
  const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget) {
    Diags.Report(diag::err_fe_unable_to_create_target) << Error;
    return false;
  }

  // FIXME: Expose these capabilities via actual APIs!!!! Aside from just
  // being gross, this is also totally broken if we ever care about
  // concurrency.

  // Set frame pointer elimination mode.
  if (!CodeGenOpts.DisableFPElim) {
    llvm::NoFramePointerElim = false;
    llvm::NoFramePointerElimNonLeaf = false;
  } else if (CodeGenOpts.OmitLeafFramePointer) {
    llvm::NoFramePointerElim = false;
    llvm::NoFramePointerElimNonLeaf = true;
  } else {
    llvm::NoFramePointerElim = true;
    llvm::NoFramePointerElimNonLeaf = true;
  }

  // Set float ABI type.
  if (CodeGenOpts.FloatABI == "soft" || CodeGenOpts.FloatABI == "softfp")
    llvm::FloatABIType = llvm::FloatABI::Soft;
  else if (CodeGenOpts.FloatABI == "hard")
    llvm::FloatABIType = llvm::FloatABI::Hard;
  else {
    assert(CodeGenOpts.FloatABI.empty() && "Invalid float abi!");
    llvm::FloatABIType = llvm::FloatABI::Default;
  }

  llvm::LessPreciseFPMADOption = CodeGenOpts.LessPreciseFPMAD;
  llvm::NoInfsFPMath = CodeGenOpts.NoInfsFPMath;
  llvm::NoNaNsFPMath = CodeGenOpts.NoNaNsFPMath;
  NoZerosInBSS = CodeGenOpts.NoZeroInitializedInBSS;
  llvm::UnsafeFPMath = CodeGenOpts.UnsafeFPMath;
  llvm::UseSoftFloat = CodeGenOpts.SoftFloat;

  TargetMachine::setAsmVerbosityDefault(CodeGenOpts.AsmVerbose);

  TargetMachine::setFunctionSections(CodeGenOpts.FunctionSections);
  TargetMachine::setDataSections    (CodeGenOpts.DataSections);

  // FIXME: Parse this earlier.
  if (CodeGenOpts.RelocationModel == "static") {
    TargetMachine::setRelocationModel(llvm::Reloc::Static);
  } else if (CodeGenOpts.RelocationModel == "pic") {
    TargetMachine::setRelocationModel(llvm::Reloc::PIC_);
  } else {
    assert(CodeGenOpts.RelocationModel == "dynamic-no-pic" &&
           "Invalid PIC model!");
    TargetMachine::setRelocationModel(llvm::Reloc::DynamicNoPIC);
  }
  // FIXME: Parse this earlier.
  if (CodeGenOpts.CodeModel == "small") {
    TargetMachine::setCodeModel(llvm::CodeModel::Small);
  } else if (CodeGenOpts.CodeModel == "kernel") {
    TargetMachine::setCodeModel(llvm::CodeModel::Kernel);
  } else if (CodeGenOpts.CodeModel == "medium") {
    TargetMachine::setCodeModel(llvm::CodeModel::Medium);
  } else if (CodeGenOpts.CodeModel == "large") {
    TargetMachine::setCodeModel(llvm::CodeModel::Large);
  } else {
    assert(CodeGenOpts.CodeModel.empty() && "Invalid code model!");
    TargetMachine::setCodeModel(llvm::CodeModel::Default);
  }

  std::vector<const char *> BackendArgs;
  BackendArgs.push_back("clang"); // Fake program name.
  if (!CodeGenOpts.DebugPass.empty()) {
    BackendArgs.push_back("-debug-pass");
    BackendArgs.push_back(CodeGenOpts.DebugPass.c_str());
  }
  if (!CodeGenOpts.LimitFloatPrecision.empty()) {
    BackendArgs.push_back("-limit-float-precision");
    BackendArgs.push_back(CodeGenOpts.LimitFloatPrecision.c_str());
  }
  if (llvm::TimePassesIsEnabled)
    BackendArgs.push_back("-time-passes");
  for (unsigned i = 0, e = CodeGenOpts.BackendOptions.size(); i != e; ++i)
    BackendArgs.push_back(CodeGenOpts.BackendOptions[i].c_str());
  BackendArgs.push_back(0);
  llvm::cl::ParseCommandLineOptions(BackendArgs.size() - 1,
                                    const_cast<char **>(&BackendArgs[0]));

  std::string FeaturesStr;
  if (TargetOpts.CPU.size() || TargetOpts.Features.size()) {
    SubtargetFeatures Features;
    Features.setCPU(TargetOpts.CPU);
    for (std::vector<std::string>::const_iterator
           it = TargetOpts.Features.begin(),
           ie = TargetOpts.Features.end(); it != ie; ++it)
      Features.AddFeature(*it);
    FeaturesStr = Features.getString();
  }
  TargetMachine *TM = TheTarget->createTargetMachine(Triple, FeaturesStr);

  if (CodeGenOpts.RelaxAll)
    TM->setMCRelaxAll(true);
  if (CodeGenOpts.SaveTempLabels)
    TM->setMCSaveTempLabels(true);
  if (CodeGenOpts.NoDwarf2CFIAsm)
    TM->setMCUseCFI(false);
  if (CodeGenOpts.NoExecStack)
    TM->setMCNoExecStack(true);

  // Create the code generator passes.
  PassManager *PM = getCodeGenPasses();
  CodeGenOpt::Level OptLevel = CodeGenOpt::Default;

  switch (CodeGenOpts.OptimizationLevel) {
  default: break;
  case 0: OptLevel = CodeGenOpt::None; break;
  case 3: OptLevel = CodeGenOpt::Aggressive; break;
  }

  // Normal mode, emit a .s or .o file by running the code generator. Note,
  // this also adds codegenerator level optimization passes.
  TargetMachine::CodeGenFileType CGFT = TargetMachine::CGFT_AssemblyFile;
  if (Action == Backend_EmitObj)
    CGFT = TargetMachine::CGFT_ObjectFile;
  else if (Action == Backend_EmitMCNull)
    CGFT = TargetMachine::CGFT_Null;
  else
    assert(Action == Backend_EmitAssembly && "Invalid action!");
  if (TM->addPassesToEmitFile(*PM, OS, CGFT, OptLevel,
                              /*DisableVerify=*/!CodeGenOpts.VerifyModule)) {
    Diags.Report(diag::err_fe_unable_to_interface_with_target);
    return false;
  }

  return true;
}

void EmitAssemblyHelper::EmitAssembly(BackendAction Action, raw_ostream *OS) {
  TimeRegion Region(llvm::TimePassesIsEnabled ? &CodeGenerationTime : 0);
  llvm::formatted_raw_ostream FormattedOS;

  CreatePasses();
  switch (Action) {
  case Backend_EmitNothing:
    break;

  case Backend_EmitBC:
    getPerModulePasses()->add(createBitcodeWriterPass(*OS));
    break;

  case Backend_EmitLL:
    FormattedOS.setStream(*OS, formatted_raw_ostream::PRESERVE_STREAM);
    getPerModulePasses()->add(createPrintModulePass(&FormattedOS));
    break;

  default:
    FormattedOS.setStream(*OS, formatted_raw_ostream::PRESERVE_STREAM);
    if (!AddEmitPasses(Action, FormattedOS))
      return;
  }

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Run passes. For now we do all passes at once, but eventually we
  // would like to have the option of streaming code generation.

  if (PerFunctionPasses) {
    PrettyStackTraceString CrashInfo("Per-function optimization");

    PerFunctionPasses->doInitialization();
    for (Module::iterator I = TheModule->begin(),
           E = TheModule->end(); I != E; ++I)
      if (!I->isDeclaration())
        PerFunctionPasses->run(*I);
    PerFunctionPasses->doFinalization();
  }

  if (PerModulePasses) {
    PrettyStackTraceString CrashInfo("Per-module optimization passes");
    PerModulePasses->run(*TheModule);
  }

  if (CodeGenPasses) {
    PrettyStackTraceString CrashInfo("Code generation");
    CodeGenPasses->run(*TheModule);
  }
}

void clang::EmitBackendOutput(Diagnostic &Diags, const CodeGenOptions &CGOpts,
                              const TargetOptions &TOpts, Module *M,
                              BackendAction Action, raw_ostream *OS) {
  EmitAssemblyHelper AsmHelper(Diags, CGOpts, TOpts, M);

  AsmHelper.EmitAssembly(Action, OS);
}
