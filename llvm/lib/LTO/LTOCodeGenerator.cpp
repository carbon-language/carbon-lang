//===-LTOCodeGenerator.cpp - LLVM Link Time Optimizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/ObjCARC.h"
using namespace llvm;

const char* LTOCodeGenerator::getVersionString() {
#ifdef LLVM_VERSION_INFO
  return PACKAGE_NAME " version " PACKAGE_VERSION ", " LLVM_VERSION_INFO;
#else
  return PACKAGE_NAME " version " PACKAGE_VERSION;
#endif
}

LTOCodeGenerator::LTOCodeGenerator()
    : Context(getGlobalContext()), Linker(new Module("ld-temp.o", Context)),
      TargetMach(nullptr), EmitDwarfDebugInfo(false),
      ScopeRestrictionsDone(false), CodeModel(LTO_CODEGEN_PIC_MODEL_DEFAULT),
      NativeObjectFile(nullptr), DiagHandler(nullptr), DiagContext(nullptr) {
  initializeLTOPasses();
}

LTOCodeGenerator::~LTOCodeGenerator() {
  delete TargetMach;
  delete NativeObjectFile;
  TargetMach = nullptr;
  NativeObjectFile = nullptr;

  Linker.deleteModule();

  for (std::vector<char *>::iterator I = CodegenOptions.begin(),
                                     E = CodegenOptions.end();
       I != E; ++I)
    free(*I);
}

// Initialize LTO passes. Please keep this funciton in sync with
// PassManagerBuilder::populateLTOPassManager(), and make sure all LTO
// passes are initialized.
void LTOCodeGenerator::initializeLTOPasses() {
  PassRegistry &R = *PassRegistry::getPassRegistry();

  initializeInternalizePassPass(R);
  initializeIPSCCPPass(R);
  initializeGlobalOptPass(R);
  initializeConstantMergePass(R);
  initializeDAHPass(R);
  initializeInstCombinerPass(R);
  initializeSimpleInlinerPass(R);
  initializePruneEHPass(R);
  initializeGlobalDCEPass(R);
  initializeArgPromotionPass(R);
  initializeJumpThreadingPass(R);
  initializeSROAPass(R);
  initializeSROA_DTPass(R);
  initializeSROA_SSAUpPass(R);
  initializeFunctionAttrsPass(R);
  initializeGlobalsModRefPass(R);
  initializeLICMPass(R);
  initializeGVNPass(R);
  initializeMemCpyOptPass(R);
  initializeDCEPass(R);
  initializeCFGSimplifyPassPass(R);
}

bool LTOCodeGenerator::addModule(LTOModule* mod, std::string& errMsg) {
  bool ret = Linker.linkInModule(mod->getLLVVMModule(), &errMsg);

  const std::vector<const char*> &undefs = mod->getAsmUndefinedRefs();
  for (int i = 0, e = undefs.size(); i != e; ++i)
    AsmUndefinedRefs[undefs[i]] = 1;

  return !ret;
}

void LTOCodeGenerator::setTargetOptions(TargetOptions options) {
  Options.LessPreciseFPMADOption = options.LessPreciseFPMADOption;
  Options.NoFramePointerElim = options.NoFramePointerElim;
  Options.AllowFPOpFusion = options.AllowFPOpFusion;
  Options.UnsafeFPMath = options.UnsafeFPMath;
  Options.NoInfsFPMath = options.NoInfsFPMath;
  Options.NoNaNsFPMath = options.NoNaNsFPMath;
  Options.HonorSignDependentRoundingFPMathOption =
    options.HonorSignDependentRoundingFPMathOption;
  Options.UseSoftFloat = options.UseSoftFloat;
  Options.FloatABIType = options.FloatABIType;
  Options.NoZerosInBSS = options.NoZerosInBSS;
  Options.GuaranteedTailCallOpt = options.GuaranteedTailCallOpt;
  Options.DisableTailCalls = options.DisableTailCalls;
  Options.StackAlignmentOverride = options.StackAlignmentOverride;
  Options.TrapFuncName = options.TrapFuncName;
  Options.PositionIndependentExecutable = options.PositionIndependentExecutable;
  Options.UseInitArray = options.UseInitArray;
}

void LTOCodeGenerator::setDebugInfo(lto_debug_model debug) {
  switch (debug) {
  case LTO_DEBUG_MODEL_NONE:
    EmitDwarfDebugInfo = false;
    return;

  case LTO_DEBUG_MODEL_DWARF:
    EmitDwarfDebugInfo = true;
    return;
  }
  llvm_unreachable("Unknown debug format!");
}

void LTOCodeGenerator::setCodePICModel(lto_codegen_model model) {
  switch (model) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
  case LTO_CODEGEN_PIC_MODEL_DEFAULT:
    CodeModel = model;
    return;
  }
  llvm_unreachable("Unknown PIC model!");
}

bool LTOCodeGenerator::writeMergedModules(const char *path,
                                          std::string &errMsg) {
  if (!determineTarget(errMsg))
    return false;

  // mark which symbols can not be internalized
  applyScopeRestrictions();

  // create output file
  std::string ErrInfo;
  tool_output_file Out(path, ErrInfo, sys::fs::F_None);
  if (!ErrInfo.empty()) {
    errMsg = "could not open bitcode file for writing: ";
    errMsg += path;
    return false;
  }

  // write bitcode to it
  WriteBitcodeToFile(Linker.getModule(), Out.os());
  Out.os().close();

  if (Out.os().has_error()) {
    errMsg = "could not write bitcode file: ";
    errMsg += path;
    Out.os().clear_error();
    return false;
  }

  Out.keep();
  return true;
}

bool LTOCodeGenerator::compile_to_file(const char** name,
                                       bool disableOpt,
                                       bool disableInline,
                                       bool disableGVNLoadPRE,
                                       std::string& errMsg) {
  // make unique temp .o file to put generated object file
  SmallString<128> Filename;
  int FD;
  error_code EC = sys::fs::createTemporaryFile("lto-llvm", "o", FD, Filename);
  if (EC) {
    errMsg = EC.message();
    return false;
  }

  // generate object file
  tool_output_file objFile(Filename.c_str(), FD);

  bool genResult = generateObjectFile(objFile.os(), disableOpt, disableInline,
                                      disableGVNLoadPRE, errMsg);
  objFile.os().close();
  if (objFile.os().has_error()) {
    objFile.os().clear_error();
    sys::fs::remove(Twine(Filename));
    return false;
  }

  objFile.keep();
  if (!genResult) {
    sys::fs::remove(Twine(Filename));
    return false;
  }

  NativeObjectPath = Filename.c_str();
  *name = NativeObjectPath.c_str();
  return true;
}

const void* LTOCodeGenerator::compile(size_t* length,
                                      bool disableOpt,
                                      bool disableInline,
                                      bool disableGVNLoadPRE,
                                      std::string& errMsg) {
  const char *name;
  if (!compile_to_file(&name, disableOpt, disableInline, disableGVNLoadPRE,
                       errMsg))
    return nullptr;

  // remove old buffer if compile() called twice
  delete NativeObjectFile;

  // read .o file into memory buffer
  std::unique_ptr<MemoryBuffer> BuffPtr;
  if (error_code ec = MemoryBuffer::getFile(name, BuffPtr, -1, false)) {
    errMsg = ec.message();
    sys::fs::remove(NativeObjectPath);
    return nullptr;
  }
  NativeObjectFile = BuffPtr.release();

  // remove temp files
  sys::fs::remove(NativeObjectPath);

  // return buffer, unless error
  if (!NativeObjectFile)
    return nullptr;
  *length = NativeObjectFile->getBufferSize();
  return NativeObjectFile->getBufferStart();
}

bool LTOCodeGenerator::determineTarget(std::string &errMsg) {
  if (TargetMach)
    return true;

  std::string TripleStr = Linker.getModule()->getTargetTriple();
  if (TripleStr.empty())
    TripleStr = sys::getDefaultTargetTriple();
  llvm::Triple Triple(TripleStr);

  // create target machine from info for merged modules
  const Target *march = TargetRegistry::lookupTarget(TripleStr, errMsg);
  if (!march)
    return false;

  // The relocation model is actually a static member of TargetMachine and
  // needs to be set before the TargetMachine is instantiated.
  Reloc::Model RelocModel = Reloc::Default;
  switch (CodeModel) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
    RelocModel = Reloc::Static;
    break;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
    RelocModel = Reloc::PIC_;
    break;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
    RelocModel = Reloc::DynamicNoPIC;
    break;
  case LTO_CODEGEN_PIC_MODEL_DEFAULT:
    // RelocModel is already the default, so leave it that way.
    break;
  }

  // construct LTOModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple);
  std::string FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  if (MCpu.empty() && Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      MCpu = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      MCpu = "yonah";
    else if (Triple.getArch() == llvm::Triple::arm64)
      MCpu = "cyclone";
  }

  TargetMach = march->createTargetMachine(TripleStr, MCpu, FeatureStr, Options,
                                          RelocModel, CodeModel::Default,
                                          CodeGenOpt::Aggressive);
  return true;
}

void LTOCodeGenerator::
applyRestriction(GlobalValue &GV,
                 const ArrayRef<StringRef> &Libcalls,
                 std::vector<const char*> &MustPreserveList,
                 SmallPtrSet<GlobalValue*, 8> &AsmUsed,
                 Mangler &Mangler) {
  // There are no restrictions to apply to declarations.
  if (GV.isDeclaration())
    return;

  // There is nothing more restrictive than private linkage.
  if (GV.hasPrivateLinkage())
    return;

  SmallString<64> Buffer;
  TargetMach->getNameWithPrefix(Buffer, &GV, Mangler);

  if (MustPreserveSymbols.count(Buffer))
    MustPreserveList.push_back(GV.getName().data());
  if (AsmUndefinedRefs.count(Buffer))
    AsmUsed.insert(&GV);

  // Conservatively append user-supplied runtime library functions to
  // llvm.compiler.used.  These could be internalized and deleted by
  // optimizations like -globalopt, causing problems when later optimizations
  // add new library calls (e.g., llvm.memset => memset and printf => puts).
  // Leave it to the linker to remove any dead code (e.g. with -dead_strip).
  if (isa<Function>(GV) &&
      std::binary_search(Libcalls.begin(), Libcalls.end(), GV.getName()))
    AsmUsed.insert(&GV);
}

static void findUsedValues(GlobalVariable *LLVMUsed,
                           SmallPtrSet<GlobalValue*, 8> &UsedValues) {
  if (!LLVMUsed) return;

  ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
  for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i)
    if (GlobalValue *GV =
        dyn_cast<GlobalValue>(Inits->getOperand(i)->stripPointerCasts()))
      UsedValues.insert(GV);
}

static void accumulateAndSortLibcalls(std::vector<StringRef> &Libcalls,
                                      const TargetLibraryInfo& TLI,
                                      const TargetLowering *Lowering)
{
  // TargetLibraryInfo has info on C runtime library calls on the current
  // target.
  for (unsigned I = 0, E = static_cast<unsigned>(LibFunc::NumLibFuncs);
       I != E; ++I) {
    LibFunc::Func F = static_cast<LibFunc::Func>(I);
    if (TLI.has(F))
      Libcalls.push_back(TLI.getName(F));
  }

  // TargetLowering has info on library calls that CodeGen expects to be
  // available, both from the C runtime and compiler-rt.
  if (Lowering)
    for (unsigned I = 0, E = static_cast<unsigned>(RTLIB::UNKNOWN_LIBCALL);
         I != E; ++I)
      if (const char *Name
          = Lowering->getLibcallName(static_cast<RTLIB::Libcall>(I)))
        Libcalls.push_back(Name);

  array_pod_sort(Libcalls.begin(), Libcalls.end());
  Libcalls.erase(std::unique(Libcalls.begin(), Libcalls.end()),
                 Libcalls.end());
}

void LTOCodeGenerator::applyScopeRestrictions() {
  if (ScopeRestrictionsDone)
    return;
  Module *mergedModule = Linker.getModule();

  // Start off with a verification pass.
  PassManager passes;
  passes.add(createVerifierPass());
  passes.add(createDebugInfoVerifierPass());

  // mark which symbols can not be internalized
  Mangler Mangler(TargetMach->getDataLayout());
  std::vector<const char*> MustPreserveList;
  SmallPtrSet<GlobalValue*, 8> AsmUsed;
  std::vector<StringRef> Libcalls;
  TargetLibraryInfo TLI(Triple(TargetMach->getTargetTriple()));
  accumulateAndSortLibcalls(Libcalls, TLI, TargetMach->getTargetLowering());

  for (Module::iterator f = mergedModule->begin(),
         e = mergedModule->end(); f != e; ++f)
    applyRestriction(*f, Libcalls, MustPreserveList, AsmUsed, Mangler);
  for (Module::global_iterator v = mergedModule->global_begin(),
         e = mergedModule->global_end(); v !=  e; ++v)
    applyRestriction(*v, Libcalls, MustPreserveList, AsmUsed, Mangler);
  for (Module::alias_iterator a = mergedModule->alias_begin(),
         e = mergedModule->alias_end(); a != e; ++a)
    applyRestriction(*a, Libcalls, MustPreserveList, AsmUsed, Mangler);

  GlobalVariable *LLVMCompilerUsed =
    mergedModule->getGlobalVariable("llvm.compiler.used");
  findUsedValues(LLVMCompilerUsed, AsmUsed);
  if (LLVMCompilerUsed)
    LLVMCompilerUsed->eraseFromParent();

  if (!AsmUsed.empty()) {
    llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(Context);
    std::vector<Constant*> asmUsed2;
    for (SmallPtrSet<GlobalValue*, 16>::const_iterator i = AsmUsed.begin(),
           e = AsmUsed.end(); i !=e; ++i) {
      GlobalValue *GV = *i;
      Constant *c = ConstantExpr::getBitCast(GV, i8PTy);
      asmUsed2.push_back(c);
    }

    llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, asmUsed2.size());
    LLVMCompilerUsed =
      new llvm::GlobalVariable(*mergedModule, ATy, false,
                               llvm::GlobalValue::AppendingLinkage,
                               llvm::ConstantArray::get(ATy, asmUsed2),
                               "llvm.compiler.used");

    LLVMCompilerUsed->setSection("llvm.metadata");
  }

  passes.add(createInternalizePass(MustPreserveList));

  // apply scope restrictions
  passes.run(*mergedModule);

  ScopeRestrictionsDone = true;
}

/// Optimize merged modules using various IPO passes
bool LTOCodeGenerator::generateObjectFile(raw_ostream &out,
                                          bool DisableOpt,
                                          bool DisableInline,
                                          bool DisableGVNLoadPRE,
                                          std::string &errMsg) {
  if (!this->determineTarget(errMsg))
    return false;

  Module *mergedModule = Linker.getModule();

  // Mark which symbols can not be internalized
  this->applyScopeRestrictions();

  // Instantiate the pass manager to organize the passes.
  PassManager passes;

  // Start off with a verification pass.
  passes.add(createVerifierPass());
  passes.add(createDebugInfoVerifierPass());

  // Add an appropriate DataLayout instance for this module...
  mergedModule->setDataLayout(TargetMach->getDataLayout());
  passes.add(new DataLayoutPass(mergedModule));

  // Add appropriate TargetLibraryInfo for this module.
  passes.add(new TargetLibraryInfo(Triple(TargetMach->getTargetTriple())));

  TargetMach->addAnalysisPasses(passes);

  // Enabling internalize here would use its AllButMain variant. It
  // keeps only main if it exists and does nothing for libraries. Instead
  // we create the pass ourselves with the symbol list provided by the linker.
  if (!DisableOpt)
    PassManagerBuilder().populateLTOPassManager(passes,
                                              /*Internalize=*/false,
                                              !DisableInline,
                                              DisableGVNLoadPRE);

  // Make sure everything is still good.
  passes.add(createVerifierPass());
  passes.add(createDebugInfoVerifierPass());

  PassManager codeGenPasses;

  codeGenPasses.add(new DataLayoutPass(mergedModule));

  formatted_raw_ostream Out(out);

  // If the bitcode files contain ARC code and were compiled with optimization,
  // the ObjCARCContractPass must be run, so do it unconditionally here.
  codeGenPasses.add(createObjCARCContractPass());

  if (TargetMach->addPassesToEmitFile(codeGenPasses, Out,
                                      TargetMachine::CGFT_ObjectFile)) {
    errMsg = "target file type not supported";
    return false;
  }

  // Run our queue of passes all at once now, efficiently.
  passes.run(*mergedModule);

  // Run the code generator, and write assembly file
  codeGenPasses.run(*mergedModule);

  return true;
}

/// setCodeGenDebugOptions - Set codegen debugging options to aid in debugging
/// LTO problems.
void LTOCodeGenerator::setCodeGenDebugOptions(const char *options) {
  for (std::pair<StringRef, StringRef> o = getToken(options);
       !o.first.empty(); o = getToken(o.second)) {
    // ParseCommandLineOptions() expects argv[0] to be program name. Lazily add
    // that.
    if (CodegenOptions.empty())
      CodegenOptions.push_back(strdup("libLLVMLTO"));
    CodegenOptions.push_back(strdup(o.first.str().c_str()));
  }
}

void LTOCodeGenerator::parseCodeGenDebugOptions() {
  // if options were requested, set them
  if (!CodegenOptions.empty())
    cl::ParseCommandLineOptions(CodegenOptions.size(),
                                const_cast<char **>(&CodegenOptions[0]));
}

void LTOCodeGenerator::DiagnosticHandler(const DiagnosticInfo &DI,
                                         void *Context) {
  ((LTOCodeGenerator *)Context)->DiagnosticHandler2(DI);
}

void LTOCodeGenerator::DiagnosticHandler2(const DiagnosticInfo &DI) {
  // Map the LLVM internal diagnostic severity to the LTO diagnostic severity.
  lto_codegen_diagnostic_severity_t Severity;
  switch (DI.getSeverity()) {
  case DS_Error:
    Severity = LTO_DS_ERROR;
    break;
  case DS_Warning:
    Severity = LTO_DS_WARNING;
    break;
  case DS_Remark:
    Severity = LTO_DS_REMARK;
    break;
  case DS_Note:
    Severity = LTO_DS_NOTE;
    break;
  }
  // Create the string that will be reported to the external diagnostic handler.
  std::string MsgStorage;
  raw_string_ostream Stream(MsgStorage);
  DiagnosticPrinterRawOStream DP(Stream);
  DI.print(DP);
  Stream.flush();

  // If this method has been called it means someone has set up an external
  // diagnostic handler. Assert on that.
  assert(DiagHandler && "Invalid diagnostic handler");
  (*DiagHandler)(Severity, MsgStorage.c_str(), DiagContext);
}

void
LTOCodeGenerator::setDiagnosticHandler(lto_diagnostic_handler_t DiagHandler,
                                       void *Ctxt) {
  this->DiagHandler = DiagHandler;
  this->DiagContext = Ctxt;
  if (!DiagHandler)
    return Context.setDiagnosticHandler(nullptr, nullptr);
  // Register the LTOCodeGenerator stub in the LLVMContext to forward the
  // diagnostic to the external DiagHandler.
  Context.setDiagnosticHandler(LTOCodeGenerator::DiagnosticHandler, this);
}
