//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Common/default-kinds.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Verifier.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

using namespace Fortran::frontend;

//===----------------------------------------------------------------------===//
// Custom BeginSourceFileAction
//===----------------------------------------------------------------------===//

bool PrescanAction::beginSourceFileAction() { return runPrescan(); }

bool PrescanAndParseAction::beginSourceFileAction() {
  return runPrescan() && runParse();
}

bool PrescanAndSemaAction::beginSourceFileAction() {
  return runPrescan() && runParse() && runSemanticChecks() &&
         generateRtTypeTables();
}

bool PrescanAndSemaDebugAction::beginSourceFileAction() {
  // This is a "debug" action for development purposes. To facilitate this, the
  // semantic checks are made to succeed unconditionally to prevent this action
  // from exiting early (i.e. in the presence of semantic errors). We should
  // never do this in actions intended for end-users or otherwise regular
  // compiler workflows!
  return runPrescan() && runParse() && (runSemanticChecks() || true) &&
         (generateRtTypeTables() || true);
}

bool CodeGenAction::beginSourceFileAction() {
  llvmCtx = std::make_unique<llvm::LLVMContext>();
  CompilerInstance &ci = this->getInstance();

  // If the input is an LLVM file, just parse it and return.
  if (this->getCurrentInput().getKind().getLanguage() == Language::LLVM_IR) {
    llvm::SMDiagnostic err;
    llvmModule = llvm::parseIRFile(getCurrentInput().getFile(), err, *llvmCtx);
    if (!llvmModule || llvm::verifyModule(*llvmModule, &llvm::errs())) {
      err.print("flang-new", llvm::errs());
      unsigned diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "Could not parse IR");
      ci.getDiagnostics().Report(diagID);
      return false;
    }

    return true;
  }

  // Load the MLIR dialects required by Flang
  mlir::DialectRegistry registry;
  mlirCtx = std::make_unique<mlir::MLIRContext>(registry);
  fir::support::registerNonCodegenDialects(registry);
  fir::support::loadNonCodegenDialects(*mlirCtx);
  fir::support::loadDialects(*mlirCtx);
  fir::support::registerLLVMTranslation(*mlirCtx);

  // If the input is an MLIR file, just parse it and return.
  if (this->getCurrentInput().getKind().getLanguage() == Language::MLIR) {
    llvm::SourceMgr sourceMgr;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(getCurrentInput().getFile());
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, mlirCtx.get());

    if (!module || mlir::failed(module->verifyInvariants())) {
      unsigned diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "Could not parse FIR");
      ci.getDiagnostics().Report(diagID);
      return false;
    }

    mlirModule = std::make_unique<mlir::ModuleOp>(module.release());
    return true;
  }

  // Otherwise, generate an MLIR module from the input Fortran source
  if (getCurrentInput().getKind().getLanguage() != Language::Fortran) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid input type - expecting a Fortran file");
    ci.getDiagnostics().Report(diagID);
    return false;
  }
  bool res = runPrescan() && runParse() && runSemanticChecks() &&
             generateRtTypeTables();
  if (!res)
    return res;

  // Create a LoweringBridge
  const common::IntrinsicTypeDefaultKinds &defKinds =
      ci.getInvocation().getSemanticsContext().defaultKinds();
  fir::KindMapping kindMap(mlirCtx.get(),
      llvm::ArrayRef<fir::KindTy>{fir::fromDefaultKinds(defKinds)});
  lower::LoweringBridge lb = Fortran::lower::LoweringBridge::create(
      *mlirCtx, defKinds, ci.getInvocation().getSemanticsContext().intrinsics(),
      ci.getParsing().allCooked(), ci.getInvocation().getTargetOpts().triple,
      kindMap);

  // Create a parse tree and lower it to FIR
  Fortran::parser::Program &parseTree{*ci.getParsing().parseTree()};
  lb.lower(parseTree, ci.getInvocation().getSemanticsContext());
  mlirModule = std::make_unique<mlir::ModuleOp>(lb.getModule());

  // run the default passes.
  mlir::PassManager pm(mlirCtx.get(), mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier(/*verifyPasses=*/true);
  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());

  if (mlir::failed(pm.run(*mlirModule))) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "verification of lowering to FIR failed");
    ci.getDiagnostics().Report(diagID);
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Custom ExecuteAction
//===----------------------------------------------------------------------===//
void InputOutputTestAction::executeAction() {
  CompilerInstance &ci = getInstance();

  // Create a stream for errors
  std::string buf;
  llvm::raw_string_ostream errorStream{buf};

  // Read the input file
  Fortran::parser::AllSources &allSources{ci.getAllSources()};
  std::string path{getCurrentFileOrBufferName()};
  const Fortran::parser::SourceFile *sf;
  if (path == "-")
    sf = allSources.ReadStandardInput(errorStream);
  else
    sf = allSources.Open(path, errorStream, std::optional<std::string>{"."s});
  llvm::ArrayRef<char> fileContent = sf->content();

  // Output file descriptor to receive the contents of the input file.
  std::unique_ptr<llvm::raw_ostream> os;

  // Copy the contents from the input file to the output file
  if (!ci.isOutputStreamNull()) {
    // An output stream (outputStream_) was set earlier
    ci.writeOutputStream(fileContent.data());
  } else {
    // No pre-set output stream - create an output file
    os = ci.createDefaultOutputFile(
        /*binary=*/true, getCurrentFileOrBufferName(), "txt");
    if (!os)
      return;
    (*os) << fileContent.data();
  }
}

void PrintPreprocessedAction::executeAction() {
  std::string buf;
  llvm::raw_string_ostream outForPP{buf};

  // Format or dump the prescanner's output
  CompilerInstance &ci = this->getInstance();
  if (ci.getInvocation().getPreprocessorOpts().noReformat) {
    ci.getParsing().DumpCookedChars(outForPP);
  } else {
    ci.getParsing().EmitPreprocessedSource(
        outForPP, !ci.getInvocation().getPreprocessorOpts().noLineDirectives);
  }

  // Print getDiagnostics from the prescanner
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());

  // If a pre-defined output stream exists, dump the preprocessed content there
  if (!ci.isOutputStreamNull()) {
    // Send the output to the pre-defined output buffer.
    ci.writeOutputStream(outForPP.str());
    return;
  }

  // Create a file and save the preprocessed output there
  std::unique_ptr<llvm::raw_pwrite_stream> os{ci.createDefaultOutputFile(
      /*Binary=*/true, /*InFile=*/getCurrentFileOrBufferName())};
  if (!os) {
    return;
  }

  (*os) << outForPP.str();
}

void DebugDumpProvenanceAction::executeAction() {
  this->getInstance().getParsing().DumpProvenance(llvm::outs());
}

void ParseSyntaxOnlyAction::executeAction() {}

void DebugUnparseNoSemaAction::executeAction() {
  auto &invoc = this->getInstance().getInvocation();
  auto &parseTree{getInstance().getParsing().parseTree()};

  // TODO: Options should come from CompilerInvocation
  Unparse(llvm::outs(), *parseTree,
          /*encoding=*/Fortran::parser::Encoding::UTF_8,
          /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
          /*preStatement=*/nullptr,
          invoc.getUseAnalyzedObjectsForUnparse() ? &invoc.getAsFortran()
                                                  : nullptr);
}

void DebugUnparseAction::executeAction() {
  auto &invoc = this->getInstance().getInvocation();
  auto &parseTree{getInstance().getParsing().parseTree()};

  CompilerInstance &ci = this->getInstance();
  auto os{ci.createDefaultOutputFile(
      /*Binary=*/false, /*InFile=*/getCurrentFileOrBufferName())};

  // TODO: Options should come from CompilerInvocation
  Unparse(*os, *parseTree,
          /*encoding=*/Fortran::parser::Encoding::UTF_8,
          /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
          /*preStatement=*/nullptr,
          invoc.getUseAnalyzedObjectsForUnparse() ? &invoc.getAsFortran()
                                                  : nullptr);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugUnparseWithSymbolsAction::executeAction() {
  auto &parseTree{*getInstance().getParsing().parseTree()};

  Fortran::semantics::UnparseWithSymbols(
      llvm::outs(), parseTree, /*encoding=*/Fortran::parser::Encoding::UTF_8);

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugDumpSymbolsAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  if (!ci.getRtTyTables().schemata) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "could not find module file for __fortran_type_info");
    ci.getDiagnostics().Report(diagID);
    llvm::errs() << "\n";
    return;
  }

  // Dump symbols
  ci.getSemantics().DumpSymbols(llvm::outs());
}

void DebugDumpAllAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Dump parse tree
  auto &parseTree{getInstance().getParsing().parseTree()};
  llvm::outs() << "========================";
  llvm::outs() << " Flang: parse tree dump ";
  llvm::outs() << "========================\n";
  Fortran::parser::DumpTree(llvm::outs(), parseTree,
                            &ci.getInvocation().getAsFortran());

  if (!ci.getRtTyTables().schemata) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "could not find module file for __fortran_type_info");
    ci.getDiagnostics().Report(diagID);
    llvm::errs() << "\n";
    return;
  }

  // Dump symbols
  llvm::outs() << "=====================";
  llvm::outs() << " Flang: symbols dump ";
  llvm::outs() << "=====================\n";
  ci.getSemantics().DumpSymbols(llvm::outs());
}

void DebugDumpParseTreeNoSemaAction::executeAction() {
  auto &parseTree{getInstance().getParsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree,
      &this->getInstance().getInvocation().getAsFortran());
}

void DebugDumpParseTreeAction::executeAction() {
  auto &parseTree{getInstance().getParsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree,
      &this->getInstance().getInvocation().getAsFortran());

  // Report fatal semantic errors
  reportFatalSemanticErrors();
}

void DebugMeasureParseTreeAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Parse. In case of failure, report and return.
  ci.getParsing().Parse(llvm::outs());

  if (!ci.getParsing().messages().empty() &&
      (ci.getInvocation().getWarnAsErr() ||
       ci.getParsing().messages().AnyFatalError())) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.getDiagnostics().Report(diagID) << getCurrentFileOrBufferName();

    ci.getParsing().messages().Emit(llvm::errs(),
                                    this->getInstance().getAllCookedSources());
    return;
  }

  // Report the getDiagnostics from parsing
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());

  auto &parseTree{*ci.getParsing().parseTree()};

  // Measure the parse tree
  MeasurementVisitor visitor;
  Fortran::parser::Walk(parseTree, visitor);
  llvm::outs() << "Parse tree comprises " << visitor.objects
               << " objects and occupies " << visitor.bytes
               << " total bytes.\n";
}

void DebugPreFIRTreeAction::executeAction() {
  CompilerInstance &ci = this->getInstance();
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  auto &parseTree{*ci.getParsing().parseTree()};

  // Dump pre-FIR tree
  if (auto ast{Fortran::lower::createPFT(
          parseTree, ci.getInvocation().getSemanticsContext())}) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
  } else {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
    ci.getDiagnostics().Report(diagID);
  }
}

void DebugDumpParsingLogAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  ci.getParsing().Parse(llvm::errs());
  ci.getParsing().DumpParsingLog(llvm::outs());
}

void GetDefinitionAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  parser::AllCookedSources &cs = ci.getAllCookedSources();
  unsigned diagID = ci.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Symbol not found");

  auto gdv = ci.getInvocation().getFrontendOpts().getDefVals;
  auto charBlock{cs.GetCharBlockFromLineAndColumns(
      gdv.line, gdv.startColumn, gdv.endColumn)};
  if (!charBlock) {
    ci.getDiagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "String range: >" << charBlock->ToString() << "<\n";

  auto *symbol{ci.getInvocation()
                   .getSemanticsContext()
                   .FindScope(*charBlock)
                   .FindSymbol(*charBlock)};
  if (!symbol) {
    ci.getDiagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";

  auto sourceInfo{cs.GetSourcePositionRange(symbol->name())};
  if (!sourceInfo) {
    llvm_unreachable(
        "Failed to obtain SourcePosition."
        "TODO: Please, write a test and replace this with a diagnostic!");
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";
  llvm::outs() << symbol->name().ToString() << ": "
               << sourceInfo->first.file.path() << ", "
               << sourceInfo->first.line << ", " << sourceInfo->first.column
               << "-" << sourceInfo->second.column << "\n";
}

void GetSymbolsSourcesAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors()) {
    return;
  }

  ci.getSemantics().DumpSymbolsSources(llvm::outs());
}

//===----------------------------------------------------------------------===//
// CodeGenActions
//===----------------------------------------------------------------------===//

CodeGenAction::~CodeGenAction() = default;

#include "flang/Tools/CLOptions.inc"

// Lower the previously generated MLIR module into an LLVM IR module
void CodeGenAction::generateLLVMIR() {
  assert(mlirModule && "The MLIR module has not been generated yet.");

  CompilerInstance &ci = this->getInstance();

  fir::support::loadDialects(*mlirCtx);
  fir::support::registerLLVMTranslation(*mlirCtx);

  // Set-up the MLIR pass manager
  mlir::PassManager pm(mlirCtx.get(), mlir::OpPassManager::Nesting::Implicit);

  pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
  pm.enableVerifier(/*verifyPasses=*/true);

  // Create the pass pipeline
  fir::createMLIRToLLVMPassPipeline(pm);
  mlir::applyPassManagerCLOptions(pm);

  // run the pass manager
  if (!mlir::succeeded(pm.run(*mlirModule))) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Lowering to LLVM IR failed");
    ci.getDiagnostics().Report(diagID);
  }

  // Translate to LLVM IR
  llvm::Optional<llvm::StringRef> moduleName = mlirModule->getName();
  llvmModule = mlir::translateModuleToLLVMIR(
      *mlirModule, *llvmCtx, moduleName ? *moduleName : "FIRModule");

  if (!llvmModule) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "failed to create the LLVM module");
    ci.getDiagnostics().Report(diagID);
    return;
  }
}

void CodeGenAction::setUpTargetMachine() {
  CompilerInstance &ci = this->getInstance();

  // Set the triple based on the CompilerInvocation set-up
  const std::string &theTriple = ci.getInvocation().getTargetOpts().triple;
  if (llvmModule->getTargetTriple() != theTriple) {
    ci.getDiagnostics().Report(clang::diag::warn_fe_override_module)
        << theTriple;
    llvmModule->setTargetTriple(theTriple);
  }

  // Create `Target`
  std::string error;
  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(theTriple, error);
  assert(theTarget && "Failed to create Target");

  // Create `TargetMachine`
  tm.reset(theTarget->createTargetMachine(theTriple, /*CPU=*/"",
                                          /*Features=*/"",
                                          llvm::TargetOptions(), llvm::None));
  assert(tm && "Failed to create TargetMachine");
  llvmModule->setDataLayout(tm->createDataLayout());
}

static std::unique_ptr<llvm::raw_pwrite_stream>
getOutputStream(CompilerInstance &ci, llvm::StringRef inFile,
                BackendActionTy action) {
  switch (action) {
  case BackendActionTy::Backend_EmitAssembly:
    return ci.createDefaultOutputFile(
        /*Binary=*/false, inFile, /*extension=*/"s");
  case BackendActionTy::Backend_EmitLL:
    return ci.createDefaultOutputFile(
        /*Binary=*/false, inFile, /*extension=*/"ll");
  case BackendActionTy::Backend_EmitMLIR:
    return ci.createDefaultOutputFile(
        /*Binary=*/false, inFile, /*extension=*/"mlir");
  case BackendActionTy::Backend_EmitBC:
    return ci.createDefaultOutputFile(
        /*Binary=*/true, inFile, /*extension=*/"bc");
  case BackendActionTy::Backend_EmitObj:
    return ci.createDefaultOutputFile(
        /*Binary=*/true, inFile, /*extension=*/"o");
  }

  llvm_unreachable("Invalid action!");
}

/// Generate target-specific machine-code or assembly file from the input LLVM
/// module.
///
/// \param [in] diags Diagnostics engine for reporting errors
/// \param [in] tm Target machine to aid the code-gen pipeline set-up
/// \param [in] act Backend act to run (assembly vs machine-code generation)
/// \param [in] llvmModule LLVM module to lower to assembly/machine-code
/// \param [out] os Output stream to emit the generated code to
static void generateMachineCodeOrAssemblyImpl(clang::DiagnosticsEngine &diags,
                                              llvm::TargetMachine &tm,
                                              BackendActionTy act,
                                              llvm::Module &llvmModule,
                                              llvm::raw_pwrite_stream &os) {
  assert(((act == BackendActionTy::Backend_EmitObj) ||
          (act == BackendActionTy::Backend_EmitAssembly)) &&
         "Unsupported action");

  // Set-up the pass manager, i.e create an LLVM code-gen pass pipeline.
  // Currently only the legacy pass manager is supported.
  // TODO: Switch to the new PM once it's available in the backend.
  llvm::legacy::PassManager codeGenPasses;
  codeGenPasses.add(
      createTargetTransformInfoWrapperPass(tm.getTargetIRAnalysis()));

  llvm::Triple triple(llvmModule.getTargetTriple());
  std::unique_ptr<llvm::TargetLibraryInfoImpl> tlii =
      std::make_unique<llvm::TargetLibraryInfoImpl>(triple);
  assert(tlii && "Failed to create TargetLibraryInfo");
  codeGenPasses.add(new llvm::TargetLibraryInfoWrapperPass(*tlii));

  llvm::CodeGenFileType cgft = (act == BackendActionTy::Backend_EmitAssembly)
                                   ? llvm::CodeGenFileType::CGFT_AssemblyFile
                                   : llvm::CodeGenFileType::CGFT_ObjectFile;
  if (tm.addPassesToEmitFile(codeGenPasses, os, nullptr, cgft)) {
    unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                              "emission of this file type is not supported");
    diags.Report(diagID);
    return;
  }

  // Run the passes
  codeGenPasses.run(llvmModule);
}

/// Generate LLVM byte code file from the input LLVM module.
///
/// \param [in] tm Target machine to aid the code-gen pipeline set-up
/// \param [in] llvmModule LLVM module to lower to assembly/machine-code
/// \param [out] os Output stream to emit the generated code to
static void generateLLVMBCImpl(llvm::TargetMachine &tm,
                               llvm::Module &llvmModule,
                               llvm::raw_pwrite_stream &os) {
  // Set-up the pass manager
  llvm::ModulePassManager mpm;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder pb(&tm);
  pb.registerModuleAnalyses(mam);
  mpm.addPass(llvm::BitcodeWriterPass(os));

  // run the passes
  mpm.run(llvmModule, mam);
}

void CodeGenAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  // If the output stream is a file, generate it and define the corresponding
  // output stream. If a pre-defined output stream is available, we will use
  // that instead.
  //
  // NOTE: `os` is a smart pointer that will be destroyed at the end of this
  // method. However, it won't be written to until `codeGenPasses` is
  // destroyed. By defining `os` before `codeGenPasses`, we make sure that the
  // output stream won't be destroyed before it is written to. This only
  // applies when an output file is used (i.e. there is no pre-defined output
  // stream).
  // TODO: Revisit once the new PM is ready (i.e. when `codeGenPasses` is
  // updated to use it).
  std::unique_ptr<llvm::raw_pwrite_stream> os;
  if (ci.isOutputStreamNull()) {
    os = getOutputStream(ci, getCurrentFileOrBufferName(), action);

    if (!os) {
      unsigned diagID = ci.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "failed to create the output file");
      ci.getDiagnostics().Report(diagID);
      return;
    }
  }

  if (action == BackendActionTy::Backend_EmitMLIR) {
    mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
    return;
  }

  // generate an LLVM module if it's not already present (it will already be
  // present if the input file is an LLVM IR/BC file).
  if (!llvmModule)
    generateLLVMIR();

  if (action == BackendActionTy::Backend_EmitLL) {
    llvmModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream(),
                      /*AssemblyAnnotationWriter=*/nullptr);
    return;
  }

  setUpTargetMachine();
  if (action == BackendActionTy::Backend_EmitBC) {
    generateLLVMBCImpl(*tm, *llvmModule, *os);
    return;
  }

  if (action == BackendActionTy::Backend_EmitAssembly ||
      action == BackendActionTy::Backend_EmitObj) {
    generateMachineCodeOrAssemblyImpl(
        ci.getDiagnostics(), *tm, action, *llvmModule,
        ci.isOutputStreamNull() ? *os : ci.getOutputStream());
    return;
  }
}

void InitOnlyAction::executeAction() {
  CompilerInstance &ci = this->getInstance();
  unsigned diagID = ci.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Warning,
      "Use `-init-only` for testing purposes only");
  ci.getDiagnostics().Report(diagID);
}

void PluginParseTreeAction::executeAction() {}

void DebugDumpPFTAction::executeAction() {
  CompilerInstance &ci = this->getInstance();

  if (auto ast = Fortran::lower::createPFT(*ci.getParsing().parseTree(),
                                           ci.getSemantics().context())) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
    return;
  }

  unsigned diagID = ci.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
  ci.getDiagnostics().Report(diagID);
}

Fortran::parser::Parsing &PluginParseTreeAction::getParsing() {
  return getInstance().getParsing();
}

std::unique_ptr<llvm::raw_pwrite_stream>
PluginParseTreeAction::createOutputFile(llvm::StringRef extension = "") {

  std::unique_ptr<llvm::raw_pwrite_stream> os{
      getInstance().createDefaultOutputFile(
          /*Binary=*/false, /*InFile=*/getCurrentFileOrBufferName(),
          extension)};
  return os;
}
