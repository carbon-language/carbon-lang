//===--- CodeGenAction.cpp - LLVM Code Generation Frontend Action ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CoverageMappingGen.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"
#include <memory>
using namespace clang;
using namespace llvm;

namespace clang {
  class BackendConsumer : public ASTConsumer {
    virtual void anchor();
    DiagnosticsEngine &Diags;
    BackendAction Action;
    const CodeGenOptions &CodeGenOpts;
    const TargetOptions &TargetOpts;
    const LangOptions &LangOpts;
    raw_pwrite_stream *AsmOutStream;
    ASTContext *Context;

    Timer LLVMIRGeneration;

    std::unique_ptr<CodeGenerator> Gen;

    std::unique_ptr<llvm::Module> TheModule, LinkModule;

  public:
    BackendConsumer(BackendAction Action, DiagnosticsEngine &Diags,
                    const CodeGenOptions &CodeGenOpts,
                    const TargetOptions &TargetOpts,
                    const LangOptions &LangOpts, bool TimePasses,
                    const std::string &InFile, llvm::Module *LinkModule,
                    raw_pwrite_stream *OS, LLVMContext &C,
                    CoverageSourceInfo *CoverageInfo = nullptr)
        : Diags(Diags), Action(Action), CodeGenOpts(CodeGenOpts),
          TargetOpts(TargetOpts), LangOpts(LangOpts), AsmOutStream(OS),
          Context(nullptr), LLVMIRGeneration("LLVM IR Generation Time"),
          Gen(CreateLLVMCodeGen(Diags, InFile, CodeGenOpts, C, CoverageInfo)),
          LinkModule(LinkModule) {
      llvm::TimePassesIsEnabled = TimePasses;
    }

    std::unique_ptr<llvm::Module> takeModule() { return std::move(TheModule); }
    llvm::Module *takeLinkModule() { return LinkModule.release(); }

    void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override {
      Gen->HandleCXXStaticMemberVarInstantiation(VD);
    }

    void Initialize(ASTContext &Ctx) override {
      if (Context) {
        assert(Context == &Ctx);
        return;
      }
        
      Context = &Ctx;

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->Initialize(Ctx);

      TheModule.reset(Gen->GetModule());

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.stopTimer();
    }

    bool HandleTopLevelDecl(DeclGroupRef D) override {
      PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of declaration");

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->HandleTopLevelDecl(D);

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.stopTimer();

      return true;
    }

    void HandleInlineMethodDefinition(CXXMethodDecl *D) override {
      PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of inline method");
      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.startTimer();

      Gen->HandleInlineMethodDefinition(D);

      if (llvm::TimePassesIsEnabled)
        LLVMIRGeneration.stopTimer();
    }

    void HandleTranslationUnit(ASTContext &C) override {
      {
        PrettyStackTraceString CrashInfo("Per-file LLVM IR generation");
        if (llvm::TimePassesIsEnabled)
          LLVMIRGeneration.startTimer();

        Gen->HandleTranslationUnit(C);

        if (llvm::TimePassesIsEnabled)
          LLVMIRGeneration.stopTimer();
      }

      // Silently ignore if we weren't initialized for some reason.
      if (!TheModule)
        return;

      // Make sure IR generation is happy with the module. This is released by
      // the module provider.
      llvm::Module *M = Gen->ReleaseModule();
      if (!M) {
        // The module has been released by IR gen on failures, do not double
        // free.
        TheModule.release();
        return;
      }

      assert(TheModule.get() == M &&
             "Unexpected module change during IR generation");

      // Link LinkModule into this module if present, preserving its validity.
      if (LinkModule) {
        if (Linker::LinkModules(
                M, LinkModule.get(),
                [=](const DiagnosticInfo &DI) { linkerDiagnosticHandler(DI); }))
          return;
      }

      // Install an inline asm handler so that diagnostics get printed through
      // our diagnostics hooks.
      LLVMContext &Ctx = TheModule->getContext();
      LLVMContext::InlineAsmDiagHandlerTy OldHandler =
        Ctx.getInlineAsmDiagnosticHandler();
      void *OldContext = Ctx.getInlineAsmDiagnosticContext();
      Ctx.setInlineAsmDiagnosticHandler(InlineAsmDiagHandler, this);

      LLVMContext::DiagnosticHandlerTy OldDiagnosticHandler =
          Ctx.getDiagnosticHandler();
      void *OldDiagnosticContext = Ctx.getDiagnosticContext();
      Ctx.setDiagnosticHandler(DiagnosticHandler, this);

      EmitBackendOutput(Diags, CodeGenOpts, TargetOpts, LangOpts,
                        C.getTargetInfo().getTargetDescription(),
                        TheModule.get(), Action, AsmOutStream);

      Ctx.setInlineAsmDiagnosticHandler(OldHandler, OldContext);

      Ctx.setDiagnosticHandler(OldDiagnosticHandler, OldDiagnosticContext);
    }

    void HandleTagDeclDefinition(TagDecl *D) override {
      PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                     Context->getSourceManager(),
                                     "LLVM IR generation of declaration");
      Gen->HandleTagDeclDefinition(D);
    }

    void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
      Gen->HandleTagDeclRequiredDefinition(D);
    }

    void CompleteTentativeDefinition(VarDecl *D) override {
      Gen->CompleteTentativeDefinition(D);
    }

    void HandleVTable(CXXRecordDecl *RD) override {
      Gen->HandleVTable(RD);
    }

    void HandleLinkerOptionPragma(llvm::StringRef Opts) override {
      Gen->HandleLinkerOptionPragma(Opts);
    }

    void HandleDetectMismatch(llvm::StringRef Name,
                                      llvm::StringRef Value) override {
      Gen->HandleDetectMismatch(Name, Value);
    }

    void HandleDependentLibrary(llvm::StringRef Opts) override {
      Gen->HandleDependentLibrary(Opts);
    }

    static void InlineAsmDiagHandler(const llvm::SMDiagnostic &SM,void *Context,
                                     unsigned LocCookie) {
      SourceLocation Loc = SourceLocation::getFromRawEncoding(LocCookie);
      ((BackendConsumer*)Context)->InlineAsmDiagHandler2(SM, Loc);
    }

    void linkerDiagnosticHandler(const llvm::DiagnosticInfo &DI);

    static void DiagnosticHandler(const llvm::DiagnosticInfo &DI,
                                  void *Context) {
      ((BackendConsumer *)Context)->DiagnosticHandlerImpl(DI);
    }

    void InlineAsmDiagHandler2(const llvm::SMDiagnostic &,
                               SourceLocation LocCookie);

    void DiagnosticHandlerImpl(const llvm::DiagnosticInfo &DI);
    /// \brief Specialized handler for InlineAsm diagnostic.
    /// \return True if the diagnostic has been successfully reported, false
    /// otherwise.
    bool InlineAsmDiagHandler(const llvm::DiagnosticInfoInlineAsm &D);
    /// \brief Specialized handler for StackSize diagnostic.
    /// \return True if the diagnostic has been successfully reported, false
    /// otherwise.
    bool StackSizeDiagHandler(const llvm::DiagnosticInfoStackSize &D);
    /// \brief Specialized handlers for optimization remarks.
    /// Note that these handlers only accept remarks and they always handle
    /// them.
    void EmitOptimizationMessage(const llvm::DiagnosticInfoOptimizationBase &D,
                                 unsigned DiagID);
    void
    OptimizationRemarkHandler(const llvm::DiagnosticInfoOptimizationRemark &D);
    void OptimizationRemarkHandler(
        const llvm::DiagnosticInfoOptimizationRemarkMissed &D);
    void OptimizationRemarkHandler(
        const llvm::DiagnosticInfoOptimizationRemarkAnalysis &D);
    void OptimizationFailureHandler(
        const llvm::DiagnosticInfoOptimizationFailure &D);
  };
  
  void BackendConsumer::anchor() {}
}

/// ConvertBackendLocation - Convert a location in a temporary llvm::SourceMgr
/// buffer to be a valid FullSourceLoc.
static FullSourceLoc ConvertBackendLocation(const llvm::SMDiagnostic &D,
                                            SourceManager &CSM) {
  // Get both the clang and llvm source managers.  The location is relative to
  // a memory buffer that the LLVM Source Manager is handling, we need to add
  // a copy to the Clang source manager.
  const llvm::SourceMgr &LSM = *D.getSourceMgr();

  // We need to copy the underlying LLVM memory buffer because llvm::SourceMgr
  // already owns its one and clang::SourceManager wants to own its one.
  const MemoryBuffer *LBuf =
  LSM.getMemoryBuffer(LSM.FindBufferContainingLoc(D.getLoc()));

  // Create the copy and transfer ownership to clang::SourceManager.
  // TODO: Avoid copying files into memory.
  std::unique_ptr<llvm::MemoryBuffer> CBuf =
      llvm::MemoryBuffer::getMemBufferCopy(LBuf->getBuffer(),
                                           LBuf->getBufferIdentifier());
  // FIXME: Keep a file ID map instead of creating new IDs for each location.
  FileID FID = CSM.createFileID(std::move(CBuf));

  // Translate the offset into the file.
  unsigned Offset = D.getLoc().getPointer() - LBuf->getBufferStart();
  SourceLocation NewLoc =
  CSM.getLocForStartOfFile(FID).getLocWithOffset(Offset);
  return FullSourceLoc(NewLoc, CSM);
}


/// InlineAsmDiagHandler2 - This function is invoked when the backend hits an
/// error parsing inline asm.  The SMDiagnostic indicates the error relative to
/// the temporary memory buffer that the inline asm parser has set up.
void BackendConsumer::InlineAsmDiagHandler2(const llvm::SMDiagnostic &D,
                                            SourceLocation LocCookie) {
  // There are a couple of different kinds of errors we could get here.  First,
  // we re-format the SMDiagnostic in terms of a clang diagnostic.

  // Strip "error: " off the start of the message string.
  StringRef Message = D.getMessage();
  if (Message.startswith("error: "))
    Message = Message.substr(7);

  // If the SMDiagnostic has an inline asm source location, translate it.
  FullSourceLoc Loc;
  if (D.getLoc() != SMLoc())
    Loc = ConvertBackendLocation(D, Context->getSourceManager());

  unsigned DiagID;
  switch (D.getKind()) {
  case llvm::SourceMgr::DK_Error:
    DiagID = diag::err_fe_inline_asm;
    break;
  case llvm::SourceMgr::DK_Warning:
    DiagID = diag::warn_fe_inline_asm;
    break;
  case llvm::SourceMgr::DK_Note:
    DiagID = diag::note_fe_inline_asm;
    break;
  }
  // If this problem has clang-level source location information, report the
  // issue in the source with a note showing the instantiated
  // code.
  if (LocCookie.isValid()) {
    Diags.Report(LocCookie, DiagID).AddString(Message);
    
    if (D.getLoc().isValid()) {
      DiagnosticBuilder B = Diags.Report(Loc, diag::note_fe_inline_asm_here);
      // Convert the SMDiagnostic ranges into SourceRange and attach them
      // to the diagnostic.
      for (unsigned i = 0, e = D.getRanges().size(); i != e; ++i) {
        std::pair<unsigned, unsigned> Range = D.getRanges()[i];
        unsigned Column = D.getColumnNo();
        B << SourceRange(Loc.getLocWithOffset(Range.first - Column),
                         Loc.getLocWithOffset(Range.second - Column));
      }
    }
    return;
  }
  
  // Otherwise, report the backend issue as occurring in the generated .s file.
  // If Loc is invalid, we still need to report the issue, it just gets no
  // location info.
  Diags.Report(Loc, DiagID).AddString(Message);
}

#define ComputeDiagID(Severity, GroupName, DiagID)                             \
  do {                                                                         \
    switch (Severity) {                                                        \
    case llvm::DS_Error:                                                       \
      DiagID = diag::err_fe_##GroupName;                                       \
      break;                                                                   \
    case llvm::DS_Warning:                                                     \
      DiagID = diag::warn_fe_##GroupName;                                      \
      break;                                                                   \
    case llvm::DS_Remark:                                                      \
      llvm_unreachable("'remark' severity not expected");                      \
      break;                                                                   \
    case llvm::DS_Note:                                                        \
      DiagID = diag::note_fe_##GroupName;                                      \
      break;                                                                   \
    }                                                                          \
  } while (false)

#define ComputeDiagRemarkID(Severity, GroupName, DiagID)                       \
  do {                                                                         \
    switch (Severity) {                                                        \
    case llvm::DS_Error:                                                       \
      DiagID = diag::err_fe_##GroupName;                                       \
      break;                                                                   \
    case llvm::DS_Warning:                                                     \
      DiagID = diag::warn_fe_##GroupName;                                      \
      break;                                                                   \
    case llvm::DS_Remark:                                                      \
      DiagID = diag::remark_fe_##GroupName;                                    \
      break;                                                                   \
    case llvm::DS_Note:                                                        \
      DiagID = diag::note_fe_##GroupName;                                      \
      break;                                                                   \
    }                                                                          \
  } while (false)

bool
BackendConsumer::InlineAsmDiagHandler(const llvm::DiagnosticInfoInlineAsm &D) {
  unsigned DiagID;
  ComputeDiagID(D.getSeverity(), inline_asm, DiagID);
  std::string Message = D.getMsgStr().str();

  // If this problem has clang-level source location information, report the
  // issue as being a problem in the source with a note showing the instantiated
  // code.
  SourceLocation LocCookie =
      SourceLocation::getFromRawEncoding(D.getLocCookie());
  if (LocCookie.isValid())
    Diags.Report(LocCookie, DiagID).AddString(Message);
  else {
    // Otherwise, report the backend diagnostic as occurring in the generated
    // .s file.
    // If Loc is invalid, we still need to report the diagnostic, it just gets
    // no location info.
    FullSourceLoc Loc;
    Diags.Report(Loc, DiagID).AddString(Message);
  }
  // We handled all the possible severities.
  return true;
}

bool
BackendConsumer::StackSizeDiagHandler(const llvm::DiagnosticInfoStackSize &D) {
  if (D.getSeverity() != llvm::DS_Warning)
    // For now, the only support we have for StackSize diagnostic is warning.
    // We do not know how to format other severities.
    return false;

  if (const Decl *ND = Gen->GetDeclForMangledName(D.getFunction().getName())) {
    Diags.Report(ND->getASTContext().getFullLoc(ND->getLocation()),
                 diag::warn_fe_frame_larger_than)
        << D.getStackSize() << Decl::castToDeclContext(ND);
    return true;
  }

  return false;
}

void BackendConsumer::EmitOptimizationMessage(
    const llvm::DiagnosticInfoOptimizationBase &D, unsigned DiagID) {
  // We only support warnings and remarks.
  assert(D.getSeverity() == llvm::DS_Remark ||
         D.getSeverity() == llvm::DS_Warning);

  SourceManager &SourceMgr = Context->getSourceManager();
  FileManager &FileMgr = SourceMgr.getFileManager();
  StringRef Filename;
  unsigned Line, Column;
  SourceLocation DILoc;

  if (D.isLocationAvailable()) {
    D.getLocation(&Filename, &Line, &Column);
    const FileEntry *FE = FileMgr.getFile(Filename);
    if (FE && Line > 0) {
      // If -gcolumn-info was not used, Column will be 0. This upsets the
      // source manager, so pass 1 if Column is not set.
      DILoc = SourceMgr.translateFileLineCol(FE, Line, Column ? Column : 1);
    }
  }

  // If a location isn't available, try to approximate it using the associated
  // function definition. We use the definition's right brace to differentiate
  // from diagnostics that genuinely relate to the function itself.
  FullSourceLoc Loc(DILoc, SourceMgr);
  if (Loc.isInvalid())
    if (const Decl *FD = Gen->GetDeclForMangledName(D.getFunction().getName()))
      Loc = FD->getASTContext().getFullLoc(FD->getBodyRBrace());

  Diags.Report(Loc, DiagID)
      << AddFlagValue(D.getPassName() ? D.getPassName() : "")
      << D.getMsg().str();

  if (DILoc.isInvalid() && D.isLocationAvailable())
    // If we were not able to translate the file:line:col information
    // back to a SourceLocation, at least emit a note stating that
    // we could not translate this location. This can happen in the
    // case of #line directives.
    Diags.Report(Loc, diag::note_fe_backend_optimization_remark_invalid_loc)
        << Filename << Line << Column;
}

void BackendConsumer::OptimizationRemarkHandler(
    const llvm::DiagnosticInfoOptimizationRemark &D) {
  // Optimization remarks are active only if the -Rpass flag has a regular
  // expression that matches the name of the pass name in \p D.
  if (CodeGenOpts.OptimizationRemarkPattern &&
      CodeGenOpts.OptimizationRemarkPattern->match(D.getPassName()))
    EmitOptimizationMessage(D, diag::remark_fe_backend_optimization_remark);
}

void BackendConsumer::OptimizationRemarkHandler(
    const llvm::DiagnosticInfoOptimizationRemarkMissed &D) {
  // Missed optimization remarks are active only if the -Rpass-missed
  // flag has a regular expression that matches the name of the pass
  // name in \p D.
  if (CodeGenOpts.OptimizationRemarkMissedPattern &&
      CodeGenOpts.OptimizationRemarkMissedPattern->match(D.getPassName()))
    EmitOptimizationMessage(D,
                            diag::remark_fe_backend_optimization_remark_missed);
}

void BackendConsumer::OptimizationRemarkHandler(
    const llvm::DiagnosticInfoOptimizationRemarkAnalysis &D) {
  // Optimization analysis remarks are active only if the -Rpass-analysis
  // flag has a regular expression that matches the name of the pass
  // name in \p D.
  if (CodeGenOpts.OptimizationRemarkAnalysisPattern &&
      CodeGenOpts.OptimizationRemarkAnalysisPattern->match(D.getPassName()))
    EmitOptimizationMessage(
        D, diag::remark_fe_backend_optimization_remark_analysis);
}

void BackendConsumer::OptimizationFailureHandler(
    const llvm::DiagnosticInfoOptimizationFailure &D) {
  EmitOptimizationMessage(D, diag::warn_fe_backend_optimization_failure);
}

void BackendConsumer::linkerDiagnosticHandler(const DiagnosticInfo &DI) {
  if (DI.getSeverity() != DS_Error)
    return;

  std::string MsgStorage;
  {
    raw_string_ostream Stream(MsgStorage);
    DiagnosticPrinterRawOStream DP(Stream);
    DI.print(DP);
  }

  Diags.Report(diag::err_fe_cannot_link_module)
      << LinkModule->getModuleIdentifier() << MsgStorage;
}

/// \brief This function is invoked when the backend needs
/// to report something to the user.
void BackendConsumer::DiagnosticHandlerImpl(const DiagnosticInfo &DI) {
  unsigned DiagID = diag::err_fe_inline_asm;
  llvm::DiagnosticSeverity Severity = DI.getSeverity();
  // Get the diagnostic ID based.
  switch (DI.getKind()) {
  case llvm::DK_InlineAsm:
    if (InlineAsmDiagHandler(cast<DiagnosticInfoInlineAsm>(DI)))
      return;
    ComputeDiagID(Severity, inline_asm, DiagID);
    break;
  case llvm::DK_StackSize:
    if (StackSizeDiagHandler(cast<DiagnosticInfoStackSize>(DI)))
      return;
    ComputeDiagID(Severity, backend_frame_larger_than, DiagID);
    break;
  case llvm::DK_OptimizationRemark:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<DiagnosticInfoOptimizationRemark>(DI));
    return;
  case llvm::DK_OptimizationRemarkMissed:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(cast<DiagnosticInfoOptimizationRemarkMissed>(DI));
    return;
  case llvm::DK_OptimizationRemarkAnalysis:
    // Optimization remarks are always handled completely by this
    // handler. There is no generic way of emitting them.
    OptimizationRemarkHandler(
        cast<DiagnosticInfoOptimizationRemarkAnalysis>(DI));
    return;
  case llvm::DK_OptimizationFailure:
    // Optimization failures are always handled completely by this
    // handler.
    OptimizationFailureHandler(cast<DiagnosticInfoOptimizationFailure>(DI));
    return;
  default:
    // Plugin IDs are not bound to any value as they are set dynamically.
    ComputeDiagRemarkID(Severity, backend_plugin, DiagID);
    break;
  }
  std::string MsgStorage;
  {
    raw_string_ostream Stream(MsgStorage);
    DiagnosticPrinterRawOStream DP(Stream);
    DI.print(DP);
  }

  // Report the backend message using the usual diagnostic mechanism.
  FullSourceLoc Loc;
  Diags.Report(Loc, DiagID).AddString(MsgStorage);
}
#undef ComputeDiagID

CodeGenAction::CodeGenAction(unsigned _Act, LLVMContext *_VMContext)
  : Act(_Act), LinkModule(nullptr),
    VMContext(_VMContext ? _VMContext : new LLVMContext),
    OwnsVMContext(!_VMContext) {}

CodeGenAction::~CodeGenAction() {
  TheModule.reset();
  if (OwnsVMContext)
    delete VMContext;
}

bool CodeGenAction::hasIRSupport() const { return true; }

void CodeGenAction::EndSourceFileAction() {
  // If the consumer creation failed, do nothing.
  if (!getCompilerInstance().hasASTConsumer())
    return;

  // If we were given a link module, release consumer's ownership of it.
  if (LinkModule)
    BEConsumer->takeLinkModule();

  // Steal the module from the consumer.
  TheModule = BEConsumer->takeModule();
}

std::unique_ptr<llvm::Module> CodeGenAction::takeModule() {
  return std::move(TheModule);
}

llvm::LLVMContext *CodeGenAction::takeLLVMContext() {
  OwnsVMContext = false;
  return VMContext;
}

static raw_pwrite_stream *
GetOutputStream(CompilerInstance &CI, StringRef InFile, BackendAction Action) {
  switch (Action) {
  case Backend_EmitAssembly:
    return CI.createDefaultOutputFile(false, InFile, "s");
  case Backend_EmitLL:
    return CI.createDefaultOutputFile(false, InFile, "ll");
  case Backend_EmitBC:
    return CI.createDefaultOutputFile(true, InFile, "bc");
  case Backend_EmitNothing:
    return nullptr;
  case Backend_EmitMCNull:
    return CI.createNullOutputFile();
  case Backend_EmitObj:
    return CI.createDefaultOutputFile(true, InFile, "o");
  }

  llvm_unreachable("Invalid action!");
}

std::unique_ptr<ASTConsumer>
CodeGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  BackendAction BA = static_cast<BackendAction>(Act);
  raw_pwrite_stream *OS = GetOutputStream(CI, InFile, BA);
  if (BA != Backend_EmitNothing && !OS)
    return nullptr;

  llvm::Module *LinkModuleToUse = LinkModule;

  // If we were not given a link module, and the user requested that one be
  // loaded from bitcode, do so now.
  const std::string &LinkBCFile = CI.getCodeGenOpts().LinkBitcodeFile;
  if (!LinkModuleToUse && !LinkBCFile.empty()) {
    auto BCBuf = CI.getFileManager().getBufferForFile(LinkBCFile);
    if (!BCBuf) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
          << LinkBCFile << BCBuf.getError().message();
      return nullptr;
    }

    ErrorOr<llvm::Module *> ModuleOrErr =
        getLazyBitcodeModule(std::move(*BCBuf), *VMContext);
    if (std::error_code EC = ModuleOrErr.getError()) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
        << LinkBCFile << EC.message();
      return nullptr;
    }
    LinkModuleToUse = ModuleOrErr.get();
  }

  CoverageSourceInfo *CoverageInfo = nullptr;
  // Add the preprocessor callback only when the coverage mapping is generated.
  if (CI.getCodeGenOpts().CoverageMapping) {
    CoverageInfo = new CoverageSourceInfo;
    CI.getPreprocessor().addPPCallbacks(
                                    std::unique_ptr<PPCallbacks>(CoverageInfo));
  }
  std::unique_ptr<BackendConsumer> Result(new BackendConsumer(
      BA, CI.getDiagnostics(), CI.getCodeGenOpts(), CI.getTargetOpts(),
      CI.getLangOpts(), CI.getFrontendOpts().ShowTimers, InFile,
      LinkModuleToUse, OS, *VMContext, CoverageInfo));
  BEConsumer = Result.get();
  return std::move(Result);
}

static void BitcodeInlineAsmDiagHandler(const llvm::SMDiagnostic &SM,
                                         void *Context,
                                         unsigned LocCookie) {
  SM.print(nullptr, llvm::errs());
}

void CodeGenAction::ExecuteAction() {
  // If this is an IR file, we have to treat it specially.
  if (getCurrentFileKind() == IK_LLVM_IR) {
    BackendAction BA = static_cast<BackendAction>(Act);
    CompilerInstance &CI = getCompilerInstance();
    raw_pwrite_stream *OS = GetOutputStream(CI, getCurrentFile(), BA);
    if (BA != Backend_EmitNothing && !OS)
      return;

    bool Invalid;
    SourceManager &SM = CI.getSourceManager();
    FileID FID = SM.getMainFileID();
    llvm::MemoryBuffer *MainFile = SM.getBuffer(FID, &Invalid);
    if (Invalid)
      return;

    llvm::SMDiagnostic Err;
    TheModule = parseIR(MainFile->getMemBufferRef(), Err, *VMContext);
    if (!TheModule) {
      // Translate from the diagnostic info to the SourceManager location if
      // available.
      // TODO: Unify this with ConvertBackendLocation()
      SourceLocation Loc;
      if (Err.getLineNo() > 0) {
        assert(Err.getColumnNo() >= 0);
        Loc = SM.translateFileLineCol(SM.getFileEntryForID(FID),
                                      Err.getLineNo(), Err.getColumnNo() + 1);
      }

      // Strip off a leading diagnostic code if there is one.
      StringRef Msg = Err.getMessage();
      if (Msg.startswith("error: "))
        Msg = Msg.substr(7);

      unsigned DiagID =
          CI.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error, "%0");

      CI.getDiagnostics().Report(Loc, DiagID) << Msg;
      return;
    }
    const TargetOptions &TargetOpts = CI.getTargetOpts();
    if (TheModule->getTargetTriple() != TargetOpts.Triple) {
      CI.getDiagnostics().Report(SourceLocation(),
                                 diag::warn_fe_override_module)
          << TargetOpts.Triple;
      TheModule->setTargetTriple(TargetOpts.Triple);
    }

    LLVMContext &Ctx = TheModule->getContext();
    Ctx.setInlineAsmDiagnosticHandler(BitcodeInlineAsmDiagHandler);
    EmitBackendOutput(CI.getDiagnostics(), CI.getCodeGenOpts(), TargetOpts,
                      CI.getLangOpts(), CI.getTarget().getTargetDescription(),
                      TheModule.get(), BA, OS);
    return;
  }

  // Otherwise follow the normal AST path.
  this->ASTFrontendAction::ExecuteAction();
}

//

void EmitAssemblyAction::anchor() { }
EmitAssemblyAction::EmitAssemblyAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitAssembly, _VMContext) {}

void EmitBCAction::anchor() { }
EmitBCAction::EmitBCAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitBC, _VMContext) {}

void EmitLLVMAction::anchor() { }
EmitLLVMAction::EmitLLVMAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitLL, _VMContext) {}

void EmitLLVMOnlyAction::anchor() { }
EmitLLVMOnlyAction::EmitLLVMOnlyAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitNothing, _VMContext) {}

void EmitCodeGenOnlyAction::anchor() { }
EmitCodeGenOnlyAction::EmitCodeGenOnlyAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitMCNull, _VMContext) {}

void EmitObjAction::anchor() { }
EmitObjAction::EmitObjAction(llvm::LLVMContext *_VMContext)
  : CodeGenAction(Backend_EmitObj, _VMContext) {}
