//===-- LLVMContext.cpp - Implement LLVMContext ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
//  class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "LLVMContextImpl.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SourceMgr.h"
#include <cctype>
using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) {
  // Create the fixed metadata kinds. This is done in the same order as the
  // MD_* enum values so that they correspond.

  // Create the 'dbg' metadata kind.
  unsigned DbgID = getMDKindID("dbg");
  assert(DbgID == MD_dbg && "dbg kind id drifted"); (void)DbgID;

  // Create the 'tbaa' metadata kind.
  unsigned TBAAID = getMDKindID("tbaa");
  assert(TBAAID == MD_tbaa && "tbaa kind id drifted"); (void)TBAAID;

  // Create the 'prof' metadata kind.
  unsigned ProfID = getMDKindID("prof");
  assert(ProfID == MD_prof && "prof kind id drifted"); (void)ProfID;

  // Create the 'fpmath' metadata kind.
  unsigned FPAccuracyID = getMDKindID("fpmath");
  assert(FPAccuracyID == MD_fpmath && "fpmath kind id drifted");
  (void)FPAccuracyID;

  // Create the 'range' metadata kind.
  unsigned RangeID = getMDKindID("range");
  assert(RangeID == MD_range && "range kind id drifted");
  (void)RangeID;

  // Create the 'tbaa.struct' metadata kind.
  unsigned TBAAStructID = getMDKindID("tbaa.struct");
  assert(TBAAStructID == MD_tbaa_struct && "tbaa.struct kind id drifted");
  (void)TBAAStructID;

  // Create the 'invariant.load' metadata kind.
  unsigned InvariantLdId = getMDKindID("invariant.load");
  assert(InvariantLdId == MD_invariant_load && "invariant.load kind id drifted");
  (void)InvariantLdId;

  // Create the 'alias.scope' metadata kind.
  unsigned AliasScopeID = getMDKindID("alias.scope");
  assert(AliasScopeID == MD_alias_scope && "alias.scope kind id drifted");
  (void)AliasScopeID;

  // Create the 'noalias' metadata kind.
  unsigned NoAliasID = getMDKindID("noalias");
  assert(NoAliasID == MD_noalias && "noalias kind id drifted");
  (void)NoAliasID;
}
LLVMContext::~LLVMContext() { delete pImpl; }

void LLVMContext::addModule(Module *M) {
  pImpl->OwnedModules.insert(M);
}

void LLVMContext::removeModule(Module *M) {
  pImpl->OwnedModules.erase(M);
}

//===----------------------------------------------------------------------===//
// Recoverable Backend Errors
//===----------------------------------------------------------------------===//

void LLVMContext::
setInlineAsmDiagnosticHandler(InlineAsmDiagHandlerTy DiagHandler,
                              void *DiagContext) {
  pImpl->InlineAsmDiagHandler = DiagHandler;
  pImpl->InlineAsmDiagContext = DiagContext;
}

/// getInlineAsmDiagnosticHandler - Return the diagnostic handler set by
/// setInlineAsmDiagnosticHandler.
LLVMContext::InlineAsmDiagHandlerTy
LLVMContext::getInlineAsmDiagnosticHandler() const {
  return pImpl->InlineAsmDiagHandler;
}

/// getInlineAsmDiagnosticContext - Return the diagnostic context set by
/// setInlineAsmDiagnosticHandler.
void *LLVMContext::getInlineAsmDiagnosticContext() const {
  return pImpl->InlineAsmDiagContext;
}

void LLVMContext::setDiagnosticHandler(DiagnosticHandlerTy DiagnosticHandler,
                                       void *DiagnosticContext) {
  pImpl->DiagnosticHandler = DiagnosticHandler;
  pImpl->DiagnosticContext = DiagnosticContext;
}

LLVMContext::DiagnosticHandlerTy LLVMContext::getDiagnosticHandler() const {
  return pImpl->DiagnosticHandler;
}

void *LLVMContext::getDiagnosticContext() const {
  return pImpl->DiagnosticContext;
}

void LLVMContext::setYieldCallback(YieldCallbackTy Callback, void *OpaqueHandle)
{
  pImpl->YieldCallback = Callback;
  pImpl->YieldOpaqueHandle = OpaqueHandle;
}

void LLVMContext::yield() {
  if (pImpl->YieldCallback)
    pImpl->YieldCallback(this, pImpl->YieldOpaqueHandle);
}

void LLVMContext::emitError(const Twine &ErrorStr) {
  diagnose(DiagnosticInfoInlineAsm(ErrorStr));
}

void LLVMContext::emitError(const Instruction *I, const Twine &ErrorStr) {
  assert (I && "Invalid instruction");
  diagnose(DiagnosticInfoInlineAsm(*I, ErrorStr));
}

void LLVMContext::diagnose(const DiagnosticInfo &DI) {
  // If there is a report handler, use it.
  if (pImpl->DiagnosticHandler) {
    pImpl->DiagnosticHandler(DI, pImpl->DiagnosticContext);
    return;
  }

  // Optimization remarks are selective. They need to check whether the regexp
  // pattern, passed via one of the -pass-remarks* flags, matches the name of
  // the pass that is emitting the diagnostic. If there is no match, ignore the
  // diagnostic and return.
  switch (DI.getKind()) {
  case llvm::DK_OptimizationRemark:
    if (!cast<DiagnosticInfoOptimizationRemark>(DI).isEnabled())
      return;
    break;
  case llvm::DK_OptimizationRemarkMissed:
    if (!cast<DiagnosticInfoOptimizationRemarkMissed>(DI).isEnabled())
      return;
    break;
  case llvm::DK_OptimizationRemarkAnalysis:
    if (!cast<DiagnosticInfoOptimizationRemarkAnalysis>(DI).isEnabled())
      return;
    break;
  default:
    break;
  }

  // Otherwise, print the message with a prefix based on the severity.
  std::string MsgStorage;
  raw_string_ostream Stream(MsgStorage);
  DiagnosticPrinterRawOStream DP(Stream);
  DI.print(DP);
  Stream.flush();
  switch (DI.getSeverity()) {
  case DS_Error:
    errs() << "error: " << MsgStorage << "\n";
    exit(1);
  case DS_Warning:
    errs() << "warning: " << MsgStorage << "\n";
    break;
  case DS_Remark:
    errs() << "remark: " << MsgStorage << "\n";
    break;
  case DS_Note:
    errs() << "note: " << MsgStorage << "\n";
    break;
  }
}

void LLVMContext::emitError(unsigned LocCookie, const Twine &ErrorStr) {
  diagnose(DiagnosticInfoInlineAsm(LocCookie, ErrorStr));
}

//===----------------------------------------------------------------------===//
// Metadata Kind Uniquing
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
/// isValidName - Return true if Name is a valid custom metadata handler name.
static bool isValidName(StringRef MDName) {
  if (MDName.empty())
    return false;

  if (!std::isalpha(static_cast<unsigned char>(MDName[0])))
    return false;

  for (StringRef::iterator I = MDName.begin() + 1, E = MDName.end(); I != E;
       ++I) {
    if (!std::isalnum(static_cast<unsigned char>(*I)) && *I != '_' &&
        *I != '-' && *I != '.')
      return false;
  }
  return true;
}
#endif

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
unsigned LLVMContext::getMDKindID(StringRef Name) const {
  assert(isValidName(Name) && "Invalid MDNode name");

  // If this is new, assign it its ID.
  return
    pImpl->CustomMDKindNames.GetOrCreateValue(
      Name, pImpl->CustomMDKindNames.size()).second;
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void LLVMContext::getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(pImpl->CustomMDKindNames.size());
  for (StringMap<unsigned>::const_iterator I = pImpl->CustomMDKindNames.begin(),
       E = pImpl->CustomMDKindNames.end(); I != E; ++I)
    Names[I->second] = I->first();
}
