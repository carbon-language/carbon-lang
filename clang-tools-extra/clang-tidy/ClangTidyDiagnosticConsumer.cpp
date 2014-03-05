//===--- tools/extra/clang-tidy/ClangTidyDiagnosticConsumer.cpp ----------=== //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements ClangTidyDiagnosticConsumer, ClangTidyMessage,
///  ClangTidyContext and ClangTidyError classes.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyDiagnosticConsumer.h"
#include "llvm/ADT/SmallString.h"

namespace clang {
namespace tidy {

ClangTidyMessage::ClangTidyMessage(StringRef Message) : Message(Message) {}

ClangTidyMessage::ClangTidyMessage(StringRef Message,
                                   const SourceManager &Sources,
                                   SourceLocation Loc)
    : Message(Message) {
  FilePath = Sources.getFilename(Loc);
  FileOffset = Sources.getFileOffset(Loc);
}

ClangTidyError::ClangTidyError(StringRef CheckName,
                               const ClangTidyMessage &Message)
    : CheckName(CheckName), Message(Message) {}

DiagnosticBuilder ClangTidyContext::diag(
    StringRef CheckName, SourceLocation Loc, StringRef Description,
    DiagnosticIDs::Level Level /* = DiagnosticIDs::Warning*/) {
  assert(Loc.isValid());
  bool Invalid;
  const char *CharacterData =
      DiagEngine->getSourceManager().getCharacterData(Loc, &Invalid);
  if (!Invalid) {
    const char *P = CharacterData;
    while (*P != '\0' && *P != '\r' && *P != '\n')
      ++P;
    StringRef RestOfLine(CharacterData, P - CharacterData + 1);
    // FIXME: Handle /\bNOLINT\b(\([^)]*\))?/ as cpplint.py does.
    if (RestOfLine.find("NOLINT") != StringRef::npos)
      Level = DiagnosticIDs::Ignored;
  }
  unsigned ID = DiagEngine->getDiagnosticIDs()->getCustomDiagID(
      Level, (Description + " [" + CheckName + "]").str());
  if (CheckNamesByDiagnosticID.count(ID) == 0)
    CheckNamesByDiagnosticID.insert(std::make_pair(ID, CheckName.str()));
  return DiagEngine->Report(Loc, ID);
}

void ClangTidyContext::setDiagnosticsEngine(DiagnosticsEngine *Engine) {
  DiagEngine = Engine;
}

void ClangTidyContext::setSourceManager(SourceManager *SourceMgr) {
  DiagEngine->setSourceManager(SourceMgr);
}

/// \brief Store a \c ClangTidyError.
void ClangTidyContext::storeError(const ClangTidyError &Error) {
  Errors->push_back(Error);
}

StringRef ClangTidyContext::getCheckName(unsigned DiagnosticID) const {
  llvm::DenseMap<unsigned, std::string>::const_iterator I =
      CheckNamesByDiagnosticID.find(DiagnosticID);
  if (I != CheckNamesByDiagnosticID.end())
    return I->second;
  return "";
}

ClangTidyDiagnosticConsumer::ClangTidyDiagnosticConsumer(ClangTidyContext &Ctx)
    : Context(Ctx), LastErrorRelatesToUserCode(false) {
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  Diags.reset(new DiagnosticsEngine(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs), &*DiagOpts, this,
      /*ShouldOwnClient=*/false));
  Context.setDiagnosticsEngine(Diags.get());
}

void ClangTidyDiagnosticConsumer::finalizeLastError() {
  if (!LastErrorRelatesToUserCode && !Errors.empty())
    Errors.pop_back();
  LastErrorRelatesToUserCode = false;
}

void ClangTidyDiagnosticConsumer::HandleDiagnostic(
    DiagnosticsEngine::Level DiagLevel, const Diagnostic &Info) {
  // FIXME: Deduplicate diagnostics.
  if (DiagLevel == DiagnosticsEngine::Note) {
    assert(!Errors.empty() &&
           "A diagnostic note can only be appended to a message.");
    Errors.back().Notes.push_back(getMessage(Info));
  } else {
    finalizeLastError();
    Errors.push_back(
        ClangTidyError(Context.getCheckName(Info.getID()), getMessage(Info)));
  }
  addFixes(Info, Errors.back());

  // Let argument parsing-related warnings through.
  if (!Info.getLocation().isValid() ||
      !Diags->getSourceManager().isInSystemHeader(Info.getLocation())) {
    LastErrorRelatesToUserCode = true;
  }
}

// Flushes the internal diagnostics buffer to the ClangTidyContext.
void ClangTidyDiagnosticConsumer::finish() {
  finalizeLastError();
  for (unsigned i = 0, e = Errors.size(); i != e; ++i)
    Context.storeError(Errors[i]);
  Errors.clear();
}

void ClangTidyDiagnosticConsumer::addFixes(const Diagnostic &Info,
                                           ClangTidyError &Error) {
  if (!Info.hasSourceManager())
    return;
  SourceManager &Sources = Info.getSourceManager();
  tooling::Replacements Replacements;
  for (unsigned i = 0, e = Info.getNumFixItHints(); i != e; ++i) {
    CharSourceRange Range = Info.getFixItHint(i).RemoveRange;
    assert(Range.getBegin().isValid() && Range.getEnd().isValid() &&
           "Invalid range in the fix-it hint.");
    assert(Range.getBegin().isFileID() && Range.getEnd().isFileID() &&
           "Only file locations supported in fix-it hints.");
    std::string Text = Info.getFixItHint(i).CodeToInsert;
    Error.Fix.insert(tooling::Replacement(Sources, Range, Text));
  }
}

ClangTidyMessage
ClangTidyDiagnosticConsumer::getMessage(const Diagnostic &Info) const {
  SmallString<100> Buf;
  Info.FormatDiagnostic(Buf);
  if (!Info.hasSourceManager()) {
    return ClangTidyMessage(Buf.str());
  }
  return ClangTidyMessage(Buf.str(), Info.getSourceManager(),
                          Info.getLocation());
}

} // namespace tidy
} // namespace clang
