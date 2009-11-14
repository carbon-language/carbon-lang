//===--- VerifyDiagnosticsClient.cpp - Verifying Diagnostic Client --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client, which buffers the diagnostic messages.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

VerifyDiagnosticsClient::VerifyDiagnosticsClient(Diagnostic &_Diags,
                                                 DiagnosticClient *_Primary)
  : Diags(_Diags), PrimaryClient(_Primary),
    Buffer(new TextDiagnosticBuffer()), CurrentPreprocessor(0), NumErrors(0) {
}

VerifyDiagnosticsClient::~VerifyDiagnosticsClient() {
  CheckDiagnostics();
}

// DiagnosticClient interface.

void VerifyDiagnosticsClient::BeginSourceFile(const LangOptions &LangOpts,
                                             const Preprocessor *PP) {
  // FIXME: Const hack, we screw up the preprocessor but in practice its ok
  // because it doesn't get reused. It would be better if we could make a copy
  // though.
  CurrentPreprocessor = const_cast<Preprocessor*>(PP);

  PrimaryClient->BeginSourceFile(LangOpts, PP);
}

void VerifyDiagnosticsClient::EndSourceFile() {
  CheckDiagnostics();

  PrimaryClient->EndSourceFile();

  CurrentPreprocessor = 0;
}

void VerifyDiagnosticsClient::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                              const DiagnosticInfo &Info) {
  // Send the diagnostic to the buffer, we will check it once we reach the end
  // of the source file (or are destructed).
  Buffer->HandleDiagnostic(DiagLevel, Info);
}

// FIXME: It would be nice to just get this from the primary diagnostic client
// or something.
bool VerifyDiagnosticsClient::HadErrors() {
  CheckDiagnostics();

  return NumErrors != 0;
}

//===----------------------------------------------------------------------===//
// Checking diagnostics implementation.
//===----------------------------------------------------------------------===//

typedef TextDiagnosticBuffer::DiagList DiagList;
typedef TextDiagnosticBuffer::const_iterator const_diag_iterator;

/// FindDiagnostics - Go through the comment and see if it indicates expected
/// diagnostics. If so, then put them in a diagnostic list.
///
static void FindDiagnostics(const char *CommentStart, unsigned CommentLen,
                            DiagList &ExpectedDiags,
                            Preprocessor &PP, SourceLocation Pos,
                            const char *ExpectedStr) {
  const char *CommentEnd = CommentStart+CommentLen;
  unsigned ExpectedStrLen = strlen(ExpectedStr);

  // Find all expected-foo diagnostics in the string and add them to
  // ExpectedDiags.
  while (CommentStart != CommentEnd) {
    CommentStart = std::find(CommentStart, CommentEnd, 'e');
    if (unsigned(CommentEnd-CommentStart) < ExpectedStrLen) return;

    // If this isn't expected-foo, ignore it.
    if (memcmp(CommentStart, ExpectedStr, ExpectedStrLen)) {
      ++CommentStart;
      continue;
    }

    CommentStart += ExpectedStrLen;

    // Skip whitespace.
    while (CommentStart != CommentEnd &&
           isspace(CommentStart[0]))
      ++CommentStart;

    // Default, if we find the '{' now, is 1 time.
    int Times = 1;
    int Temp = 0;
    // In extended syntax, there could be a digit now.
    while (CommentStart != CommentEnd &&
           CommentStart[0] >= '0' && CommentStart[0] <= '9') {
      Temp *= 10;
      Temp += CommentStart[0] - '0';
      ++CommentStart;
    }
    if (Temp > 0)
      Times = Temp;

    // Skip whitespace again.
    while (CommentStart != CommentEnd &&
           isspace(CommentStart[0]))
      ++CommentStart;

    // We should have a {{ now.
    if (CommentEnd-CommentStart < 2 ||
        CommentStart[0] != '{' || CommentStart[1] != '{') {
      if (std::find(CommentStart, CommentEnd, '{') != CommentEnd)
        PP.Diag(Pos, diag::err_verify_bogus_characters);
      else
        PP.Diag(Pos, diag::err_verify_missing_start);
      return;
    }
    CommentStart += 2;

    // Find the }}.
    const char *ExpectedEnd = CommentStart;
    while (1) {
      ExpectedEnd = std::find(ExpectedEnd, CommentEnd, '}');
      if (CommentEnd-ExpectedEnd < 2) {
        PP.Diag(Pos, diag::err_verify_missing_end);
        return;
      }

      if (ExpectedEnd[1] == '}')
        break;

      ++ExpectedEnd;  // Skip over singular }'s
    }

    std::string Msg(CommentStart, ExpectedEnd);
    std::string::size_type FindPos;
    while ((FindPos = Msg.find("\\n")) != std::string::npos)
      Msg.replace(FindPos, 2, "\n");
    // Add is possibly multiple times.
    for (int i = 0; i < Times; ++i)
      ExpectedDiags.push_back(std::make_pair(Pos, Msg));

    CommentStart = ExpectedEnd;
  }
}

/// FindExpectedDiags - Lex the main source file to find all of the
//   expected errors and warnings.
static void FindExpectedDiags(Preprocessor &PP,
                              DiagList &ExpectedErrors,
                              DiagList &ExpectedWarnings,
                              DiagList &ExpectedNotes) {
  // Create a raw lexer to pull all the comments out of the main file.  We don't
  // want to look in #include'd headers for expected-error strings.
  FileID FID = PP.getSourceManager().getMainFileID();
  if (PP.getSourceManager().getMainFileID().isInvalid())
    return;

  // Create a lexer to lex all the tokens of the main file in raw mode.
  Lexer RawLex(FID, PP.getSourceManager(), PP.getLangOptions());

  // Return comments as tokens, this is how we find expected diagnostics.
  RawLex.SetCommentRetentionState(true);

  Token Tok;
  Tok.setKind(tok::comment);
  while (Tok.isNot(tok::eof)) {
    RawLex.Lex(Tok);
    if (!Tok.is(tok::comment)) continue;

    std::string Comment = PP.getSpelling(Tok);
    if (Comment.empty()) continue;

    // Find all expected errors.
    FindDiagnostics(&Comment[0], Comment.size(), ExpectedErrors, PP,
                    Tok.getLocation(), "expected-error");

    // Find all expected warnings.
    FindDiagnostics(&Comment[0], Comment.size(), ExpectedWarnings, PP,
                    Tok.getLocation(), "expected-warning");

    // Find all expected notes.
    FindDiagnostics(&Comment[0], Comment.size(), ExpectedNotes, PP,
                    Tok.getLocation(), "expected-note");
  };
}

/// PrintProblem - This takes a diagnostic map of the delta between expected and
/// seen diagnostics. If there's anything in it, then something unexpected
/// happened. Print the map out in a nice format and return "true". If the map
/// is empty and we're not going to print things, then return "false".
///
static unsigned PrintProblem(Diagnostic &Diags, SourceManager *SourceMgr,
                             const_diag_iterator diag_begin,
                             const_diag_iterator diag_end,
                             const char *Kind, bool Expected) {
  if (diag_begin == diag_end) return 0;

  llvm::SmallString<256> Fmt;
  llvm::raw_svector_ostream OS(Fmt);
  for (const_diag_iterator I = diag_begin, E = diag_end; I != E; ++I) {
    if (I->first.isInvalid() || !SourceMgr)
      OS << "\n  (frontend)";
    else
      OS << "\n  Line " << SourceMgr->getInstantiationLineNumber(I->first);
    OS << ": " << I->second;
  }

  Diags.Report(diag::err_verify_inconsistent_diags)
    << Kind << !Expected << OS.str();
  return std::distance(diag_begin, diag_end);
}

/// CompareDiagLists - Compare two diagnostic lists and return the difference
/// between them.
///
static unsigned CompareDiagLists(Diagnostic &Diags,
                                 SourceManager &SourceMgr,
                                 const_diag_iterator d1_begin,
                                 const_diag_iterator d1_end,
                                 const_diag_iterator d2_begin,
                                 const_diag_iterator d2_end,
                                 const char *Label) {
  DiagList LeftOnly;
  DiagList Left(d1_begin, d1_end);
  DiagList Right(d2_begin, d2_end);

  for (const_diag_iterator I = Left.begin(), E = Left.end(); I != E; ++I) {
    unsigned LineNo1 = SourceMgr.getInstantiationLineNumber(I->first);
    const std::string &Diag1 = I->second;

    DiagList::iterator II, IE;
    for (II = Right.begin(), IE = Right.end(); II != IE; ++II) {
      unsigned LineNo2 = SourceMgr.getInstantiationLineNumber(II->first);
      if (LineNo1 != LineNo2) continue;

      const std::string &Diag2 = II->second;
      if (Diag2.find(Diag1) != std::string::npos ||
          Diag1.find(Diag2) != std::string::npos) {
        break;
      }
    }
    if (II == IE) {
      // Not found.
      LeftOnly.push_back(*I);
    } else {
      // Found. The same cannot be found twice.
      Right.erase(II);
    }
  }
  // Now all that's left in Right are those that were not matched.

  return (PrintProblem(Diags, &SourceMgr,
                      LeftOnly.begin(), LeftOnly.end(), Label, true) +
          PrintProblem(Diags, &SourceMgr,
                       Right.begin(), Right.end(), Label, false));
}

/// CheckResults - This compares the expected results to those that
/// were actually reported. It emits any discrepencies. Return "true" if there
/// were problems. Return "false" otherwise.
///
static unsigned CheckResults(Diagnostic &Diags, SourceManager &SourceMgr,
                             const TextDiagnosticBuffer &Buffer,
                             const DiagList &ExpectedErrors,
                             const DiagList &ExpectedWarnings,
                             const DiagList &ExpectedNotes) {
  // We want to capture the delta between what was expected and what was
  // seen.
  //
  //   Expected \ Seen - set expected but not seen
  //   Seen \ Expected - set seen but not expected
  unsigned NumProblems = 0;

  // See if there are error mismatches.
  NumProblems += CompareDiagLists(Diags, SourceMgr,
                                  ExpectedErrors.begin(), ExpectedErrors.end(),
                                  Buffer.err_begin(), Buffer.err_end(),
                                  "error");

  // See if there are warning mismatches.
  NumProblems += CompareDiagLists(Diags, SourceMgr,
                                  ExpectedWarnings.begin(),
                                  ExpectedWarnings.end(),
                                  Buffer.warn_begin(), Buffer.warn_end(),
                                  "warning");

  // See if there are note mismatches.
  NumProblems += CompareDiagLists(Diags, SourceMgr,
                                  ExpectedNotes.begin(),
                                  ExpectedNotes.end(),
                                  Buffer.note_begin(), Buffer.note_end(),
                                  "note");

  return NumProblems;
}


void VerifyDiagnosticsClient::CheckDiagnostics() {
  DiagList ExpectedErrors, ExpectedWarnings, ExpectedNotes;

  // Ensure any diagnostics go to the primary client.
  DiagnosticClient *CurClient = Diags.getClient();
  Diags.setClient(PrimaryClient.get());

  // If we have a preprocessor, scan the source for expected diagnostic
  // markers. If not then any diagnostics are unexpected.
  if (CurrentPreprocessor) {
    FindExpectedDiags(*CurrentPreprocessor, ExpectedErrors, ExpectedWarnings,
                      ExpectedNotes);

    // Check that the expected diagnostics occurred.
    NumErrors += CheckResults(Diags, CurrentPreprocessor->getSourceManager(),
                              *Buffer,
                              ExpectedErrors, ExpectedWarnings, ExpectedNotes);
  } else {
    NumErrors += (PrintProblem(Diags, 0,
                               Buffer->err_begin(), Buffer->err_end(),
                               "error", false) +
                  PrintProblem(Diags, 0,
                               Buffer->warn_begin(), Buffer->warn_end(),
                               "warn", false) +
                  PrintProblem(Diags, 0,
                               Buffer->note_begin(), Buffer->note_end(),
                               "note", false));
  }

  Diags.setClient(CurClient);

  // Reset the buffer, we have processed all the diagnostics in it.
  Buffer.reset(new TextDiagnosticBuffer());
}
