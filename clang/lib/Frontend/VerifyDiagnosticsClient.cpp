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
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Compiler.h"
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

namespace {

/// Directive - Abstract class representing a parsed verify directive.
///
class Directive {
public:
  static Directive* Create(bool RegexKind, const SourceLocation &Location,
                           const std::string &Text, unsigned Count);
public:
  SourceLocation Location;
  const std::string Text;
  unsigned Count;

  virtual ~Directive() { }

  // Returns true if directive text is valid.
  // Otherwise returns false and populates E.
  virtual bool isValid(std::string &Error) = 0;

  // Returns true on match.
  virtual bool Match(const std::string &S) = 0;

protected:
  Directive(const SourceLocation &Location, const std::string &Text,
            unsigned Count)
    : Location(Location), Text(Text), Count(Count) { }

private:
  Directive(const Directive&) ATTRIBUTE_UNUSED; // DO NOT IMPLEMENT
  void operator=(const Directive&) ATTRIBUTE_UNUSED; // DO NOT IMPLEMENT
};

/// StandardDirective - Directive with string matching.
///
class StandardDirective : public Directive {
public:
  StandardDirective(const SourceLocation &Location, const std::string &Text,
                    unsigned Count)
    : Directive(Location, Text, Count) { }

  virtual bool isValid(std::string &Error) {
    // all strings are considered valid; even empty ones
    return true;
  }

  virtual bool Match(const std::string &S) {
    return S.find(Text) != std::string::npos ||
           Text.find(S) != std::string::npos;
  }
};

/// RegexDirective - Directive with regular-expression matching.
///
class RegexDirective : public Directive {
public:
  RegexDirective(const SourceLocation &Location, const std::string &Text,
                 unsigned Count)
    : Directive(Location, Text, Count), Regex(Text) { }

  virtual bool isValid(std::string &Error) {
    if (Regex.isValid(Error))
      return true;
    return false;
  }

  virtual bool Match(const std::string &S) {
    return Regex.match(S);
  }

private:
  llvm::Regex Regex;
};

typedef std::vector<Directive*> DirectiveList;

/// ExpectedData - owns directive objects and deletes on destructor.
///
struct ExpectedData {
  DirectiveList Errors;
  DirectiveList Warnings;
  DirectiveList Notes;

  ~ExpectedData() {
    DirectiveList* Lists[] = { &Errors, &Warnings, &Notes, 0 };
    for (DirectiveList **PL = Lists; *PL; ++PL) {
      DirectiveList * const L = *PL;
      for (DirectiveList::iterator I = L->begin(), E = L->end(); I != E; ++I)
        delete *I;
    }
  }
};  

class ParseHelper
{
public:
  ParseHelper(const char *Begin, const char *End)
    : Begin(Begin), End(End), C(Begin), P(Begin), PEnd(NULL) { }

  // Return true if string literal is next.
  bool Next(const std::string &S) {
    std::string::size_type LEN = S.length();
    P = C;
    PEnd = C + LEN;
    if (PEnd > End)
      return false;
    return !memcmp(P, S.c_str(), LEN);
  }

  // Return true if number is next.
  // Output N only if number is next.
  bool Next(unsigned &N) {
    unsigned TMP = 0;
    P = C;
    for (; P < End && P[0] >= '0' && P[0] <= '9'; ++P) {
      TMP *= 10;
      TMP += P[0] - '0';
    }
    if (P == C)
      return false;
    PEnd = P;
    N = TMP;
    return true;
  }

  // Return true if string literal is found.
  // When true, P marks begin-position of S in content.
  bool Search(const std::string &S) {
    P = std::search(C, End, S.begin(), S.end());
    PEnd = P + S.length();
    return P != End;
  }

  // Advance 1-past previous next/search.
  // Behavior is undefined if previous next/search failed.
  bool Advance() {
    C = PEnd;
    return C < End;
  }

  // Skip zero or more whitespace.
  void SkipWhitespace() {
    for (; C < End && isspace(*C); ++C)
      ;
  }

  // Return true if EOF reached.
  bool Done() {
    return !(C < End);
  }

  const char * const Begin; // beginning of expected content
  const char * const End;   // end of expected content (1-past)
  const char *C;            // position of next char in content
  const char *P;

private:
  const char *PEnd; // previous next/search subject end (1-past)
};

} // namespace anonymous

/// ParseDirective - Go through the comment and see if it indicates expected
/// diagnostics. If so, then put them in the appropriate directive list.
///
static void ParseDirective(const char *CommentStart, unsigned CommentLen,
                           ExpectedData &ED, Preprocessor &PP,
                           SourceLocation Pos) {
  // A single comment may contain multiple directives.
  for (ParseHelper PH(CommentStart, CommentStart+CommentLen); !PH.Done();) {
    // search for token: expected
    if (!PH.Search("expected"))
      break;
    PH.Advance();

    // next token: -
    if (!PH.Next("-"))
      continue;
    PH.Advance();

    // next token: { error | warning | note }
    DirectiveList* DL = NULL;
    if (PH.Next("error"))
      DL = &ED.Errors;
    else if (PH.Next("warning"))
      DL = &ED.Warnings;
    else if (PH.Next("note"))
      DL = &ED.Notes;
    else
      continue;
    PH.Advance();

    // default directive kind
    bool RegexKind = false;
    const char* KindStr = "string";

    // next optional token: -
    if (PH.Next("-re")) {
      PH.Advance();
      RegexKind = true;
      KindStr = "regex";
    }

    // skip optional whitespace
    PH.SkipWhitespace();

    // next optional token: positive integer
    unsigned Count = 1;
    if (PH.Next(Count))
      PH.Advance();

    // skip optional whitespace
    PH.SkipWhitespace();

    // next token: {{
    if (!PH.Next("{{")) {
      PP.Diag(Pos.getFileLocWithOffset(PH.C-PH.Begin),
              diag::err_verify_missing_start) << KindStr;
      continue;
    }
    PH.Advance();
    const char* const ContentBegin = PH.C; // mark content begin

    // search for token: }}
    if (!PH.Search("}}")) {
      PP.Diag(Pos.getFileLocWithOffset(PH.C-PH.Begin),
              diag::err_verify_missing_end) << KindStr;
      continue;
    }
    const char* const ContentEnd = PH.P; // mark content end
    PH.Advance();

    // build directive text; convert \n to newlines
    std::string Text;
    llvm::StringRef NewlineStr = "\\n";
    llvm::StringRef Content(ContentBegin, ContentEnd-ContentBegin);
    size_t CPos = 0;
    size_t FPos;
    while ((FPos = Content.find(NewlineStr, CPos)) != llvm::StringRef::npos) {
      Text += Content.substr(CPos, FPos-CPos);
      Text += '\n';
      CPos = FPos + NewlineStr.size();
    }
    if (Text.empty())
      Text.assign(ContentBegin, ContentEnd);

    // construct new directive
    Directive *D = Directive::Create(RegexKind, Pos, Text, Count);
    std::string Error;
    if (D->isValid(Error))
      DL->push_back(D);
    else {
      PP.Diag(Pos.getFileLocWithOffset(ContentBegin-PH.Begin),
              diag::err_verify_invalid_content)
        << KindStr << Error;
    }
  }
}

/// FindExpectedDiags - Lex the main source file to find all of the
//   expected errors and warnings.
static void FindExpectedDiags(Preprocessor &PP, ExpectedData &ED) {
  // Create a raw lexer to pull all the comments out of the main file.  We don't
  // want to look in #include'd headers for expected-error strings.
  SourceManager &SM = PP.getSourceManager();
  FileID FID = SM.getMainFileID();
  if (SM.getMainFileID().isInvalid())
    return;

  // Create a lexer to lex all the tokens of the main file in raw mode.
  const llvm::MemoryBuffer *FromFile = SM.getBuffer(FID);
  Lexer RawLex(FID, FromFile, SM, PP.getLangOptions());

  // Return comments as tokens, this is how we find expected diagnostics.
  RawLex.SetCommentRetentionState(true);

  Token Tok;
  Tok.setKind(tok::comment);
  while (Tok.isNot(tok::eof)) {
    RawLex.Lex(Tok);
    if (!Tok.is(tok::comment)) continue;

    std::string Comment = PP.getSpelling(Tok);
    if (Comment.empty()) continue;

    // Find all expected errors/warnings/notes.
    ParseDirective(&Comment[0], Comment.size(), ED, PP, Tok.getLocation());
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

static unsigned PrintProblem(Diagnostic &Diags, SourceManager *SourceMgr,
                             DirectiveList &DL, const char *Kind,
                             bool Expected) {
  if (DL.empty())
    return 0;

  llvm::SmallString<256> Fmt;
  llvm::raw_svector_ostream OS(Fmt);
  for (DirectiveList::iterator I = DL.begin(), E = DL.end(); I != E; ++I) {
    Directive& D = **I;
    if (D.Location.isInvalid() || !SourceMgr)
      OS << "\n  (frontend)";
    else
      OS << "\n  Line " << SourceMgr->getInstantiationLineNumber(D.Location);
    OS << ": " << D.Text;
  }

  Diags.Report(diag::err_verify_inconsistent_diags)
    << Kind << !Expected << OS.str();
  return DL.size();
}

/// CheckLists - Compare expected to seen diagnostic lists and return the
/// the difference between them.
///
static unsigned CheckLists(Diagnostic &Diags, SourceManager &SourceMgr,
                           const char *Label,
                           DirectiveList &Left,
                           const_diag_iterator d2_begin,
                           const_diag_iterator d2_end) {
  DirectiveList LeftOnly;
  DiagList Right(d2_begin, d2_end);

  for (DirectiveList::iterator I = Left.begin(), E = Left.end(); I != E; ++I) {
    Directive& D = **I;
    unsigned LineNo1 = SourceMgr.getInstantiationLineNumber(D.Location);

    for (unsigned i = 0; i < D.Count; ++i) {
      DiagList::iterator II, IE;
      for (II = Right.begin(), IE = Right.end(); II != IE; ++II) {
        unsigned LineNo2 = SourceMgr.getInstantiationLineNumber(II->first);
        if (LineNo1 != LineNo2)
          continue;

        const std::string &RightText = II->second;
        if (D.Match(RightText))
          break;
      }
      if (II == IE) {
        // Not found.
        LeftOnly.push_back(*I);
      } else {
        // Found. The same cannot be found twice.
        Right.erase(II);
      }
    }
  }
  // Now all that's left in Right are those that were not matched.

  return (PrintProblem(Diags, &SourceMgr, LeftOnly, Label, true) +
          PrintProblem(Diags, &SourceMgr, Right.begin(), Right.end(),
                       Label, false));
}

/// CheckResults - This compares the expected results to those that
/// were actually reported. It emits any discrepencies. Return "true" if there
/// were problems. Return "false" otherwise.
///
static unsigned CheckResults(Diagnostic &Diags, SourceManager &SourceMgr,
                             const TextDiagnosticBuffer &Buffer,
                             ExpectedData &ED) {
  // We want to capture the delta between what was expected and what was
  // seen.
  //
  //   Expected \ Seen - set expected but not seen
  //   Seen \ Expected - set seen but not expected
  unsigned NumProblems = 0;

  // See if there are error mismatches.
  NumProblems += CheckLists(Diags, SourceMgr, "error", ED.Errors,
                            Buffer.err_begin(), Buffer.err_end());

  // See if there are warning mismatches.
  NumProblems += CheckLists(Diags, SourceMgr, "warning", ED.Warnings,
                            Buffer.warn_begin(), Buffer.warn_end());

  // See if there are note mismatches.
  NumProblems += CheckLists(Diags, SourceMgr, "note", ED.Notes,
                            Buffer.note_begin(), Buffer.note_end());

  return NumProblems;
}

void VerifyDiagnosticsClient::CheckDiagnostics() {
  ExpectedData ED;

  // Ensure any diagnostics go to the primary client.
  DiagnosticClient *CurClient = Diags.getClient();
  Diags.setClient(PrimaryClient.get());

  // If we have a preprocessor, scan the source for expected diagnostic
  // markers. If not then any diagnostics are unexpected.
  if (CurrentPreprocessor) {
    FindExpectedDiags(*CurrentPreprocessor, ED);

    // Check that the expected diagnostics occurred.
    NumErrors += CheckResults(Diags, CurrentPreprocessor->getSourceManager(),
                              *Buffer, ED);
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

Directive* Directive::Create(bool RegexKind, const SourceLocation &Location,
                             const std::string &Text, unsigned Count) {
  if (RegexKind)
    return new RegexDirective(Location, Text, Count);
  return new StandardDirective(Location, Text, Count);
}
