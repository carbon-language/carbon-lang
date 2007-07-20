//===--- DiagChecker.cpp - Diagnostic Checking Functions ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Process the input files and check that the diagnostic messages are expected.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "ASTStreamers.h"
#include "TextDiagnosticBuffer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang;

typedef TextDiagnosticBuffer::DiagList DiagList;
typedef TextDiagnosticBuffer::const_iterator const_diag_iterator;

// USING THE DIAGNOSTIC CHECKER:
//
// Indicating that a line expects an error or a warning is simple. Put a comment
// on the line that has the diagnostic, use "expected-{error,warning}" to tag
// if it's an expected error or warning, and place the expected text between {{
// and }} markers. The full text doesn't have to be included, only enough to
// ensure that the correct diagnostic was emitted.
//
// Here's an example:
//
//   int A = B; // expected-error {{use of undeclared identifier 'B'}}
//
// You can place as many diagnostics on one line as you wish. To make the code
// more readable, you can use slash-newline to separate out the diagnostics.

static const char * const ExpectedErrStr = "expected-error";
static const char * const ExpectedWarnStr = "expected-warning";

/// FindDiagnostics - Go through the comment and see if it indicates expected
/// diagnostics. If so, then put them in a diagnostic list.
/// 
static void FindDiagnostics(const std::string &Comment,
                            DiagList &ExpectedDiags,
                            SourceManager &SourceMgr,
                            SourceLocation Pos,
                            const char * const ExpectedStr) {
  // Find all expected diagnostics
  typedef std::string::size_type size_type;
  size_type ColNo = std::string::npos;

  for (;;) {
    ColNo = Comment.find(ExpectedStr, ColNo);
    if (ColNo == std::string::npos) break;

    size_type OpenDiag = Comment.find_first_of("{{", ColNo);

    if (OpenDiag == std::string::npos) {
      fprintf(stderr,
              "oops:%d: Cannot find beginning of expected error string\n",
              SourceMgr.getLogicalLineNumber(Pos));
      break;
    }

    OpenDiag += 2;
    size_type CloseDiag = Comment.find_first_of("}}", OpenDiag);

    if (CloseDiag == std::string::npos) {
      fprintf(stderr,
              "oops:%d: Cannot find end of expected error string\n",
              SourceMgr.getLogicalLineNumber(Pos));
      break;
    }

    std::string Msg(Comment.substr(OpenDiag, CloseDiag - OpenDiag));
    ExpectedDiags.push_back(std::make_pair(Pos, Msg));
    ColNo = CloseDiag + 2;
  }
}

/// FindExpectedDiags - Lex the file to finds all of the expected errors and
/// warnings.
static void FindExpectedDiags(Preprocessor &PP, unsigned MainFileID,
                              DiagList &ExpectedErrors,
                              DiagList &ExpectedWarnings) {
  // Return comments as tokens, this is how we find expected diagnostics.
  PP.SetCommentRetentionState(true, true);

  // Enter the cave.
  PP.EnterSourceFile(MainFileID, 0, true);

  LexerToken Tok;
  do {
    PP.Lex(Tok);

    if (Tok.getKind() == tok::comment) {
      std::string Comment = PP.getSpelling(Tok);

      // Find all expected errors
      FindDiagnostics(Comment, ExpectedErrors,PP.getSourceManager(),
                      Tok.getLocation(), ExpectedErrStr);

      // Find all expected warnings
      FindDiagnostics(Comment, ExpectedWarnings, PP.getSourceManager(),
                      Tok.getLocation(), ExpectedWarnStr);
    }
  } while (Tok.getKind() != tok::eof);

  PP.SetCommentRetentionState(false, false);
}

/// PrintProblem - This takes a diagnostic map of the delta between expected and
/// seen diagnostics. If there's anything in it, then something unexpected
/// happened. Print the map out in a nice format and return "true". If the map
/// is empty and we're not going to print things, then return "false".
/// 
static bool PrintProblem(SourceManager &SourceMgr,
                         const_diag_iterator diag_begin,
                         const_diag_iterator diag_end,
                         const char *Msg) {
  if (diag_begin == diag_end) return false;

  fprintf(stderr, "%s\n", Msg);

  for (const_diag_iterator I = diag_begin, E = diag_end; I != E; ++I)
    fprintf(stderr, "  Line %d: %s\n",
            SourceMgr.getLogicalLineNumber(I->first),
            I->second.c_str());

  return true;
}

/// CompareDiagLists - Compare two diangnostic lists and return the difference
/// between them.
/// 
static bool CompareDiagLists(SourceManager &SourceMgr,
                             const_diag_iterator d1_begin,
                             const_diag_iterator d1_end,
                             const_diag_iterator d2_begin,
                             const_diag_iterator d2_end,
                             const char *Msg) {
  DiagList DiffList;

  for (const_diag_iterator I = d1_begin, E = d1_end; I != E; ++I) {
    unsigned LineNo1 = SourceMgr.getLogicalLineNumber(I->first);
    const std::string &Diag1 = I->second;
    bool Found = false;

    for (const_diag_iterator II = d2_begin, IE = d2_end; II != IE; ++II) {
      unsigned LineNo2 = SourceMgr.getLogicalLineNumber(II->first);
      if (LineNo1 != LineNo2) continue;

      const std::string &Diag2 = II->second;
      if (Diag2.find(Diag1) != std::string::npos ||
          Diag1.find(Diag2) != std::string::npos) {
        Found = true;
        break;
      }
    }

    if (!Found)
      DiffList.push_back(std::make_pair(I->first, Diag1));
  }

  return PrintProblem(SourceMgr, DiffList.begin(), DiffList.end(), Msg);
}

/// CheckResults - This compares the expected results to those that
/// were actually reported. It emits any discrepencies. Return "true" if there
/// were problems. Return "false" otherwise.
/// 
static bool CheckResults(Preprocessor &PP,
                         const DiagList &ExpectedErrors,
                         const DiagList &ExpectedWarnings) {
  const TextDiagnosticBuffer &Diags =
    static_cast<const TextDiagnosticBuffer&>(PP.getDiagnostics().getClient());
  SourceManager &SourceMgr = PP.getSourceManager();

  // We want to capture the delta between what was expected and what was
  // seen.
  //
  //   Expected \ Seen - set expected but not seen
  //   Seen \ Expected - set seen but not expected
  bool HadProblem = false;

  // See if there were errors that were expected but not seen.
  HadProblem |= CompareDiagLists(SourceMgr,
                                 ExpectedErrors.begin(), ExpectedErrors.end(),
                                 Diags.err_begin(), Diags.err_end(),
                                 "Errors expected but not seen:");

  // See if there were errors that were seen but not expected.
  HadProblem |= CompareDiagLists(SourceMgr,
                                 Diags.err_begin(), Diags.err_end(),
                                 ExpectedErrors.begin(), ExpectedErrors.end(),
                                 "Errors seen but not expected:");

  // See if there were warnings that were expected but not seen.
  HadProblem |= CompareDiagLists(SourceMgr,
                                 ExpectedWarnings.begin(),
                                 ExpectedWarnings.end(),
                                 Diags.warn_begin(), Diags.warn_end(),
                                 "Warnings expected but not seen:");

  // See if there were warnings that were seen but not expected.
  HadProblem |= CompareDiagLists(SourceMgr,
                                 Diags.warn_begin(), Diags.warn_end(),
                                 ExpectedWarnings.begin(),
                                 ExpectedWarnings.end(),
                                 "Warnings seen but not expected:");

  return HadProblem;
}

/// CheckDiagnostics - Implement the -parse-ast-check diagnostic verifier.
bool clang::CheckDiagnostics(Preprocessor &PP, unsigned MainFileID) {
  // Gather the set of expected diagnostics.
  DiagList ExpectedErrors, ExpectedWarnings;
  FindExpectedDiags(PP, MainFileID, ExpectedErrors, ExpectedWarnings);
  
  // Parse the specified input file.
  BuildASTs(PP, MainFileID, false);

  // Check that the expected diagnostics occurred.
  return CheckResults(PP, ExpectedErrors, ExpectedWarnings);
}


