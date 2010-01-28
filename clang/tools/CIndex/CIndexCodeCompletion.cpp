//===- CIndexCodeCompletion.cpp - Code Completion API hooks ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Clang-C Source Indexing library hooks for
// code completion.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CIndexDiagnostic.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Program.h"

using namespace clang;

extern "C" {

enum CXCompletionChunkKind
clang_getCompletionChunkKind(CXCompletionString completion_string,
                             unsigned chunk_number) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  if (!CCStr || chunk_number >= CCStr->size())
    return CXCompletionChunk_Text;

  switch ((*CCStr)[chunk_number].Kind) {
  case CodeCompletionString::CK_TypedText:
    return CXCompletionChunk_TypedText;
  case CodeCompletionString::CK_Text:
    return CXCompletionChunk_Text;
  case CodeCompletionString::CK_Optional:
    return CXCompletionChunk_Optional;
  case CodeCompletionString::CK_Placeholder:
    return CXCompletionChunk_Placeholder;
  case CodeCompletionString::CK_Informative:
    return CXCompletionChunk_Informative;
  case CodeCompletionString::CK_ResultType:
    return CXCompletionChunk_ResultType;
  case CodeCompletionString::CK_CurrentParameter:
    return CXCompletionChunk_CurrentParameter;
  case CodeCompletionString::CK_LeftParen:
    return CXCompletionChunk_LeftParen;
  case CodeCompletionString::CK_RightParen:
    return CXCompletionChunk_RightParen;
  case CodeCompletionString::CK_LeftBracket:
    return CXCompletionChunk_LeftBracket;
  case CodeCompletionString::CK_RightBracket:
    return CXCompletionChunk_RightBracket;
  case CodeCompletionString::CK_LeftBrace:
    return CXCompletionChunk_LeftBrace;
  case CodeCompletionString::CK_RightBrace:
    return CXCompletionChunk_RightBrace;
  case CodeCompletionString::CK_LeftAngle:
    return CXCompletionChunk_LeftAngle;
  case CodeCompletionString::CK_RightAngle:
    return CXCompletionChunk_RightAngle;
  case CodeCompletionString::CK_Comma:
    return CXCompletionChunk_Comma;
  case CodeCompletionString::CK_Colon:
    return CXCompletionChunk_Colon;
  case CodeCompletionString::CK_SemiColon:
    return CXCompletionChunk_SemiColon;
  case CodeCompletionString::CK_Equal:
    return CXCompletionChunk_Equal;
  case CodeCompletionString::CK_HorizontalSpace:
    return CXCompletionChunk_HorizontalSpace;
  case CodeCompletionString::CK_VerticalSpace:
    return CXCompletionChunk_VerticalSpace;
  }

  // Should be unreachable, but let's be careful.
  return CXCompletionChunk_Text;
}

const char *clang_getCompletionChunkText(CXCompletionString completion_string,
                                         unsigned chunk_number) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  if (!CCStr || chunk_number >= CCStr->size())
    return 0;

  switch ((*CCStr)[chunk_number].Kind) {
  case CodeCompletionString::CK_TypedText:
  case CodeCompletionString::CK_Text:
  case CodeCompletionString::CK_Placeholder:
  case CodeCompletionString::CK_CurrentParameter:
  case CodeCompletionString::CK_Informative:
  case CodeCompletionString::CK_LeftParen:
  case CodeCompletionString::CK_RightParen:
  case CodeCompletionString::CK_LeftBracket:
  case CodeCompletionString::CK_RightBracket:
  case CodeCompletionString::CK_LeftBrace:
  case CodeCompletionString::CK_RightBrace:
  case CodeCompletionString::CK_LeftAngle:
  case CodeCompletionString::CK_RightAngle:
  case CodeCompletionString::CK_Comma:
  case CodeCompletionString::CK_ResultType:
  case CodeCompletionString::CK_Colon:
  case CodeCompletionString::CK_SemiColon:
  case CodeCompletionString::CK_Equal:
  case CodeCompletionString::CK_HorizontalSpace:
  case CodeCompletionString::CK_VerticalSpace:
    return (*CCStr)[chunk_number].Text;

  case CodeCompletionString::CK_Optional:
    // Note: treated as an empty text block.
    return "";
  }

  // Should be unreachable, but let's be careful.
  return 0;
}

CXCompletionString
clang_getCompletionChunkCompletionString(CXCompletionString completion_string,
                                         unsigned chunk_number) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  if (!CCStr || chunk_number >= CCStr->size())
    return 0;

  switch ((*CCStr)[chunk_number].Kind) {
  case CodeCompletionString::CK_TypedText:
  case CodeCompletionString::CK_Text:
  case CodeCompletionString::CK_Placeholder:
  case CodeCompletionString::CK_CurrentParameter:
  case CodeCompletionString::CK_Informative:
  case CodeCompletionString::CK_LeftParen:
  case CodeCompletionString::CK_RightParen:
  case CodeCompletionString::CK_LeftBracket:
  case CodeCompletionString::CK_RightBracket:
  case CodeCompletionString::CK_LeftBrace:
  case CodeCompletionString::CK_RightBrace:
  case CodeCompletionString::CK_LeftAngle:
  case CodeCompletionString::CK_RightAngle:
  case CodeCompletionString::CK_Comma:
  case CodeCompletionString::CK_ResultType:
  case CodeCompletionString::CK_Colon:
  case CodeCompletionString::CK_SemiColon:
  case CodeCompletionString::CK_Equal:
  case CodeCompletionString::CK_HorizontalSpace:
  case CodeCompletionString::CK_VerticalSpace:
    return 0;

  case CodeCompletionString::CK_Optional:
    // Note: treated as an empty text block.
    return (*CCStr)[chunk_number].Optional;
  }

  // Should be unreachable, but let's be careful.
  return 0;
}

unsigned clang_getNumCompletionChunks(CXCompletionString completion_string) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  return CCStr? CCStr->size() : 0;
}

static bool ReadUnsigned(const char *&Memory, const char *MemoryEnd,
                         unsigned &Value) {
  if (Memory + sizeof(unsigned) > MemoryEnd)
    return true;

  memmove(&Value, Memory, sizeof(unsigned));
  Memory += sizeof(unsigned);
  return false;
}

/// \brief The CXCodeCompleteResults structure we allocate internally;
/// the client only sees the initial CXCodeCompleteResults structure.
struct AllocatedCXCodeCompleteResults : public CXCodeCompleteResults {
  /// \brief The memory buffer from which we parsed the results. We
  /// retain this buffer because the completion strings point into it.
  llvm::MemoryBuffer *Buffer;
};

CXCodeCompleteResults *clang_codeComplete(CXIndex CIdx,
                                          const char *source_filename,
                                          int num_command_line_args,
                                          const char **command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files,
                                          const char *complete_filename,
                                          unsigned complete_line,
                                          unsigned complete_column,
                                          CXDiagnosticCallback diag_callback,
                                          CXClientData diag_client_data) {
  // The indexer, which is mainly used to determine where diagnostics go.
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  // Configure the diagnostics.
  DiagnosticOptions DiagOpts;
  llvm::OwningPtr<Diagnostic> Diags;
  Diags.reset(CompilerInstance::createDiagnostics(DiagOpts, 0, 0));
  CIndexDiagnosticClient DiagClient(diag_callback, diag_client_data);
  Diags->setClient(&DiagClient);
  
  // The set of temporary files that we've built.
  std::vector<llvm::sys::Path> TemporaryFiles;

  // Build up the arguments for invoking 'clang'.
  std::vector<const char *> argv;

  // First add the complete path to the 'clang' executable.
  llvm::sys::Path ClangPath = CXXIdx->getClangPath();
  argv.push_back(ClangPath.c_str());

  // Add the '-fsyntax-only' argument so that we only perform a basic
  // syntax check of the code.
  argv.push_back("-fsyntax-only");

  // Add the appropriate '-code-completion-at=file:line:column' argument
  // to perform code completion, with an "-Xclang" preceding it.
  std::string code_complete_at;
  code_complete_at += complete_filename;
  code_complete_at += ":";
  code_complete_at += llvm::utostr(complete_line);
  code_complete_at += ":";
  code_complete_at += llvm::utostr(complete_column);
  argv.push_back("-Xclang");
  argv.push_back("-code-completion-at");
  argv.push_back("-Xclang");
  argv.push_back(code_complete_at.c_str());
  argv.push_back("-Xclang");
  argv.push_back("-no-code-completion-debug-printer");
  argv.push_back("-Xclang");
  argv.push_back("-code-completion-macros");
  argv.push_back("-fdiagnostics-binary");

  // Remap any unsaved files to temporary files.
  std::vector<std::string> RemapArgs;
  if (RemapFiles(num_unsaved_files, unsaved_files, RemapArgs, TemporaryFiles))
    return 0;

  // The pointers into the elements of RemapArgs are stable because we
  // won't be adding anything to RemapArgs after this point.
  for (unsigned i = 0, e = RemapArgs.size(); i != e; ++i)
    argv.push_back(RemapArgs[i].c_str());

  // Add the source file name (FIXME: later, we'll want to build temporary
  // file from the buffer, or just feed the source text via standard input).
  if (source_filename)
    argv.push_back(source_filename);

  // Process the compiler options, stripping off '-o', '-c', '-fsyntax-only'.
  for (int i = 0; i < num_command_line_args; ++i)
    if (const char *arg = command_line_args[i]) {
      if (strcmp(arg, "-o") == 0) {
        ++i; // Also skip the matching argument.
        continue;
      }
      if (strcmp(arg, "-emit-ast") == 0 ||
          strcmp(arg, "-c") == 0 ||
          strcmp(arg, "-fsyntax-only") == 0) {
        continue;
      }

      // Keep the argument.
      argv.push_back(arg);
    }

  // Add the null terminator.
  argv.push_back(NULL);

  // Generate a temporary name for the code-completion results file.
  char tmpFile[L_tmpnam];
  char *tmpFileName = tmpnam(tmpFile);
  llvm::sys::Path ResultsFile(tmpFileName);
  TemporaryFiles.push_back(ResultsFile);

  // Generate a temporary name for the diagnostics file.
  char tmpFileResults[L_tmpnam];
  char *tmpResultsFileName = tmpnam(tmpFileResults);
  llvm::sys::Path DiagnosticsFile(tmpResultsFileName);
  TemporaryFiles.push_back(DiagnosticsFile);

  // Invoke 'clang'.
  llvm::sys::Path DevNull; // leave empty, causes redirection to /dev/null
                           // on Unix or NUL (Windows).
  std::string ErrMsg;
  const llvm::sys::Path *Redirects[] = { &DevNull, &ResultsFile, 
                                         &DiagnosticsFile, 0 };
  llvm::sys::Program::ExecuteAndWait(ClangPath, &argv[0], /* env */ NULL,
                                     /* redirects */ &Redirects[0],
                                     /* secondsToWait */ 0,
                                     /* memoryLimits */ 0, &ErrMsg);

  if (!ErrMsg.empty()) {
    std::string AllArgs;
    for (std::vector<const char*>::iterator I = argv.begin(), E = argv.end();
         I != E; ++I) {
      AllArgs += ' ';
      if (*I)
        AllArgs += *I;
    }
    
    Diags->Report(diag::err_fe_clang) << AllArgs << ErrMsg;
  }

  // Parse the resulting source file to find code-completion results.
  using llvm::MemoryBuffer;
  using llvm::StringRef;
  AllocatedCXCodeCompleteResults *Results = 0;
  if (MemoryBuffer *F = MemoryBuffer::getFile(ResultsFile.c_str())) {
    llvm::SmallVector<CXCompletionResult, 4> CompletionResults;
    StringRef Buffer = F->getBuffer();
    for (const char *Str = Buffer.data(), *StrEnd = Str + Buffer.size();
         Str < StrEnd;) {
      unsigned KindValue;
      if (ReadUnsigned(Str, StrEnd, KindValue))
        break;

      CodeCompletionString *CCStr 
        = CodeCompletionString::Deserialize(Str, StrEnd);
      if (!CCStr)
        continue;

      if (!CCStr->empty()) {
        // Vend the code-completion result to the caller.
        CXCompletionResult Result;
        Result.CursorKind = (CXCursorKind)KindValue;
        Result.CompletionString = CCStr;
        CompletionResults.push_back(Result);
      }
    };

    // Allocate the results.
    Results = new AllocatedCXCodeCompleteResults;
    Results->Results = new CXCompletionResult [CompletionResults.size()];
    Results->NumResults = CompletionResults.size();
    memcpy(Results->Results, CompletionResults.data(),
           CompletionResults.size() * sizeof(CXCompletionResult));
    Results->Buffer = F;
  }

  ReportSerializedDiagnostics(DiagnosticsFile, *Diags, 
                              num_unsaved_files, unsaved_files);
  
  for (unsigned i = 0, e = TemporaryFiles.size(); i != e; ++i)
    TemporaryFiles[i].eraseFromDisk();

  return Results;
}

void clang_disposeCodeCompleteResults(CXCodeCompleteResults *ResultsIn) {
  if (!ResultsIn)
    return;

  AllocatedCXCodeCompleteResults *Results
    = static_cast<AllocatedCXCodeCompleteResults*>(ResultsIn);

  for (unsigned I = 0, N = Results->NumResults; I != N; ++I)
    delete (CXCompletionString *)Results->Results[I].CompletionString;
  delete [] Results->Results;

  Results->Results = 0;
  Results->NumResults = 0;
  delete Results->Buffer;
  Results->Buffer = 0;
  delete Results;
}

} // end extern "C"
