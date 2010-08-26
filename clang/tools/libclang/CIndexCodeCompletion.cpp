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
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Program.h"
#include <cstdlib>
#include <cstdio>


#ifdef UDP_CODE_COMPLETION_LOGGER
#include "clang/Basic/Version.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

using namespace clang;
using namespace clang::cxstring;

namespace {
  /// \brief Stored representation of a completion string.
  ///
  /// This is the representation behind a CXCompletionString.
  class CXStoredCodeCompletionString : public CodeCompletionString {
    unsigned Priority;
    CXAvailabilityKind Availability;
    
  public:
    CXStoredCodeCompletionString(unsigned Priority,
                                 CXAvailabilityKind Availability) 
      : Priority(Priority), Availability(Availability) { }
    
    unsigned getPriority() const { return Priority; }
    CXAvailabilityKind getAvailability() const { return Availability; }
  };
}

extern "C" {

enum CXCompletionChunkKind
clang_getCompletionChunkKind(CXCompletionString completion_string,
                             unsigned chunk_number) {
  CXStoredCodeCompletionString *CCStr 
    = (CXStoredCodeCompletionString *)completion_string;
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

CXString clang_getCompletionChunkText(CXCompletionString completion_string,
                                      unsigned chunk_number) {
  CXStoredCodeCompletionString *CCStr
    = (CXStoredCodeCompletionString *)completion_string;
  if (!CCStr || chunk_number >= CCStr->size())
    return createCXString(0);

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
    return createCXString((*CCStr)[chunk_number].Text, false);

  case CodeCompletionString::CK_VerticalSpace:
    // FIXME: Temporary hack until we figure out how to handle vertical space.
    return createCXString(" ");
      
  case CodeCompletionString::CK_Optional:
    // Note: treated as an empty text block.
    return createCXString("");
  }

  // Should be unreachable, but let's be careful.
  return createCXString(0);
}


CXCompletionString
clang_getCompletionChunkCompletionString(CXCompletionString completion_string,
                                         unsigned chunk_number) {
  CXStoredCodeCompletionString *CCStr 
    = (CXStoredCodeCompletionString *)completion_string;
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
  CXStoredCodeCompletionString *CCStr
    = (CXStoredCodeCompletionString *)completion_string;
  return CCStr? CCStr->size() : 0;
}

unsigned clang_getCompletionPriority(CXCompletionString completion_string) {
  CXStoredCodeCompletionString *CCStr
    = (CXStoredCodeCompletionString *)completion_string;
  return CCStr? CCStr->getPriority() : unsigned(CCP_Unlikely);
}
  
enum CXAvailabilityKind 
clang_getCompletionAvailability(CXCompletionString completion_string) {
  CXStoredCodeCompletionString *CCStr
    = (CXStoredCodeCompletionString *)completion_string;
  return CCStr? CCStr->getAvailability() : CXAvailability_Available;
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
  AllocatedCXCodeCompleteResults();
  ~AllocatedCXCodeCompleteResults();
  
  /// \brief Diagnostics produced while performing code completion.
  llvm::SmallVector<StoredDiagnostic, 8> Diagnostics;

  /// \brief Diag object
  llvm::IntrusiveRefCntPtr<Diagnostic> Diag;
  
  /// \brief Language options used to adjust source locations.
  LangOptions LangOpts;

  /// \brief Source manager, used for diagnostics.
  SourceManager SourceMgr;
  
  /// \brief File manager, used for diagnostics.
  FileManager FileMgr;
  
  /// \brief Temporary files that should be removed once we have finished
  /// with the code-completion results.
  std::vector<llvm::sys::Path> TemporaryFiles;

  /// \brief Temporary buffers that will be deleted once we have finished with the code-completion results.
  llvm::SmallVector<const llvm::MemoryBuffer *, 1> TemporaryBuffers;
};

AllocatedCXCodeCompleteResults::AllocatedCXCodeCompleteResults() 
  : CXCodeCompleteResults(), Diag(new Diagnostic), SourceMgr(*Diag) { }
  
AllocatedCXCodeCompleteResults::~AllocatedCXCodeCompleteResults() {
  for (unsigned I = 0, N = NumResults; I != N; ++I)
    delete (CXStoredCodeCompletionString *)Results[I].CompletionString;
  delete [] Results;
  
  for (unsigned I = 0, N = TemporaryFiles.size(); I != N; ++I)
    TemporaryFiles[I].eraseFromDisk();
  for (unsigned I = 0, N = TemporaryBuffers.size(); I != N; ++I)
    delete TemporaryBuffers[I];
}
  
CXCodeCompleteResults *clang_codeComplete(CXIndex CIdx,
                                          const char *source_filename,
                                          int num_command_line_args,
                                          const char **command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files,
                                          const char *complete_filename,
                                          unsigned complete_line,
                                          unsigned complete_column) {
#ifdef UDP_CODE_COMPLETION_LOGGER
#ifdef UDP_CODE_COMPLETION_LOGGER_PORT
  const llvm::TimeRecord &StartTime =  llvm::TimeRecord::getCurrentTime();
#endif
#endif

  bool EnableLogging = getenv("LIBCLANG_CODE_COMPLETION_LOGGING") != 0;

  llvm::OwningPtr<llvm::NamedRegionTimer> CCTimer;
  if (getenv("LIBCLANG_TIMING")) {
    llvm::SmallString<128> TimerName;
    llvm::raw_svector_ostream TimerNameOut(TimerName);
    TimerNameOut << "Code completion (out-of-process) @ " << complete_filename 
      << ":" << complete_line << ":" << complete_column;
    CCTimer.reset(new llvm::NamedRegionTimer(TimerNameOut.str()));
  }

  // The indexer, which is mainly used to determine where diagnostics go.
  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  // Configure the diagnostics.
  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags;
  Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0);
  
  // The set of temporary files that we've built.
  std::vector<llvm::sys::Path> TemporaryFiles;

  // Build up the arguments for invoking 'clang'.
  std::vector<const char *> argv;

  // First add the complete path to the 'clang' executable.
  llvm::sys::Path ClangPath = CXXIdx->getClangPath();
  argv.push_back(ClangPath.c_str());

  // Always use Clang C++ support.
  argv.push_back("-ccc-clang-cxx");
  
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

  if (EnableLogging) {
    std::string Log = ClangPath.str();
    for (unsigned I = 0, N = argv.size(); I != N; ++I) {
      Log += ' ';
      Log += argv[I];
    }
    fprintf(stderr, "libclang (Code Completion): %s\n", Log.c_str());
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
    
    Diags->Report(diag::err_fe_invoking) << AllArgs << ErrMsg;
  }

  // Parse the resulting source file to find code-completion results.
  using llvm::MemoryBuffer;
  using llvm::StringRef;
  AllocatedCXCodeCompleteResults *Results = new AllocatedCXCodeCompleteResults;
  Results->Results = 0;
  Results->NumResults = 0;
  // FIXME: Set Results->LangOpts!
  if (MemoryBuffer *F = MemoryBuffer::getFile(ResultsFile.c_str())) {
    llvm::SmallVector<CXCompletionResult, 4> CompletionResults;
    StringRef Buffer = F->getBuffer();
    for (const char *Str = Buffer.data(), *StrEnd = Str + Buffer.size();
         Str < StrEnd;) {
      unsigned KindValue;
      if (ReadUnsigned(Str, StrEnd, KindValue))
        break;

      unsigned Priority;
      if (ReadUnsigned(Str, StrEnd, Priority))
        break;
      
      unsigned Availability;
      if (ReadUnsigned(Str, StrEnd, Availability))
        break;
      
      CXStoredCodeCompletionString *CCStr
        = new CXStoredCodeCompletionString(Priority, 
                                           (CXAvailabilityKind)Availability);
      if (!CCStr->Deserialize(Str, StrEnd)) {
        delete CCStr;
        continue;
      }

      if (!CCStr->empty()) {
        // Vend the code-completion result to the caller.
        CXCompletionResult Result;
        Result.CursorKind = (CXCursorKind)KindValue;
        Result.CompletionString = CCStr;
        CompletionResults.push_back(Result);
      }
    };

    // Allocate the results.
    Results->Results = new CXCompletionResult [CompletionResults.size()];
    Results->NumResults = CompletionResults.size();
    memcpy(Results->Results, CompletionResults.data(),
           CompletionResults.size() * sizeof(CXCompletionResult));
    Results->TemporaryBuffers.push_back(F);
  }

  LoadSerializedDiagnostics(DiagnosticsFile, num_unsaved_files, unsaved_files,
                            Results->FileMgr, Results->SourceMgr, 
                            Results->Diagnostics);

  // Make sure we delete temporary files when the code-completion results are
  // destroyed.
  Results->TemporaryFiles.swap(TemporaryFiles);

#ifdef UDP_CODE_COMPLETION_LOGGER
#ifdef UDP_CODE_COMPLETION_LOGGER_PORT
  const llvm::TimeRecord &EndTime =  llvm::TimeRecord::getCurrentTime();
  llvm::SmallString<256> LogResult;
  llvm::raw_svector_ostream os(LogResult);

  // Figure out the language and whether or not it uses PCH.
  const char *lang = 0;
  bool usesPCH = false;

  for (std::vector<const char*>::iterator I = argv.begin(), E = argv.end();
       I != E; ++I) {
    if (*I == 0)
      continue;
    if (strcmp(*I, "-x") == 0) {
      if (I + 1 != E) {
        lang = *(++I);
        continue;
      }
    }
    else if (strcmp(*I, "-include") == 0) {
      if (I+1 != E) {
        const char *arg = *(++I);
        llvm::SmallString<512> pchName;
        {
          llvm::raw_svector_ostream os(pchName);
          os << arg << ".pth";
        }
        pchName.push_back('\0');
        struct stat stat_results;
        if (stat(pchName.data(), &stat_results) == 0)
          usesPCH = true;
        continue;
      }
    }
  }

  os << "{ ";
  os << "\"wall\": " << (EndTime.getWallTime() - StartTime.getWallTime());
  os << ", \"numRes\": " << Results->NumResults;
  os << ", \"diags\": " << Results->Diagnostics.size();
  os << ", \"pch\": " << (usesPCH ? "true" : "false");
  os << ", \"lang\": \"" << (lang ? lang : "<unknown>") << '"';
  const char *name = getlogin();
  os << ", \"user\": \"" << (name ? name : "unknown") << '"';
  os << ", \"clangVer\": \"" << getClangFullVersion() << '"';
  os << " }";

  llvm::StringRef res = os.str();
  if (res.size() > 0) {
    do {
      // Setup the UDP socket.
      struct sockaddr_in servaddr;
      bzero(&servaddr, sizeof(servaddr));
      servaddr.sin_family = AF_INET;
      servaddr.sin_port = htons(UDP_CODE_COMPLETION_LOGGER_PORT);
      if (inet_pton(AF_INET, UDP_CODE_COMPLETION_LOGGER,
                    &servaddr.sin_addr) <= 0)
        break;

      int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
      if (sockfd < 0)
        break;

      sendto(sockfd, res.data(), res.size(), 0,
             (struct sockaddr *)&servaddr, sizeof(servaddr));
      close(sockfd);
    }
    while (false);
  }
#endif
#endif
  return Results;
}

} // end extern "C"

namespace {
  class CaptureCompletionResults : public CodeCompleteConsumer {
    AllocatedCXCodeCompleteResults &AllocatedResults;

  public:
    explicit CaptureCompletionResults(AllocatedCXCodeCompleteResults &Results)
      : CodeCompleteConsumer(true, false, true, false), 
        AllocatedResults(Results) { }

    virtual void ProcessCodeCompleteResults(Sema &S, 
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults) {
      AllocatedResults.Results = new CXCompletionResult [NumResults];
      AllocatedResults.NumResults = NumResults;
      for (unsigned I = 0; I != NumResults; ++I) {
        CXStoredCodeCompletionString *StoredCompletion
          = new CXStoredCodeCompletionString(Results[I].Priority,
                                             Results[I].Availability);
        (void)Results[I].CreateCodeCompletionString(S, StoredCompletion);
        AllocatedResults.Results[I].CursorKind = Results[I].CursorKind;
        AllocatedResults.Results[I].CompletionString = StoredCompletion;
      }
    }
    
    // FIXME: Add ProcessOverloadCandidates?
  };
}

extern "C" {
struct CodeCompleteAtInfo {
  CXTranslationUnit TU;
  const char *complete_filename;
  unsigned complete_line;
  unsigned complete_column;
  struct CXUnsavedFile *unsaved_files;
  unsigned num_unsaved_files;
  unsigned options;
  CXCodeCompleteResults *result;
};
void clang_codeCompleteAt_Impl(void *UserData) {
  CodeCompleteAtInfo *CCAI = static_cast<CodeCompleteAtInfo*>(UserData);
  CXTranslationUnit TU = CCAI->TU;
  const char *complete_filename = CCAI->complete_filename;
  unsigned complete_line = CCAI->complete_line;
  unsigned complete_column = CCAI->complete_column;
  struct CXUnsavedFile *unsaved_files = CCAI->unsaved_files;
  unsigned num_unsaved_files = CCAI->num_unsaved_files;
  unsigned options = CCAI->options;
  CCAI->result = 0;

#ifdef UDP_CODE_COMPLETION_LOGGER
#ifdef UDP_CODE_COMPLETION_LOGGER_PORT
  const llvm::TimeRecord &StartTime =  llvm::TimeRecord::getCurrentTime();
#endif
#endif

  bool EnableLogging = getenv("LIBCLANG_CODE_COMPLETION_LOGGING") != 0;
  
  ASTUnit *AST = static_cast<ASTUnit *>(TU);
  if (!AST)
    return;

  // Perform the remapping of source files.
  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    llvm::StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles.push_back(std::make_pair(unsaved_files[I].Filename,
                                           Buffer));
  }
  
  if (EnableLogging) {
    // FIXME: Add logging.
  }

  // Parse the resulting source file to find code-completion results.
  AllocatedCXCodeCompleteResults *Results = new AllocatedCXCodeCompleteResults;
  Results->Results = 0;
  Results->NumResults = 0;

  // Create a code-completion consumer to capture the results.
  CaptureCompletionResults Capture(*Results);

  // Perform completion.
  AST->CodeComplete(complete_filename, complete_line, complete_column,
                    RemappedFiles.data(), RemappedFiles.size(), 
                    (options & CXCodeComplete_IncludeMacros),
                    (options & CXCodeComplete_IncludeCodePatterns),
                    Capture,
                    *Results->Diag, Results->LangOpts, Results->SourceMgr,
                    Results->FileMgr, Results->Diagnostics,
                    Results->TemporaryBuffers);

  

#ifdef UDP_CODE_COMPLETION_LOGGER
#ifdef UDP_CODE_COMPLETION_LOGGER_PORT
  const llvm::TimeRecord &EndTime =  llvm::TimeRecord::getCurrentTime();
  llvm::SmallString<256> LogResult;
  llvm::raw_svector_ostream os(LogResult);

  // Figure out the language and whether or not it uses PCH.
  const char *lang = 0;
  bool usesPCH = false;

  for (std::vector<const char*>::iterator I = argv.begin(), E = argv.end();
       I != E; ++I) {
    if (*I == 0)
      continue;
    if (strcmp(*I, "-x") == 0) {
      if (I + 1 != E) {
        lang = *(++I);
        continue;
      }
    }
    else if (strcmp(*I, "-include") == 0) {
      if (I+1 != E) {
        const char *arg = *(++I);
        llvm::SmallString<512> pchName;
        {
          llvm::raw_svector_ostream os(pchName);
          os << arg << ".pth";
        }
        pchName.push_back('\0');
        struct stat stat_results;
        if (stat(pchName.data(), &stat_results) == 0)
          usesPCH = true;
        continue;
      }
    }
  }

  os << "{ ";
  os << "\"wall\": " << (EndTime.getWallTime() - StartTime.getWallTime());
  os << ", \"numRes\": " << Results->NumResults;
  os << ", \"diags\": " << Results->Diagnostics.size();
  os << ", \"pch\": " << (usesPCH ? "true" : "false");
  os << ", \"lang\": \"" << (lang ? lang : "<unknown>") << '"';
  const char *name = getlogin();
  os << ", \"user\": \"" << (name ? name : "unknown") << '"';
  os << ", \"clangVer\": \"" << getClangFullVersion() << '"';
  os << " }";

  llvm::StringRef res = os.str();
  if (res.size() > 0) {
    do {
      // Setup the UDP socket.
      struct sockaddr_in servaddr;
      bzero(&servaddr, sizeof(servaddr));
      servaddr.sin_family = AF_INET;
      servaddr.sin_port = htons(UDP_CODE_COMPLETION_LOGGER_PORT);
      if (inet_pton(AF_INET, UDP_CODE_COMPLETION_LOGGER,
                    &servaddr.sin_addr) <= 0)
        break;

      int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
      if (sockfd < 0)
        break;

      sendto(sockfd, res.data(), res.size(), 0,
             (struct sockaddr *)&servaddr, sizeof(servaddr));
      close(sockfd);
    }
    while (false);
  }
#endif
#endif
  CCAI->result = Results;
}
CXCodeCompleteResults *clang_codeCompleteAt(CXTranslationUnit TU,
                                            const char *complete_filename,
                                            unsigned complete_line,
                                            unsigned complete_column,
                                            struct CXUnsavedFile *unsaved_files,
                                            unsigned num_unsaved_files,
                                            unsigned options) {
  CodeCompleteAtInfo CCAI = { TU, complete_filename, complete_line,
                              complete_column, unsaved_files, num_unsaved_files,
                              options, 0 };
  llvm::CrashRecoveryContext CRC;

  if (!CRC.RunSafely(clang_codeCompleteAt_Impl, &CCAI)) {
    fprintf(stderr, "libclang: crash detected in code completion\n");
    static_cast<ASTUnit *>(TU)->setUnsafeToFree(true);
    return 0;
  }

  return CCAI.result;
}

unsigned clang_defaultCodeCompleteOptions(void) {
  return CXCodeComplete_IncludeMacros;
}

void clang_disposeCodeCompleteResults(CXCodeCompleteResults *ResultsIn) {
  if (!ResultsIn)
    return;

  AllocatedCXCodeCompleteResults *Results
    = static_cast<AllocatedCXCodeCompleteResults*>(ResultsIn);
  delete Results;
}
  
unsigned 
clang_codeCompleteGetNumDiagnostics(CXCodeCompleteResults *ResultsIn) {
  AllocatedCXCodeCompleteResults *Results
    = static_cast<AllocatedCXCodeCompleteResults*>(ResultsIn);
  if (!Results)
    return 0;

  return Results->Diagnostics.size();
}

CXDiagnostic 
clang_codeCompleteGetDiagnostic(CXCodeCompleteResults *ResultsIn,
                                unsigned Index) {
  AllocatedCXCodeCompleteResults *Results
    = static_cast<AllocatedCXCodeCompleteResults*>(ResultsIn);
  if (!Results || Index >= Results->Diagnostics.size())
    return 0;

  return new CXStoredDiagnostic(Results->Diagnostics[Index], Results->LangOpts);
}


} // end extern "C"

namespace {
  struct OrderCompletionResults {
    bool operator()(const CXCompletionResult &XR, 
                    const CXCompletionResult &YR) const {
      CXStoredCodeCompletionString *X
        = (CXStoredCodeCompletionString *)XR.CompletionString;
      CXStoredCodeCompletionString *Y
        = (CXStoredCodeCompletionString *)YR.CompletionString;
      
      const char *XText = X->getTypedText();
      const char *YText = Y->getTypedText();
      if (!XText || !YText)
        return XText != 0;
      
      int result = llvm::StringRef(XText).compare_lower(YText);
      if (result < 0)
        return true;
      if (result > 0)
        return false;
      
      result = llvm::StringRef(XText).compare(YText);
      return result < 0;
    }
  };
}

extern "C" {
  void clang_sortCodeCompletionResults(CXCompletionResult *Results,
                                       unsigned NumResults) {
    std::stable_sort(Results, Results + NumResults, OrderCompletionResults());
  }
}
