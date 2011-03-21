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
#include "CXTranslationUnit.h"
#include "CXString.h"
#include "CIndexDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Program.h"
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

CXString clang_getCompletionChunkText(CXCompletionString completion_string,
                                      unsigned chunk_number) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  if (!CCStr || chunk_number >= CCStr->size())
    return createCXString((const char*)0);

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
    return createCXString((*CCStr)[chunk_number].Text, false);
      
  case CodeCompletionString::CK_Optional:
    // Note: treated as an empty text block.
    return createCXString("");
  }

  // Should be unreachable, but let's be careful.
  return createCXString((const char*)0);
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

unsigned clang_getCompletionPriority(CXCompletionString completion_string) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  return CCStr? CCStr->getPriority() : unsigned(CCP_Unlikely);
}
  
enum CXAvailabilityKind 
clang_getCompletionAvailability(CXCompletionString completion_string) {
  CodeCompletionString *CCStr = (CodeCompletionString *)completion_string;
  return CCStr? static_cast<CXAvailabilityKind>(CCStr->getAvailability())
              : CXAvailability_Available;
}

/// \brief The CXCodeCompleteResults structure we allocate internally;
/// the client only sees the initial CXCodeCompleteResults structure.
struct AllocatedCXCodeCompleteResults : public CXCodeCompleteResults {
  AllocatedCXCodeCompleteResults(const FileSystemOptions& FileSystemOpts);
  ~AllocatedCXCodeCompleteResults();
  
  /// \brief Diagnostics produced while performing code completion.
  llvm::SmallVector<StoredDiagnostic, 8> Diagnostics;

  /// \brief Diag object
  llvm::IntrusiveRefCntPtr<Diagnostic> Diag;
  
  /// \brief Language options used to adjust source locations.
  LangOptions LangOpts;

  FileSystemOptions FileSystemOpts;

  /// \brief File manager, used for diagnostics.
  llvm::IntrusiveRefCntPtr<FileManager> FileMgr;

  /// \brief Source manager, used for diagnostics.
  llvm::IntrusiveRefCntPtr<SourceManager> SourceMgr;
  
  /// \brief Temporary files that should be removed once we have finished
  /// with the code-completion results.
  std::vector<llvm::sys::Path> TemporaryFiles;

  /// \brief Temporary buffers that will be deleted once we have finished with
  /// the code-completion results.
  llvm::SmallVector<const llvm::MemoryBuffer *, 1> TemporaryBuffers;
  
  /// \brief Allocator used to store globally cached code-completion results.
  llvm::IntrusiveRefCntPtr<clang::GlobalCodeCompletionAllocator> 
    CachedCompletionAllocator;
  
  /// \brief Allocator used to store code completion results.
  clang::CodeCompletionAllocator CodeCompletionAllocator;
};

/// \brief Tracks the number of code-completion result objects that are 
/// currently active.
///
/// Used for debugging purposes only.
static llvm::sys::cas_flag CodeCompletionResultObjects;
  
AllocatedCXCodeCompleteResults::AllocatedCXCodeCompleteResults(
                                      const FileSystemOptions& FileSystemOpts)
  : CXCodeCompleteResults(),
    Diag(new Diagnostic(
                   llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs))),
    FileSystemOpts(FileSystemOpts),
    FileMgr(new FileManager(FileSystemOpts)),
    SourceMgr(new SourceManager(*Diag, *FileMgr)) { 
  if (getenv("LIBCLANG_OBJTRACKING")) {
    llvm::sys::AtomicIncrement(&CodeCompletionResultObjects);
    fprintf(stderr, "+++ %d completion results\n", CodeCompletionResultObjects);
  }    
}
  
AllocatedCXCodeCompleteResults::~AllocatedCXCodeCompleteResults() {
  delete [] Results;
  
  for (unsigned I = 0, N = TemporaryFiles.size(); I != N; ++I)
    TemporaryFiles[I].eraseFromDisk();
  for (unsigned I = 0, N = TemporaryBuffers.size(); I != N; ++I)
    delete TemporaryBuffers[I];

  if (getenv("LIBCLANG_OBJTRACKING")) {
    llvm::sys::AtomicDecrement(&CodeCompletionResultObjects);
    fprintf(stderr, "--- %d completion results\n", CodeCompletionResultObjects);
  }    
}
  
} // end extern "C"

namespace {
  class CaptureCompletionResults : public CodeCompleteConsumer {
    AllocatedCXCodeCompleteResults &AllocatedResults;
    llvm::SmallVector<CXCompletionResult, 16> StoredResults;
  public:
    CaptureCompletionResults(AllocatedCXCodeCompleteResults &Results)
      : CodeCompleteConsumer(true, false, true, false), 
        AllocatedResults(Results) { }
    ~CaptureCompletionResults() { Finish(); }
    
    virtual void ProcessCodeCompleteResults(Sema &S, 
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults) {
      StoredResults.reserve(StoredResults.size() + NumResults);
      for (unsigned I = 0; I != NumResults; ++I) {
        CodeCompletionString *StoredCompletion        
          = Results[I].CreateCodeCompletionString(S, 
                                      AllocatedResults.CodeCompletionAllocator);
        
        CXCompletionResult R;
        R.CursorKind = Results[I].CursorKind;
        R.CompletionString = StoredCompletion;
        StoredResults.push_back(R);
      }
    }
    
    virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                           OverloadCandidate *Candidates,
                                           unsigned NumCandidates) {
      StoredResults.reserve(StoredResults.size() + NumCandidates);
      for (unsigned I = 0; I != NumCandidates; ++I) {
        CodeCompletionString *StoredCompletion
          = Candidates[I].CreateSignatureString(CurrentArg, S, 
                                      AllocatedResults.CodeCompletionAllocator);
        
        CXCompletionResult R;
        R.CursorKind = CXCursor_NotImplemented;
        R.CompletionString = StoredCompletion;
        StoredResults.push_back(R);
      }
    }
    
    virtual CodeCompletionAllocator &getAllocator() { 
      return AllocatedResults.CodeCompletionAllocator;
    }
    
  private:
    void Finish() {
      AllocatedResults.Results = new CXCompletionResult [StoredResults.size()];
      AllocatedResults.NumResults = StoredResults.size();
      std::memcpy(AllocatedResults.Results, StoredResults.data(), 
                  StoredResults.size() * sizeof(CXCompletionResult));
      StoredResults.clear();
    }
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
  
  ASTUnit *AST = static_cast<ASTUnit *>(TU->TUData);
  if (!AST)
    return;

  ASTUnit::ConcurrencyCheck Check(*AST);

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
  AllocatedCXCodeCompleteResults *Results = 
        new AllocatedCXCodeCompleteResults(AST->getFileSystemOpts());
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
                    *Results->Diag, Results->LangOpts, *Results->SourceMgr,
                    *Results->FileMgr, Results->Diagnostics,
                    Results->TemporaryBuffers);
  
  // Keep a reference to the allocator used for cached global completions, so
  // that we can be sure that the memory used by our code completion strings
  // doesn't get freed due to subsequent reparses (while the code completion
  // results are still active).
  Results->CachedCompletionAllocator = AST->getCachedCompletionAllocator();

  

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

  if (!RunSafely(CRC, clang_codeCompleteAt_Impl, &CCAI)) {
    fprintf(stderr, "libclang: crash detected in code completion\n");
    static_cast<ASTUnit *>(TU->TUData)->setUnsafeToFree(true);
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

/// \brief Simple utility function that appends a \p New string to the given
/// \p Old string, using the \p Buffer for storage.
///
/// \param Old The string to which we are appending. This parameter will be
/// updated to reflect the complete string.
///
///
/// \param New The string to append to \p Old.
///
/// \param Buffer A buffer that stores the actual, concatenated string. It will
/// be used if the old string is already-non-empty.
static void AppendToString(llvm::StringRef &Old, llvm::StringRef New,
                           llvm::SmallString<256> &Buffer) {
  if (Old.empty()) {
    Old = New;
    return;
  }
  
  if (Buffer.empty())
    Buffer.append(Old.begin(), Old.end());
  Buffer.append(New.begin(), New.end());
  Old = Buffer.str();
}

/// \brief Get the typed-text blocks from the given code-completion string
/// and return them as a single string.
///
/// \param String The code-completion string whose typed-text blocks will be
/// concatenated.
///
/// \param Buffer A buffer used for storage of the completed name.
static llvm::StringRef GetTypedName(CodeCompletionString *String,
                                    llvm::SmallString<256> &Buffer) {
  llvm::StringRef Result;
  for (CodeCompletionString::iterator C = String->begin(), CEnd = String->end();
       C != CEnd; ++C) {
    if (C->Kind == CodeCompletionString::CK_TypedText)
      AppendToString(Result, C->Text, Buffer);
  }
  
  return Result;
}

namespace {
  struct OrderCompletionResults {
    bool operator()(const CXCompletionResult &XR, 
                    const CXCompletionResult &YR) const {
      CodeCompletionString *X
        = (CodeCompletionString *)XR.CompletionString;
      CodeCompletionString *Y
        = (CodeCompletionString *)YR.CompletionString;
      
      llvm::SmallString<256> XBuffer;
      llvm::StringRef XText = GetTypedName(X, XBuffer);
      llvm::SmallString<256> YBuffer;      
      llvm::StringRef YText = GetTypedName(Y, YBuffer);
      
      if (XText.empty() || YText.empty())
        return !XText.empty();
            
      int result = XText.compare_lower(YText);
      if (result < 0)
        return true;
      if (result > 0)
        return false;
      
      result = XText.compare(YText);
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
