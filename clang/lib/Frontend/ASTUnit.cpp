//===--- ASTUnit.cpp - ASTUnit utility ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ASTUnit Implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTSerializationListener.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
using namespace clang;

using llvm::TimeRecord;

namespace {
  class SimpleTimer {
    bool WantTiming;
    TimeRecord Start;
    std::string Output;

  public:
    explicit SimpleTimer(bool WantTiming) : WantTiming(WantTiming) {
      if (WantTiming)
        Start = TimeRecord::getCurrentTime();
    }

    void setOutput(const llvm::Twine &Output) {
      if (WantTiming)
        this->Output = Output.str();
    }

    ~SimpleTimer() {
      if (WantTiming) {
        TimeRecord Elapsed = TimeRecord::getCurrentTime();
        Elapsed -= Start;
        llvm::errs() << Output << ':';
        Elapsed.print(Elapsed, llvm::errs());
        llvm::errs() << '\n';
      }
    }
  };
}

/// \brief After failing to build a precompiled preamble (due to
/// errors in the source that occurs in the preamble), the number of
/// reparses during which we'll skip even trying to precompile the
/// preamble.
const unsigned DefaultPreambleRebuildInterval = 5;

/// \brief Tracks the number of ASTUnit objects that are currently active.
///
/// Used for debugging purposes only.
static llvm::sys::cas_flag ActiveASTUnitObjects;

ASTUnit::ASTUnit(bool _MainFileIsAST)
  : CaptureDiagnostics(false), MainFileIsAST(_MainFileIsAST), 
    CompleteTranslationUnit(true), WantTiming(getenv("LIBCLANG_TIMING")),
    NumStoredDiagnosticsFromDriver(0),
    ConcurrencyCheckValue(CheckUnlocked), 
    PreambleRebuildCounter(0), SavedMainFileBuffer(0), PreambleBuffer(0),
    ShouldCacheCodeCompletionResults(false),
    NumTopLevelDeclsAtLastCompletionCache(0),
    CacheCodeCompletionCoolDown(0),
    UnsafeToFree(false) { 
  if (getenv("LIBCLANG_OBJTRACKING")) {
    llvm::sys::AtomicIncrement(&ActiveASTUnitObjects);
    fprintf(stderr, "+++ %d translation units\n", ActiveASTUnitObjects);
  }    
}

ASTUnit::~ASTUnit() {
  ConcurrencyCheckValue = CheckLocked;
  CleanTemporaryFiles();
  if (!PreambleFile.empty())
    llvm::sys::Path(PreambleFile).eraseFromDisk();
  
  // Free the buffers associated with remapped files. We are required to
  // perform this operation here because we explicitly request that the
  // compiler instance *not* free these buffers for each invocation of the
  // parser.
  if (Invocation.get()) {
    PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
    for (PreprocessorOptions::remapped_file_buffer_iterator
           FB = PPOpts.remapped_file_buffer_begin(),
           FBEnd = PPOpts.remapped_file_buffer_end();
         FB != FBEnd;
         ++FB)
      delete FB->second;
  }
  
  delete SavedMainFileBuffer;
  delete PreambleBuffer;

  ClearCachedCompletionResults();  
  
  if (getenv("LIBCLANG_OBJTRACKING")) {
    llvm::sys::AtomicDecrement(&ActiveASTUnitObjects);
    fprintf(stderr, "--- %d translation units\n", ActiveASTUnitObjects);
  }    
}

void ASTUnit::CleanTemporaryFiles() {
  for (unsigned I = 0, N = TemporaryFiles.size(); I != N; ++I)
    TemporaryFiles[I].eraseFromDisk();
  TemporaryFiles.clear();
}

/// \brief Determine the set of code-completion contexts in which this 
/// declaration should be shown.
static unsigned getDeclShowContexts(NamedDecl *ND,
                                    const LangOptions &LangOpts,
                                    bool &IsNestedNameSpecifier) {
  IsNestedNameSpecifier = false;
  
  if (isa<UsingShadowDecl>(ND))
    ND = dyn_cast<NamedDecl>(ND->getUnderlyingDecl());
  if (!ND)
    return 0;
  
  unsigned Contexts = 0;
  if (isa<TypeDecl>(ND) || isa<ObjCInterfaceDecl>(ND) || 
      isa<ClassTemplateDecl>(ND) || isa<TemplateTemplateParmDecl>(ND)) {
    // Types can appear in these contexts.
    if (LangOpts.CPlusPlus || !isa<TagDecl>(ND))
      Contexts |= (1 << (CodeCompletionContext::CCC_TopLevel - 1))
                | (1 << (CodeCompletionContext::CCC_ObjCIvarList - 1))
                | (1 << (CodeCompletionContext::CCC_ClassStructUnion - 1))
                | (1 << (CodeCompletionContext::CCC_Statement - 1))
                | (1 << (CodeCompletionContext::CCC_Type - 1))
              | (1 << (CodeCompletionContext::CCC_ParenthesizedExpression - 1));

    // In C++, types can appear in expressions contexts (for functional casts).
    if (LangOpts.CPlusPlus)
      Contexts |= (1 << (CodeCompletionContext::CCC_Expression - 1));
    
    // In Objective-C, message sends can send interfaces. In Objective-C++,
    // all types are available due to functional casts.
    if (LangOpts.CPlusPlus || isa<ObjCInterfaceDecl>(ND))
      Contexts |= (1 << (CodeCompletionContext::CCC_ObjCMessageReceiver - 1));

    // Deal with tag names.
    if (isa<EnumDecl>(ND)) {
      Contexts |= (1 << (CodeCompletionContext::CCC_EnumTag - 1));
      
      // Part of the nested-name-specifier in C++0x.
      if (LangOpts.CPlusPlus0x)
        IsNestedNameSpecifier = true;
    } else if (RecordDecl *Record = dyn_cast<RecordDecl>(ND)) {
      if (Record->isUnion())
        Contexts |= (1 << (CodeCompletionContext::CCC_UnionTag - 1));
      else
        Contexts |= (1 << (CodeCompletionContext::CCC_ClassOrStructTag - 1));
      
      if (LangOpts.CPlusPlus)
        IsNestedNameSpecifier = true;
    } else if (isa<ClassTemplateDecl>(ND))
      IsNestedNameSpecifier = true;
  } else if (isa<ValueDecl>(ND) || isa<FunctionTemplateDecl>(ND)) {
    // Values can appear in these contexts.
    Contexts = (1 << (CodeCompletionContext::CCC_Statement - 1))
             | (1 << (CodeCompletionContext::CCC_Expression - 1))
             | (1 << (CodeCompletionContext::CCC_ParenthesizedExpression - 1))
             | (1 << (CodeCompletionContext::CCC_ObjCMessageReceiver - 1));
  } else if (isa<ObjCProtocolDecl>(ND)) {
    Contexts = (1 << (CodeCompletionContext::CCC_ObjCProtocolName - 1));
  } else if (isa<NamespaceDecl>(ND) || isa<NamespaceAliasDecl>(ND)) {
    Contexts = (1 << (CodeCompletionContext::CCC_Namespace - 1));
   
    // Part of the nested-name-specifier.
    IsNestedNameSpecifier = true;
  }
  
  return Contexts;
}

void ASTUnit::CacheCodeCompletionResults() {
  if (!TheSema)
    return;
  
  SimpleTimer Timer(WantTiming);
  Timer.setOutput("Cache global code completions for " + getMainFileName());

  // Clear out the previous results.
  ClearCachedCompletionResults();
  
  // Gather the set of global code completions.
  typedef CodeCompletionResult Result;
  llvm::SmallVector<Result, 8> Results;
  TheSema->GatherGlobalCodeCompletions(Results);
  
  // Translate global code completions into cached completions.
  llvm::DenseMap<CanQualType, unsigned> CompletionTypes;
  
  for (unsigned I = 0, N = Results.size(); I != N; ++I) {
    switch (Results[I].Kind) {
    case Result::RK_Declaration: {
      bool IsNestedNameSpecifier = false;
      CachedCodeCompletionResult CachedResult;
      CachedResult.Completion = Results[I].CreateCodeCompletionString(*TheSema);
      CachedResult.ShowInContexts = getDeclShowContexts(Results[I].Declaration,
                                                        Ctx->getLangOptions(),
                                                        IsNestedNameSpecifier);
      CachedResult.Priority = Results[I].Priority;
      CachedResult.Kind = Results[I].CursorKind;
      CachedResult.Availability = Results[I].Availability;

      // Keep track of the type of this completion in an ASTContext-agnostic 
      // way.
      QualType UsageType = getDeclUsageType(*Ctx, Results[I].Declaration);
      if (UsageType.isNull()) {
        CachedResult.TypeClass = STC_Void;
        CachedResult.Type = 0;
      } else {
        CanQualType CanUsageType
          = Ctx->getCanonicalType(UsageType.getUnqualifiedType());
        CachedResult.TypeClass = getSimplifiedTypeClass(CanUsageType);

        // Determine whether we have already seen this type. If so, we save
        // ourselves the work of formatting the type string by using the 
        // temporary, CanQualType-based hash table to find the associated value.
        unsigned &TypeValue = CompletionTypes[CanUsageType];
        if (TypeValue == 0) {
          TypeValue = CompletionTypes.size();
          CachedCompletionTypes[QualType(CanUsageType).getAsString()]
            = TypeValue;
        }
        
        CachedResult.Type = TypeValue;
      }
      
      CachedCompletionResults.push_back(CachedResult);
      
      /// Handle nested-name-specifiers in C++.
      if (TheSema->Context.getLangOptions().CPlusPlus && 
          IsNestedNameSpecifier && !Results[I].StartsNestedNameSpecifier) {
        // The contexts in which a nested-name-specifier can appear in C++.
        unsigned NNSContexts
          = (1 << (CodeCompletionContext::CCC_TopLevel - 1))
          | (1 << (CodeCompletionContext::CCC_ObjCIvarList - 1))
          | (1 << (CodeCompletionContext::CCC_ClassStructUnion - 1))
          | (1 << (CodeCompletionContext::CCC_Statement - 1))
          | (1 << (CodeCompletionContext::CCC_Expression - 1))
          | (1 << (CodeCompletionContext::CCC_ObjCMessageReceiver - 1))
          | (1 << (CodeCompletionContext::CCC_EnumTag - 1))
          | (1 << (CodeCompletionContext::CCC_UnionTag - 1))
          | (1 << (CodeCompletionContext::CCC_ClassOrStructTag - 1))
          | (1 << (CodeCompletionContext::CCC_Type - 1))
          | (1 << (CodeCompletionContext::CCC_PotentiallyQualifiedName - 1))
          | (1 << (CodeCompletionContext::CCC_ParenthesizedExpression - 1));

        if (isa<NamespaceDecl>(Results[I].Declaration) ||
            isa<NamespaceAliasDecl>(Results[I].Declaration))
          NNSContexts |= (1 << (CodeCompletionContext::CCC_Namespace - 1));

        if (unsigned RemainingContexts 
                                = NNSContexts & ~CachedResult.ShowInContexts) {
          // If there any contexts where this completion can be a 
          // nested-name-specifier but isn't already an option, create a 
          // nested-name-specifier completion.
          Results[I].StartsNestedNameSpecifier = true;
          CachedResult.Completion = Results[I].CreateCodeCompletionString(*TheSema);
          CachedResult.ShowInContexts = RemainingContexts;
          CachedResult.Priority = CCP_NestedNameSpecifier;
          CachedResult.TypeClass = STC_Void;
          CachedResult.Type = 0;
          CachedCompletionResults.push_back(CachedResult);
        }
      }
      break;
    }
        
    case Result::RK_Keyword:
    case Result::RK_Pattern:
      // Ignore keywords and patterns; we don't care, since they are so
      // easily regenerated.
      break;
      
    case Result::RK_Macro: {
      CachedCodeCompletionResult CachedResult;
      CachedResult.Completion = Results[I].CreateCodeCompletionString(*TheSema);
      CachedResult.ShowInContexts
        = (1 << (CodeCompletionContext::CCC_TopLevel - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCInterface - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCImplementation - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCIvarList - 1))
        | (1 << (CodeCompletionContext::CCC_ClassStructUnion - 1))
        | (1 << (CodeCompletionContext::CCC_Statement - 1))
        | (1 << (CodeCompletionContext::CCC_Expression - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCMessageReceiver - 1))
        | (1 << (CodeCompletionContext::CCC_MacroNameUse - 1))
        | (1 << (CodeCompletionContext::CCC_PreprocessorExpression - 1))
        | (1 << (CodeCompletionContext::CCC_ParenthesizedExpression - 1));

      
      CachedResult.Priority = Results[I].Priority;
      CachedResult.Kind = Results[I].CursorKind;
      CachedResult.Availability = Results[I].Availability;
      CachedResult.TypeClass = STC_Void;
      CachedResult.Type = 0;
      CachedCompletionResults.push_back(CachedResult);
      break;
    }
    }
    Results[I].Destroy();
  }

  // Make a note of the state when we performed this caching.
  NumTopLevelDeclsAtLastCompletionCache = top_level_size();
}

void ASTUnit::ClearCachedCompletionResults() {
  for (unsigned I = 0, N = CachedCompletionResults.size(); I != N; ++I)
    delete CachedCompletionResults[I].Completion;
  CachedCompletionResults.clear();
  CachedCompletionTypes.clear();
}

namespace {

/// \brief Gathers information from ASTReader that will be used to initialize
/// a Preprocessor.
class ASTInfoCollector : public ASTReaderListener {
  LangOptions &LangOpt;
  HeaderSearch &HSI;
  std::string &TargetTriple;
  std::string &Predefines;
  unsigned &Counter;

  unsigned NumHeaderInfos;

public:
  ASTInfoCollector(LangOptions &LangOpt, HeaderSearch &HSI,
                   std::string &TargetTriple, std::string &Predefines,
                   unsigned &Counter)
    : LangOpt(LangOpt), HSI(HSI), TargetTriple(TargetTriple),
      Predefines(Predefines), Counter(Counter), NumHeaderInfos(0) {}

  virtual bool ReadLanguageOptions(const LangOptions &LangOpts) {
    LangOpt = LangOpts;
    return false;
  }

  virtual bool ReadTargetTriple(llvm::StringRef Triple) {
    TargetTriple = Triple;
    return false;
  }

  virtual bool ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                    llvm::StringRef OriginalFileName,
                                    std::string &SuggestedPredefines) {
    Predefines = Buffers[0].Data;
    for (unsigned I = 1, N = Buffers.size(); I != N; ++I) {
      Predefines += Buffers[I].Data;
    }
    return false;
  }

  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID) {
    HSI.setHeaderFileInfoForUID(HFI, NumHeaderInfos++);
  }

  virtual void ReadCounter(unsigned Value) {
    Counter = Value;
  }
};

class StoredDiagnosticClient : public DiagnosticClient {
  llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiags;
  
public:
  explicit StoredDiagnosticClient(
                          llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : StoredDiags(StoredDiags) { }
  
  virtual void HandleDiagnostic(Diagnostic::Level Level,
                                const DiagnosticInfo &Info);
};

/// \brief RAII object that optionally captures diagnostics, if
/// there is no diagnostic client to capture them already.
class CaptureDroppedDiagnostics {
  Diagnostic &Diags;
  StoredDiagnosticClient Client;
  DiagnosticClient *PreviousClient;

public:
  CaptureDroppedDiagnostics(bool RequestCapture, Diagnostic &Diags, 
                          llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : Diags(Diags), Client(StoredDiags), PreviousClient(0)
  {
    if (RequestCapture || Diags.getClient() == 0) {
      PreviousClient = Diags.takeClient();
      Diags.setClient(&Client);
    }
  }

  ~CaptureDroppedDiagnostics() {
    if (Diags.getClient() == &Client) {
      Diags.takeClient();
      Diags.setClient(PreviousClient);
    }
  }
};

} // anonymous namespace

void StoredDiagnosticClient::HandleDiagnostic(Diagnostic::Level Level,
                                              const DiagnosticInfo &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticClient::HandleDiagnostic(Level, Info);

  StoredDiags.push_back(StoredDiagnostic(Level, Info));
}

const std::string &ASTUnit::getOriginalSourceFileName() {
  return OriginalSourceFile;
}

const std::string &ASTUnit::getASTFileName() {
  assert(isMainFileAST() && "Not an ASTUnit from an AST file!");
  return static_cast<ASTReader *>(Ctx->getExternalSource())->getFileName();
}

llvm::MemoryBuffer *ASTUnit::getBufferForFile(llvm::StringRef Filename,
                                              std::string *ErrorStr) {
  assert(FileMgr);
  return FileMgr->getBufferForFile(Filename, ErrorStr);
}

/// \brief Configure the diagnostics object for use with ASTUnit.
void ASTUnit::ConfigureDiags(llvm::IntrusiveRefCntPtr<Diagnostic> &Diags,
                             ASTUnit &AST, bool CaptureDiagnostics) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    DiagnosticOptions DiagOpts;
    DiagnosticClient *Client = 0;
    if (CaptureDiagnostics)
      Client = new StoredDiagnosticClient(AST.StoredDiagnostics);
    Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0, Client);
  } else if (CaptureDiagnostics) {
    Diags->setClient(new StoredDiagnosticClient(AST.StoredDiagnostics));
  }
}

ASTUnit *ASTUnit::LoadFromASTFile(const std::string &Filename,
                                  llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                  const FileSystemOptions &FileSystemOpts,
                                  bool OnlyLocalDecls,
                                  RemappedFile *RemappedFiles,
                                  unsigned NumRemappedFiles,
                                  bool CaptureDiagnostics) {
  llvm::OwningPtr<ASTUnit> AST(new ASTUnit(true));
  ConfigureDiags(Diags, *AST, CaptureDiagnostics);

  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->Diagnostics = Diags;
  AST->FileMgr.reset(new FileManager(FileSystemOpts));
  AST->SourceMgr.reset(new SourceManager(AST->getDiagnostics(),
                                         AST->getFileManager()));
  AST->HeaderInfo.reset(new HeaderSearch(AST->getFileManager()));
  
  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    // Create the file entry for the file that we're mapping from.
    const FileEntry *FromFile
      = AST->getFileManager().getVirtualFile(RemappedFiles[I].first,
                                    RemappedFiles[I].second->getBufferSize(),
                                             0);
    if (!FromFile) {
      AST->getDiagnostics().Report(diag::err_fe_remap_missing_from_file)
        << RemappedFiles[I].first;
      delete RemappedFiles[I].second;
      continue;
    }
    
    // Override the contents of the "from" file with the contents of
    // the "to" file.
    AST->getSourceManager().overrideFileContents(FromFile, 
                                                 RemappedFiles[I].second);    
  }
  
  // Gather Info for preprocessor construction later on.

  LangOptions LangInfo;
  HeaderSearch &HeaderInfo = *AST->HeaderInfo.get();
  std::string TargetTriple;
  std::string Predefines;
  unsigned Counter;

  llvm::OwningPtr<ASTReader> Reader;

  Reader.reset(new ASTReader(AST->getSourceManager(), AST->getFileManager(),
                             AST->getDiagnostics()));
  Reader->setListener(new ASTInfoCollector(LangInfo, HeaderInfo, TargetTriple,
                                           Predefines, Counter));

  switch (Reader->ReadAST(Filename, ASTReader::MainFile)) {
  case ASTReader::Success:
    break;

  case ASTReader::Failure:
  case ASTReader::IgnorePCH:
    AST->getDiagnostics().Report(diag::err_fe_unable_to_load_pch);
    return NULL;
  }

  AST->OriginalSourceFile = Reader->getOriginalSourceFile();

  // AST file loaded successfully. Now create the preprocessor.

  // Get information about the target being compiled for.
  //
  // FIXME: This is broken, we should store the TargetOptions in the AST file.
  TargetOptions TargetOpts;
  TargetOpts.ABI = "";
  TargetOpts.CXXABI = "";
  TargetOpts.CPU = "";
  TargetOpts.Features.clear();
  TargetOpts.Triple = TargetTriple;
  AST->Target.reset(TargetInfo::CreateTargetInfo(AST->getDiagnostics(),
                                                 TargetOpts));
  AST->PP.reset(new Preprocessor(AST->getDiagnostics(), LangInfo, 
                                 *AST->Target.get(),
                                 AST->getSourceManager(), HeaderInfo));
  Preprocessor &PP = *AST->PP.get();

  PP.setPredefines(Reader->getSuggestedPredefines());
  PP.setCounterValue(Counter);
  Reader->setPreprocessor(PP);

  // Create and initialize the ASTContext.

  AST->Ctx.reset(new ASTContext(LangInfo,
                                AST->getSourceManager(),
                                *AST->Target.get(),
                                PP.getIdentifierTable(),
                                PP.getSelectorTable(),
                                PP.getBuiltinInfo(),
                                /* size_reserve = */0));
  ASTContext &Context = *AST->Ctx.get();

  Reader->InitializeContext(Context);

  // Attach the AST reader to the AST context as an external AST
  // source, so that declarations will be deserialized from the
  // AST file as needed.
  ASTReader *ReaderPtr = Reader.get();
  llvm::OwningPtr<ExternalASTSource> Source(Reader.take());
  Context.setExternalSource(Source);

  // Create an AST consumer, even though it isn't used.
  AST->Consumer.reset(new ASTConsumer);
  
  // Create a semantic analysis object and tell the AST reader about it.
  AST->TheSema.reset(new Sema(PP, Context, *AST->Consumer));
  AST->TheSema->Initialize();
  ReaderPtr->InitializeSema(*AST->TheSema);

  return AST.take();
}

namespace {

class TopLevelDeclTrackerConsumer : public ASTConsumer {
  ASTUnit &Unit;

public:
  TopLevelDeclTrackerConsumer(ASTUnit &_Unit) : Unit(_Unit) {}

  void HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it) {
      Decl *D = *it;
      // FIXME: Currently ObjC method declarations are incorrectly being
      // reported as top-level declarations, even though their DeclContext
      // is the containing ObjC @interface/@implementation.  This is a
      // fundamental problem in the parser right now.
      if (isa<ObjCMethodDecl>(D))
        continue;
      Unit.addTopLevelDecl(D);
    }
  }

  // We're not interested in "interesting" decls.
  void HandleInterestingDecl(DeclGroupRef) {}
};

class TopLevelDeclTrackerAction : public ASTFrontendAction {
public:
  ASTUnit &Unit;

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile) {
    return new TopLevelDeclTrackerConsumer(Unit);
  }

public:
  TopLevelDeclTrackerAction(ASTUnit &_Unit) : Unit(_Unit) {}

  virtual bool hasCodeCompletionSupport() const { return false; }
  virtual bool usesCompleteTranslationUnit()  { 
    return Unit.isCompleteTranslationUnit(); 
  }
};

class PrecompilePreambleConsumer : public PCHGenerator, 
                                   public ASTSerializationListener {
  ASTUnit &Unit;
  std::vector<Decl *> TopLevelDecls;
                                     
public:
  PrecompilePreambleConsumer(ASTUnit &Unit,
                             const Preprocessor &PP, bool Chaining,
                             const char *isysroot, llvm::raw_ostream *Out)
    : PCHGenerator(PP, Chaining, isysroot, Out), Unit(Unit) { }

  virtual void HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it) {
      Decl *D = *it;
      // FIXME: Currently ObjC method declarations are incorrectly being
      // reported as top-level declarations, even though their DeclContext
      // is the containing ObjC @interface/@implementation.  This is a
      // fundamental problem in the parser right now.
      if (isa<ObjCMethodDecl>(D))
        continue;
      TopLevelDecls.push_back(D);
    }
  }

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    PCHGenerator::HandleTranslationUnit(Ctx);
    if (!Unit.getDiagnostics().hasErrorOccurred()) {
      // Translate the top-level declarations we captured during
      // parsing into declaration IDs in the precompiled
      // preamble. This will allow us to deserialize those top-level
      // declarations when requested.
      for (unsigned I = 0, N = TopLevelDecls.size(); I != N; ++I)
        Unit.addTopLevelDeclFromPreamble(
                                      getWriter().getDeclID(TopLevelDecls[I]));
    }
  }
                                     
  virtual void SerializedPreprocessedEntity(PreprocessedEntity *Entity,
                                            uint64_t Offset) {
    Unit.addPreprocessedEntityFromPreamble(Offset);
  }
                                     
  virtual ASTSerializationListener *GetASTSerializationListener() {
    return this;
  }
};

class PrecompilePreambleAction : public ASTFrontendAction {
  ASTUnit &Unit;

public:
  explicit PrecompilePreambleAction(ASTUnit &Unit) : Unit(Unit) {}

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile) {
    std::string Sysroot;
    llvm::raw_ostream *OS = 0;
    bool Chaining;
    if (GeneratePCHAction::ComputeASTConsumerArguments(CI, InFile, Sysroot, 
                                                       OS, Chaining))
      return 0;
    
    const char *isysroot = CI.getFrontendOpts().RelocatablePCH ?
                             Sysroot.c_str() : 0;  
    return new PrecompilePreambleConsumer(Unit, CI.getPreprocessor(), Chaining,
                                          isysroot, OS);
  }

  virtual bool hasCodeCompletionSupport() const { return false; }
  virtual bool hasASTFileSupport() const { return false; }
  virtual bool usesCompleteTranslationUnit() { return false; }
};

}

/// Parse the source file into a translation unit using the given compiler
/// invocation, replacing the current translation unit.
///
/// \returns True if a failure occurred that causes the ASTUnit not to
/// contain any translation-unit information, false otherwise.
bool ASTUnit::Parse(llvm::MemoryBuffer *OverrideMainBuffer) {
  delete SavedMainFileBuffer;
  SavedMainFileBuffer = 0;
  
  if (!Invocation.get()) {
    delete OverrideMainBuffer;
    return true;
  }
  
  // Create the compiler instance to use for building the AST.
  CompilerInstance Clang;
  Clang.setInvocation(Invocation.take());
  OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;
    
  // Set up diagnostics, capturing any diagnostics that would
  // otherwise be dropped.
  Clang.setDiagnostics(&getDiagnostics());
  
  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    delete OverrideMainBuffer;
    return true;
  }

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());
  
  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_LLVM_IR &&
         "IR inputs not support here!");

  // Configure the various subsystems.
  // FIXME: Should we retain the previous file manager?
  FileSystemOpts = Clang.getFileSystemOpts();
  FileMgr.reset(new FileManager(Clang.getFileSystemOpts()));
  SourceMgr.reset(new SourceManager(getDiagnostics(), *FileMgr));
  TheSema.reset();
  Ctx.reset();
  PP.reset();
  
  // Clear out old caches and data.
  TopLevelDecls.clear();
  PreprocessedEntities.clear();
  CleanTemporaryFiles();
  PreprocessedEntitiesByFile.clear();

  if (!OverrideMainBuffer) {
    StoredDiagnostics.erase(
                    StoredDiagnostics.begin() + NumStoredDiagnosticsFromDriver,
                            StoredDiagnostics.end());
    TopLevelDeclsInPreamble.clear();
    PreprocessedEntitiesInPreamble.clear();
  }

  // Create a file manager object to provide access to and cache the filesystem.
  Clang.setFileManager(&getFileManager());
  
  // Create the source manager.
  Clang.setSourceManager(&getSourceManager());
  
  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  PreprocessorOptions &PreprocessorOpts = Clang.getPreprocessorOpts();
  std::string PriorImplicitPCHInclude;
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile, OverrideMainBuffer);
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PriorImplicitPCHInclude = PreprocessorOpts.ImplicitPCHInclude;
    PreprocessorOpts.ImplicitPCHInclude = PreambleFile;
    PreprocessorOpts.DisablePCHValidation = true;
    
    // The stored diagnostic has the old source manager in it; update
    // the locations to refer into the new source manager. Since we've
    // been careful to make sure that the source manager's state
    // before and after are identical, so that we can reuse the source
    // location itself.
    for (unsigned I = NumStoredDiagnosticsFromDriver, 
                  N = StoredDiagnostics.size(); 
         I < N; ++I) {
      FullSourceLoc Loc(StoredDiagnostics[I].getLocation(),
                        getSourceManager());
      StoredDiagnostics[I].setLocation(Loc);
    }

    // Keep track of the override buffer;
    SavedMainFileBuffer = OverrideMainBuffer;
  } else {
    PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
    PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  }
  
  llvm::OwningPtr<TopLevelDeclTrackerAction> Act;
  Act.reset(new TopLevelDeclTrackerAction(*this));
  if (!Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                            Clang.getFrontendOpts().Inputs[0].first))
    goto error;
  
  Act->Execute();
  
  // Steal the created target, context, and preprocessor, and take back the
  // source and file managers.
  TheSema.reset(Clang.takeSema());
  Consumer.reset(Clang.takeASTConsumer());
  Ctx.reset(Clang.takeASTContext());
  PP.reset(Clang.takePreprocessor());
  Clang.takeSourceManager();
  Clang.takeFileManager();
  Target.reset(Clang.takeTarget());
  
  Act->EndSourceFile();

  // Remove the overridden buffer we used for the preamble.
  if (OverrideMainBuffer) {
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    PreprocessorOpts.ImplicitPCHInclude = PriorImplicitPCHInclude;
  }

  Invocation.reset(Clang.takeInvocation());
  return false;
  
error:
  // Remove the overridden buffer we used for the preamble.
  if (OverrideMainBuffer) {
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    PreprocessorOpts.ImplicitPCHInclude = PriorImplicitPCHInclude;
    delete OverrideMainBuffer;
    SavedMainFileBuffer = 0;
  }
  
  StoredDiagnostics.clear();
  Clang.takeSourceManager();
  Clang.takeFileManager();
  Invocation.reset(Clang.takeInvocation());
  return true;
}

/// \brief Simple function to retrieve a path for a preamble precompiled header.
static std::string GetPreamblePCHPath() {
  // FIXME: This is lame; sys::Path should provide this function (in particular,
  // it should know how to find the temporary files dir).
  // FIXME: This is really lame. I copied this code from the Driver!
  // FIXME: This is a hack so that we can override the preamble file during
  // crash-recovery testing, which is the only case where the preamble files
  // are not necessarily cleaned up. 
  const char *TmpFile = ::getenv("CINDEXTEST_PREAMBLE_FILE");
  if (TmpFile)
    return TmpFile;
  
  std::string Error;
  const char *TmpDir = ::getenv("TMPDIR");
  if (!TmpDir)
    TmpDir = ::getenv("TEMP");
  if (!TmpDir)
    TmpDir = ::getenv("TMP");
#ifdef LLVM_ON_WIN32
  if (!TmpDir)
    TmpDir = ::getenv("USERPROFILE");
#endif
  if (!TmpDir)
    TmpDir = "/tmp";
  llvm::sys::Path P(TmpDir);
  P.createDirectoryOnDisk(true);
  P.appendComponent("preamble");
  P.appendSuffix("pch");
  if (P.createTemporaryFileOnDisk())
    return std::string();
  
  return P.str();
}

/// \brief Compute the preamble for the main file, providing the source buffer
/// that corresponds to the main file along with a pair (bytes, start-of-line)
/// that describes the preamble.
std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> > 
ASTUnit::ComputePreamble(CompilerInvocation &Invocation, 
                         unsigned MaxLines, bool &CreatedBuffer) {
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts = Invocation.getPreprocessorOpts();
  CreatedBuffer = false;
  
  // Try to determine if the main file has been remapped, either from the 
  // command line (to another file) or directly through the compiler invocation
  // (to a memory buffer).
  llvm::MemoryBuffer *Buffer = 0;
  llvm::sys::PathWithStatus MainFilePath(FrontendOpts.Inputs[0].second);
  if (const llvm::sys::FileStatus *MainFileStatus = MainFilePath.getFileStatus()) {
    // Check whether there is a file-file remapping of the main file
    for (PreprocessorOptions::remapped_file_iterator
          M = PreprocessorOpts.remapped_file_begin(),
          E = PreprocessorOpts.remapped_file_end();
         M != E;
         ++M) {
      llvm::sys::PathWithStatus MPath(M->first);    
      if (const llvm::sys::FileStatus *MStatus = MPath.getFileStatus()) {
        if (MainFileStatus->uniqueID == MStatus->uniqueID) {
          // We found a remapping. Try to load the resulting, remapped source.
          if (CreatedBuffer) {
            delete Buffer;
            CreatedBuffer = false;
          }
          
          Buffer = getBufferForFile(M->second);
          if (!Buffer)
            return std::make_pair((llvm::MemoryBuffer*)0, 
                                  std::make_pair(0, true));
          CreatedBuffer = true;
        }
      }
    }
    
    // Check whether there is a file-buffer remapping. It supercedes the
    // file-file remapping.
    for (PreprocessorOptions::remapped_file_buffer_iterator
           M = PreprocessorOpts.remapped_file_buffer_begin(),
           E = PreprocessorOpts.remapped_file_buffer_end();
         M != E;
         ++M) {
      llvm::sys::PathWithStatus MPath(M->first);    
      if (const llvm::sys::FileStatus *MStatus = MPath.getFileStatus()) {
        if (MainFileStatus->uniqueID == MStatus->uniqueID) {
          // We found a remapping. 
          if (CreatedBuffer) {
            delete Buffer;
            CreatedBuffer = false;
          }
          
          Buffer = const_cast<llvm::MemoryBuffer *>(M->second);
        }
      }
    }
  }
  
  // If the main source file was not remapped, load it now.
  if (!Buffer) {
    Buffer = getBufferForFile(FrontendOpts.Inputs[0].second);
    if (!Buffer)
      return std::make_pair((llvm::MemoryBuffer*)0, std::make_pair(0, true));    
    
    CreatedBuffer = true;
  }
  
  return std::make_pair(Buffer, Lexer::ComputePreamble(Buffer, MaxLines));
}

static llvm::MemoryBuffer *CreatePaddedMainFileBuffer(llvm::MemoryBuffer *Old,
                                                      unsigned NewSize,
                                                      llvm::StringRef NewName) {
  llvm::MemoryBuffer *Result
    = llvm::MemoryBuffer::getNewUninitMemBuffer(NewSize, NewName);
  memcpy(const_cast<char*>(Result->getBufferStart()), 
         Old->getBufferStart(), Old->getBufferSize());
  memset(const_cast<char*>(Result->getBufferStart()) + Old->getBufferSize(), 
         ' ', NewSize - Old->getBufferSize() - 1);
  const_cast<char*>(Result->getBufferEnd())[-1] = '\n';  
  
  return Result;
}

/// \brief Attempt to build or re-use a precompiled preamble when (re-)parsing
/// the source file.
///
/// This routine will compute the preamble of the main source file. If a
/// non-trivial preamble is found, it will precompile that preamble into a 
/// precompiled header so that the precompiled preamble can be used to reduce
/// reparsing time. If a precompiled preamble has already been constructed,
/// this routine will determine if it is still valid and, if so, avoid 
/// rebuilding the precompiled preamble.
///
/// \param AllowRebuild When true (the default), this routine is
/// allowed to rebuild the precompiled preamble if it is found to be
/// out-of-date.
///
/// \param MaxLines When non-zero, the maximum number of lines that
/// can occur within the preamble.
///
/// \returns If the precompiled preamble can be used, returns a newly-allocated
/// buffer that should be used in place of the main file when doing so.
/// Otherwise, returns a NULL pointer.
llvm::MemoryBuffer *ASTUnit::getMainBufferWithPrecompiledPreamble(
                                          CompilerInvocation PreambleInvocation,
                                                           bool AllowRebuild,
                                                           unsigned MaxLines) {
  FrontendOptions &FrontendOpts = PreambleInvocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts
    = PreambleInvocation.getPreprocessorOpts();

  bool CreatedPreambleBuffer = false;
  std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> > NewPreamble 
    = ComputePreamble(PreambleInvocation, MaxLines, CreatedPreambleBuffer);

  // If ComputePreamble() Take ownership of the
  llvm::OwningPtr<llvm::MemoryBuffer> OwnedPreambleBuffer;
  if (CreatedPreambleBuffer)
    OwnedPreambleBuffer.reset(NewPreamble.first);

  if (!NewPreamble.second.first) {
    // We couldn't find a preamble in the main source. Clear out the current
    // preamble, if we have one. It's obviously no good any more.
    Preamble.clear();
    if (!PreambleFile.empty()) {
      llvm::sys::Path(PreambleFile).eraseFromDisk();
      PreambleFile.clear();
    }

    // The next time we actually see a preamble, precompile it.
    PreambleRebuildCounter = 1;
    return 0;
  }
  
  if (!Preamble.empty()) {
    // We've previously computed a preamble. Check whether we have the same
    // preamble now that we did before, and that there's enough space in
    // the main-file buffer within the precompiled preamble to fit the
    // new main file.
    if (Preamble.size() == NewPreamble.second.first &&
        PreambleEndsAtStartOfLine == NewPreamble.second.second &&
        NewPreamble.first->getBufferSize() < PreambleReservedSize-2 &&
        memcmp(&Preamble[0], NewPreamble.first->getBufferStart(),
               NewPreamble.second.first) == 0) {
      // The preamble has not changed. We may be able to re-use the precompiled
      // preamble.

      // Check that none of the files used by the preamble have changed.
      bool AnyFileChanged = false;
          
      // First, make a record of those files that have been overridden via
      // remapping or unsaved_files.
      llvm::StringMap<std::pair<off_t, time_t> > OverriddenFiles;
      for (PreprocessorOptions::remapped_file_iterator
                R = PreprocessorOpts.remapped_file_begin(),
             REnd = PreprocessorOpts.remapped_file_end();
           !AnyFileChanged && R != REnd;
           ++R) {
        struct stat StatBuf;
        if (stat(R->second.c_str(), &StatBuf)) {
          // If we can't stat the file we're remapping to, assume that something
          // horrible happened.
          AnyFileChanged = true;
          break;
        }
        
        OverriddenFiles[R->first] = std::make_pair(StatBuf.st_size, 
                                                   StatBuf.st_mtime);
      }
      for (PreprocessorOptions::remapped_file_buffer_iterator
                R = PreprocessorOpts.remapped_file_buffer_begin(),
             REnd = PreprocessorOpts.remapped_file_buffer_end();
           !AnyFileChanged && R != REnd;
           ++R) {
        // FIXME: Should we actually compare the contents of file->buffer
        // remappings?
        OverriddenFiles[R->first] = std::make_pair(R->second->getBufferSize(), 
                                                   0);
      }
       
      // Check whether anything has changed.
      for (llvm::StringMap<std::pair<off_t, time_t> >::iterator 
             F = FilesInPreamble.begin(), FEnd = FilesInPreamble.end();
           !AnyFileChanged && F != FEnd; 
           ++F) {
        llvm::StringMap<std::pair<off_t, time_t> >::iterator Overridden
          = OverriddenFiles.find(F->first());
        if (Overridden != OverriddenFiles.end()) {
          // This file was remapped; check whether the newly-mapped file 
          // matches up with the previous mapping.
          if (Overridden->second != F->second)
            AnyFileChanged = true;
          continue;
        }
        
        // The file was not remapped; check whether it has changed on disk.
        struct stat StatBuf;
        if (stat(F->first(), &StatBuf)) {
          // If we can't stat the file, assume that something horrible happened.
          AnyFileChanged = true;
        } else if (StatBuf.st_size != F->second.first || 
                   StatBuf.st_mtime != F->second.second)
          AnyFileChanged = true;
      }
          
      if (!AnyFileChanged) {
        // Okay! We can re-use the precompiled preamble.

        // Set the state of the diagnostic object to mimic its state
        // after parsing the preamble.
        // FIXME: This won't catch any #pragma push warning changes that
        // have occurred in the preamble.
        getDiagnostics().Reset();
        ProcessWarningOptions(getDiagnostics(), 
                              PreambleInvocation.getDiagnosticOpts());
        getDiagnostics().setNumWarnings(NumWarningsInPreamble);
        if (StoredDiagnostics.size() > NumStoredDiagnosticsInPreamble)
          StoredDiagnostics.erase(
            StoredDiagnostics.begin() + NumStoredDiagnosticsInPreamble,
                                  StoredDiagnostics.end());

        // Create a version of the main file buffer that is padded to
        // buffer size we reserved when creating the preamble.
        return CreatePaddedMainFileBuffer(NewPreamble.first, 
                                          PreambleReservedSize,
                                          FrontendOpts.Inputs[0].second);
      }
    }

    // If we aren't allowed to rebuild the precompiled preamble, just
    // return now.
    if (!AllowRebuild)
      return 0;

    // We can't reuse the previously-computed preamble. Build a new one.
    Preamble.clear();
    llvm::sys::Path(PreambleFile).eraseFromDisk();
    PreambleRebuildCounter = 1;
  } else if (!AllowRebuild) {
    // We aren't allowed to rebuild the precompiled preamble; just
    // return now.
    return 0;
  }

  // If the preamble rebuild counter > 1, it's because we previously
  // failed to build a preamble and we're not yet ready to try
  // again. Decrement the counter and return a failure.
  if (PreambleRebuildCounter > 1) {
    --PreambleRebuildCounter;
    return 0;
  }

  // Create a temporary file for the precompiled preamble. In rare 
  // circumstances, this can fail.
  std::string PreamblePCHPath = GetPreamblePCHPath();
  if (PreamblePCHPath.empty()) {
    // Try again next time.
    PreambleRebuildCounter = 1;
    return 0;
  }
  
  // We did not previously compute a preamble, or it can't be reused anyway.
  SimpleTimer PreambleTimer(WantTiming);
  PreambleTimer.setOutput("Precompiling preamble");
  
  // Create a new buffer that stores the preamble. The buffer also contains
  // extra space for the original contents of the file (which will be present
  // when we actually parse the file) along with more room in case the file
  // grows.  
  PreambleReservedSize = NewPreamble.first->getBufferSize();
  if (PreambleReservedSize < 4096)
    PreambleReservedSize = 8191;
  else
    PreambleReservedSize *= 2;

  // Save the preamble text for later; we'll need to compare against it for
  // subsequent reparses.
  Preamble.assign(NewPreamble.first->getBufferStart(), 
                  NewPreamble.first->getBufferStart() 
                                                  + NewPreamble.second.first);
  PreambleEndsAtStartOfLine = NewPreamble.second.second;

  delete PreambleBuffer;
  PreambleBuffer
    = llvm::MemoryBuffer::getNewUninitMemBuffer(PreambleReservedSize,
                                                FrontendOpts.Inputs[0].second);
  memcpy(const_cast<char*>(PreambleBuffer->getBufferStart()), 
         NewPreamble.first->getBufferStart(), Preamble.size());
  memset(const_cast<char*>(PreambleBuffer->getBufferStart()) + Preamble.size(), 
         ' ', PreambleReservedSize - Preamble.size() - 1);
  const_cast<char*>(PreambleBuffer->getBufferEnd())[-1] = '\n';  
  
  // Remap the main source file to the preamble buffer.
  llvm::sys::PathWithStatus MainFilePath(FrontendOpts.Inputs[0].second);
  PreprocessorOpts.addRemappedFile(MainFilePath.str(), PreambleBuffer);
  
  // Tell the compiler invocation to generate a temporary precompiled header.
  FrontendOpts.ProgramAction = frontend::GeneratePCH;
  FrontendOpts.ChainedPCH = true;
  // FIXME: Generate the precompiled header into memory?
  FrontendOpts.OutputFile = PreamblePCHPath;
  PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
  PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  
  // Create the compiler instance to use for building the precompiled preamble.
  CompilerInstance Clang;
  Clang.setInvocation(&PreambleInvocation);
  OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;
  
  // Set up diagnostics, capturing all of the diagnostics produced.
  Clang.setDiagnostics(&getDiagnostics());
  
  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    return 0;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());
  
  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_LLVM_IR &&
         "IR inputs not support here!");
  
  // Clear out old caches and data.
  getDiagnostics().Reset();
  ProcessWarningOptions(getDiagnostics(), Clang.getDiagnosticOpts());
  StoredDiagnostics.erase(
                    StoredDiagnostics.begin() + NumStoredDiagnosticsFromDriver,
                          StoredDiagnostics.end());
  TopLevelDecls.clear();
  TopLevelDeclsInPreamble.clear();
  PreprocessedEntities.clear();
  PreprocessedEntitiesInPreamble.clear();
  
  // Create a file manager object to provide access to and cache the filesystem.
  Clang.setFileManager(new FileManager(Clang.getFileSystemOpts()));
  
  // Create the source manager.
  Clang.setSourceManager(new SourceManager(getDiagnostics(),
                                           Clang.getFileManager()));
  
  llvm::OwningPtr<PrecompilePreambleAction> Act;
  Act.reset(new PrecompilePreambleAction(*this));
  if (!Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                            Clang.getFrontendOpts().Inputs[0].first)) {
    Clang.takeInvocation();
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    return 0;
  }
  
  Act->Execute();
  Act->EndSourceFile();
  Clang.takeInvocation();
  
  if (Diagnostics->hasErrorOccurred()) {
    // There were errors parsing the preamble, so no precompiled header was
    // generated. Forget that we even tried.
    // FIXME: Should we leave a note for ourselves to try again?
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    TopLevelDeclsInPreamble.clear();
    PreprocessedEntities.clear();
    PreprocessedEntitiesInPreamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    return 0;
  }
  
  // Keep track of the preamble we precompiled.
  PreambleFile = FrontendOpts.OutputFile;
  NumStoredDiagnosticsInPreamble = StoredDiagnostics.size();
  NumWarningsInPreamble = getDiagnostics().getNumWarnings();
  
  // Keep track of all of the files that the source manager knows about,
  // so we can verify whether they have changed or not.
  FilesInPreamble.clear();
  SourceManager &SourceMgr = Clang.getSourceManager();
  const llvm::MemoryBuffer *MainFileBuffer
    = SourceMgr.getBuffer(SourceMgr.getMainFileID());
  for (SourceManager::fileinfo_iterator F = SourceMgr.fileinfo_begin(),
                                     FEnd = SourceMgr.fileinfo_end();
       F != FEnd;
       ++F) {
    const FileEntry *File = F->second->Entry;
    if (!File || F->second->getRawBuffer() == MainFileBuffer)
      continue;
    
    FilesInPreamble[File->getName()]
      = std::make_pair(F->second->getSize(), File->getModificationTime());
  }
  
  PreambleRebuildCounter = 1;
  PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
  return CreatePaddedMainFileBuffer(NewPreamble.first, 
                                    PreambleReservedSize,
                                    FrontendOpts.Inputs[0].second);
}

void ASTUnit::RealizeTopLevelDeclsFromPreamble() {
  std::vector<Decl *> Resolved;
  Resolved.reserve(TopLevelDeclsInPreamble.size());
  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (unsigned I = 0, N = TopLevelDeclsInPreamble.size(); I != N; ++I) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    Decl *D = Source.GetExternalDecl(TopLevelDeclsInPreamble[I]);
    if (D)
      Resolved.push_back(D);
  }
  TopLevelDeclsInPreamble.clear();
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());
}

void ASTUnit::RealizePreprocessedEntitiesFromPreamble() {
  if (!PP)
    return;
  
  PreprocessingRecord *PPRec = PP->getPreprocessingRecord();
  if (!PPRec)
    return;
  
  ExternalPreprocessingRecordSource *External = PPRec->getExternalSource();
  if (!External)
    return;

  for (unsigned I = 0, N = PreprocessedEntitiesInPreamble.size(); I != N; ++I) {
    if (PreprocessedEntity *PE
          = External->ReadPreprocessedEntity(PreprocessedEntitiesInPreamble[I]))
      PreprocessedEntities.push_back(PE);
  }
  
  if (PreprocessedEntities.empty())
    return;
  
  PreprocessedEntities.insert(PreprocessedEntities.end(), 
                              PPRec->begin(true), PPRec->end(true));
}

ASTUnit::pp_entity_iterator ASTUnit::pp_entity_begin() {
  if (!PreprocessedEntitiesInPreamble.empty() &&
      PreprocessedEntities.empty())
    RealizePreprocessedEntitiesFromPreamble();
  
  if (PreprocessedEntities.empty())
    if (PreprocessingRecord *PPRec = PP->getPreprocessingRecord())
      return PPRec->begin(true);
  
  return PreprocessedEntities.begin();
}

ASTUnit::pp_entity_iterator ASTUnit::pp_entity_end() {
  if (!PreprocessedEntitiesInPreamble.empty() &&
      PreprocessedEntities.empty())
    RealizePreprocessedEntitiesFromPreamble();
  
  if (PreprocessedEntities.empty())
    if (PreprocessingRecord *PPRec = PP->getPreprocessingRecord())
      return PPRec->end(true);
  
  return PreprocessedEntities.end();
}

unsigned ASTUnit::getMaxPCHLevel() const {
  if (!getOnlyLocalDecls())
    return Decl::MaxPCHLevel;

  return 0;
}

llvm::StringRef ASTUnit::getMainFileName() const {
  return Invocation->getFrontendOpts().Inputs[0].second;
}

bool ASTUnit::LoadFromCompilerInvocation(bool PrecompilePreamble) {
  if (!Invocation)
    return true;
  
  // We'll manage file buffers ourselves.
  Invocation->getPreprocessorOpts().RetainRemappedFileBuffers = true;
  Invocation->getFrontendOpts().DisableFree = false;

  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (PrecompilePreamble) {
    PreambleRebuildCounter = 2;
    OverrideMainBuffer
      = getMainBufferWithPrecompiledPreamble(*Invocation);
  }
  
  SimpleTimer ParsingTimer(WantTiming);
  ParsingTimer.setOutput("Parsing " + getMainFileName());
  
  return Parse(OverrideMainBuffer);
}

ASTUnit *ASTUnit::LoadFromCompilerInvocation(CompilerInvocation *CI,
                                   llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                             bool OnlyLocalDecls,
                                             bool CaptureDiagnostics,
                                             bool PrecompilePreamble,
                                             bool CompleteTranslationUnit,
                                             bool CacheCodeCompletionResults) {  
  // Create the AST unit.
  llvm::OwningPtr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->CompleteTranslationUnit = CompleteTranslationUnit;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->CacheCodeCompletionCoolDown = 1;
  AST->Invocation.reset(CI);
  
  return AST->LoadFromCompilerInvocation(PrecompilePreamble)? 0 : AST.take();
}

ASTUnit *ASTUnit::LoadFromCommandLine(const char **ArgBegin,
                                      const char **ArgEnd,
                                    llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                      llvm::StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls,
                                      bool CaptureDiagnostics,
                                      RemappedFile *RemappedFiles,
                                      unsigned NumRemappedFiles,
                                      bool PrecompilePreamble,
                                      bool CompleteTranslationUnit,
                                      bool CacheCodeCompletionResults,
                                      bool CXXPrecompilePreamble,
                                      bool CXXChainedPCH) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    DiagnosticOptions DiagOpts;
    Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0);
  }
  
  llvm::SmallVector<const char *, 16> Args;
  Args.push_back("<clang>"); // FIXME: Remove dummy argument.
  Args.insert(Args.end(), ArgBegin, ArgEnd);

  // FIXME: Find a cleaner way to force the driver into restricted modes. We
  // also want to force it to use clang.
  Args.push_back("-fsyntax-only");

  llvm::SmallVector<StoredDiagnostic, 4> StoredDiagnostics;
  
  llvm::OwningPtr<CompilerInvocation> CI;

  {
    CaptureDroppedDiagnostics Capture(CaptureDiagnostics, *Diags, 
                                      StoredDiagnostics);

    // FIXME: We shouldn't have to pass in the path info.
    driver::Driver TheDriver("clang", llvm::sys::getHostTriple(),
                             "a.out", false, false, *Diags);

    // Don't check that inputs exist, they have been remapped.
    TheDriver.setCheckInputsExist(false);

    llvm::OwningPtr<driver::Compilation> C(
      TheDriver.BuildCompilation(Args.size(), Args.data()));

    // We expect to get back exactly one command job, if we didn't something
    // failed.
    const driver::JobList &Jobs = C->getJobs();
    if (Jobs.size() != 1 || !isa<driver::Command>(Jobs.begin())) {
      llvm::SmallString<256> Msg;
      llvm::raw_svector_ostream OS(Msg);
      C->PrintJob(OS, C->getJobs(), "; ", true);
      Diags->Report(diag::err_fe_expected_compiler_job) << OS.str();
      return 0;
    }

    const driver::Command *Cmd = cast<driver::Command>(*Jobs.begin());
    if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
      Diags->Report(diag::err_fe_expected_clang_command);
      return 0;
    }

    const driver::ArgStringList &CCArgs = Cmd->getArguments();
    CI.reset(new CompilerInvocation);
    CompilerInvocation::CreateFromArgs(*CI,
                                     const_cast<const char **>(CCArgs.data()),
                                     const_cast<const char **>(CCArgs.data()) +
                                       CCArgs.size(),
                                       *Diags);
  }

  // Override any files that need remapping
  for (unsigned I = 0; I != NumRemappedFiles; ++I)
    CI->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                              RemappedFiles[I].second);
  
  // Override the resources path.
  CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;

  // Check whether we should precompile the preamble and/or use chained PCH.
  // FIXME: This is a temporary hack while we debug C++ chained PCH.
  if (CI->getLangOpts().CPlusPlus) {
    PrecompilePreamble = PrecompilePreamble && CXXPrecompilePreamble;
    
    if (PrecompilePreamble && !CXXChainedPCH &&
        !CI->getPreprocessorOpts().ImplicitPCHInclude.empty())
      PrecompilePreamble = false;
  }
  
  // Create the AST unit.
  llvm::OwningPtr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  ConfigureDiags(Diags, *AST, CaptureDiagnostics);
  AST->Diagnostics = Diags;
  
  AST->FileMgr.reset(new FileManager(FileSystemOptions()));
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->CompleteTranslationUnit = CompleteTranslationUnit;
  AST->ShouldCacheCodeCompletionResults = CacheCodeCompletionResults;
  AST->CacheCodeCompletionCoolDown = 1;
  AST->NumStoredDiagnosticsFromDriver = StoredDiagnostics.size();
  AST->NumStoredDiagnosticsInPreamble = StoredDiagnostics.size();
  AST->StoredDiagnostics.swap(StoredDiagnostics);
  AST->Invocation.reset(CI.take());
  return AST->LoadFromCompilerInvocation(PrecompilePreamble) ? 0 : AST.take();
}

bool ASTUnit::Reparse(RemappedFile *RemappedFiles, unsigned NumRemappedFiles) {
  if (!Invocation.get())
    return true;
  
  SimpleTimer ParsingTimer(WantTiming);
  ParsingTimer.setOutput("Reparsing " + getMainFileName());

  // Remap files.
  PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
  for (PreprocessorOptions::remapped_file_buffer_iterator 
         R = PPOpts.remapped_file_buffer_begin(),
         REnd = PPOpts.remapped_file_buffer_end();
       R != REnd; 
       ++R) {
    delete R->second;
  }
  Invocation->getPreprocessorOpts().clearRemappedFiles();
  for (unsigned I = 0; I != NumRemappedFiles; ++I)
    Invocation->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                                      RemappedFiles[I].second);
  
  // If we have a preamble file lying around, or if we might try to
  // build a precompiled preamble, do so now.
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (!PreambleFile.empty() || PreambleRebuildCounter > 0)
    OverrideMainBuffer = getMainBufferWithPrecompiledPreamble(*Invocation);
    
  // Clear out the diagnostics state.
  if (!OverrideMainBuffer) {
    getDiagnostics().Reset();
    ProcessWarningOptions(getDiagnostics(), Invocation->getDiagnosticOpts());
  }
  
  // Parse the sources
  bool Result = Parse(OverrideMainBuffer);  
  
  if (ShouldCacheCodeCompletionResults) {
    if (CacheCodeCompletionCoolDown > 0)
      --CacheCodeCompletionCoolDown;
    else if (top_level_size() != NumTopLevelDeclsAtLastCompletionCache)
      CacheCodeCompletionResults();
  }
  
  return Result;
}

//----------------------------------------------------------------------------//
// Code completion
//----------------------------------------------------------------------------//

namespace {
  /// \brief Code completion consumer that combines the cached code-completion
  /// results from an ASTUnit with the code-completion results provided to it,
  /// then passes the result on to 
  class AugmentedCodeCompleteConsumer : public CodeCompleteConsumer {
    unsigned NormalContexts;
    ASTUnit &AST;
    CodeCompleteConsumer &Next;
    
  public:
    AugmentedCodeCompleteConsumer(ASTUnit &AST, CodeCompleteConsumer &Next,
                                  bool IncludeMacros, bool IncludeCodePatterns,
                                  bool IncludeGlobals)
      : CodeCompleteConsumer(IncludeMacros, IncludeCodePatterns, IncludeGlobals,
                             Next.isOutputBinary()), AST(AST), Next(Next) 
    { 
      // Compute the set of contexts in which we will look when we don't have
      // any information about the specific context.
      NormalContexts 
        = (1 << (CodeCompletionContext::CCC_TopLevel - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCInterface - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCImplementation - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCIvarList - 1))
        | (1 << (CodeCompletionContext::CCC_Statement - 1))
        | (1 << (CodeCompletionContext::CCC_Expression - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCMessageReceiver - 1))
        | (1 << (CodeCompletionContext::CCC_MemberAccess - 1))
        | (1 << (CodeCompletionContext::CCC_ObjCProtocolName - 1))
        | (1 << (CodeCompletionContext::CCC_ParenthesizedExpression - 1))
        | (1 << (CodeCompletionContext::CCC_Recovery - 1));

      if (AST.getASTContext().getLangOptions().CPlusPlus)
        NormalContexts |= (1 << (CodeCompletionContext::CCC_EnumTag - 1))
                    | (1 << (CodeCompletionContext::CCC_UnionTag - 1))
                    | (1 << (CodeCompletionContext::CCC_ClassOrStructTag - 1));
    }
    
    virtual void ProcessCodeCompleteResults(Sema &S, 
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults);
    
    virtual void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                           OverloadCandidate *Candidates,
                                           unsigned NumCandidates) { 
      Next.ProcessOverloadCandidates(S, CurrentArg, Candidates, NumCandidates);
    }
  };
}

/// \brief Helper function that computes which global names are hidden by the
/// local code-completion results.
static void CalculateHiddenNames(const CodeCompletionContext &Context,
                                 CodeCompletionResult *Results,
                                 unsigned NumResults,
                                 ASTContext &Ctx,
                          llvm::StringSet<llvm::BumpPtrAllocator> &HiddenNames){
  bool OnlyTagNames = false;
  switch (Context.getKind()) {
  case CodeCompletionContext::CCC_Recovery:
  case CodeCompletionContext::CCC_TopLevel:
  case CodeCompletionContext::CCC_ObjCInterface:
  case CodeCompletionContext::CCC_ObjCImplementation:
  case CodeCompletionContext::CCC_ObjCIvarList:
  case CodeCompletionContext::CCC_ClassStructUnion:
  case CodeCompletionContext::CCC_Statement:
  case CodeCompletionContext::CCC_Expression:
  case CodeCompletionContext::CCC_ObjCMessageReceiver:
  case CodeCompletionContext::CCC_MemberAccess:
  case CodeCompletionContext::CCC_Namespace:
  case CodeCompletionContext::CCC_Type:
  case CodeCompletionContext::CCC_Name:
  case CodeCompletionContext::CCC_PotentiallyQualifiedName:
  case CodeCompletionContext::CCC_ParenthesizedExpression:
    break;
    
  case CodeCompletionContext::CCC_EnumTag:
  case CodeCompletionContext::CCC_UnionTag:
  case CodeCompletionContext::CCC_ClassOrStructTag:
    OnlyTagNames = true;
    break;
    
  case CodeCompletionContext::CCC_ObjCProtocolName:
  case CodeCompletionContext::CCC_MacroName:
  case CodeCompletionContext::CCC_MacroNameUse:
  case CodeCompletionContext::CCC_PreprocessorExpression:
  case CodeCompletionContext::CCC_PreprocessorDirective:
  case CodeCompletionContext::CCC_NaturalLanguage:
  case CodeCompletionContext::CCC_SelectorName:
  case CodeCompletionContext::CCC_TypeQualifiers:
  case CodeCompletionContext::CCC_Other:
    // We're looking for nothing, or we're looking for names that cannot
    // be hidden.
    return;
  }
  
  typedef CodeCompletionResult Result;
  for (unsigned I = 0; I != NumResults; ++I) {
    if (Results[I].Kind != Result::RK_Declaration)
      continue;
    
    unsigned IDNS
      = Results[I].Declaration->getUnderlyingDecl()->getIdentifierNamespace();

    bool Hiding = false;
    if (OnlyTagNames)
      Hiding = (IDNS & Decl::IDNS_Tag);
    else {
      unsigned HiddenIDNS = (Decl::IDNS_Type | Decl::IDNS_Member | 
                             Decl::IDNS_Namespace | Decl::IDNS_Ordinary |
                             Decl::IDNS_NonMemberOperator);
      if (Ctx.getLangOptions().CPlusPlus)
        HiddenIDNS |= Decl::IDNS_Tag;
      Hiding = (IDNS & HiddenIDNS);
    }
  
    if (!Hiding)
      continue;
    
    DeclarationName Name = Results[I].Declaration->getDeclName();
    if (IdentifierInfo *Identifier = Name.getAsIdentifierInfo())
      HiddenNames.insert(Identifier->getName());
    else
      HiddenNames.insert(Name.getAsString());
  }
}


void AugmentedCodeCompleteConsumer::ProcessCodeCompleteResults(Sema &S,
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults) { 
  // Merge the results we were given with the results we cached.
  bool AddedResult = false;
  unsigned InContexts  
    = (Context.getKind() == CodeCompletionContext::CCC_Recovery? NormalContexts
                                            : (1 << (Context.getKind() - 1)));

  // Contains the set of names that are hidden by "local" completion results.
  llvm::StringSet<llvm::BumpPtrAllocator> HiddenNames;
  llvm::SmallVector<CodeCompletionString *, 4> StringsToDestroy;
  typedef CodeCompletionResult Result;
  llvm::SmallVector<Result, 8> AllResults;
  for (ASTUnit::cached_completion_iterator 
            C = AST.cached_completion_begin(),
         CEnd = AST.cached_completion_end();
       C != CEnd; ++C) {
    // If the context we are in matches any of the contexts we are 
    // interested in, we'll add this result.
    if ((C->ShowInContexts & InContexts) == 0)
      continue;
    
    // If we haven't added any results previously, do so now.
    if (!AddedResult) {
      CalculateHiddenNames(Context, Results, NumResults, S.Context, 
                           HiddenNames);
      AllResults.insert(AllResults.end(), Results, Results + NumResults);
      AddedResult = true;
    }
    
    // Determine whether this global completion result is hidden by a local
    // completion result. If so, skip it.
    if (C->Kind != CXCursor_MacroDefinition &&
        HiddenNames.count(C->Completion->getTypedText()))
      continue;
    
    // Adjust priority based on similar type classes.
    unsigned Priority = C->Priority;
    CXCursorKind CursorKind = C->Kind;
    CodeCompletionString *Completion = C->Completion;
    if (!Context.getPreferredType().isNull()) {
      if (C->Kind == CXCursor_MacroDefinition) {
        Priority = getMacroUsagePriority(C->Completion->getTypedText(),
                                         S.getLangOptions(),
                               Context.getPreferredType()->isAnyPointerType());        
      } else if (C->Type) {
        CanQualType Expected
          = S.Context.getCanonicalType(
                               Context.getPreferredType().getUnqualifiedType());
        SimplifiedTypeClass ExpectedSTC = getSimplifiedTypeClass(Expected);
        if (ExpectedSTC == C->TypeClass) {
          // We know this type is similar; check for an exact match.
          llvm::StringMap<unsigned> &CachedCompletionTypes
            = AST.getCachedCompletionTypes();
          llvm::StringMap<unsigned>::iterator Pos
            = CachedCompletionTypes.find(QualType(Expected).getAsString());
          if (Pos != CachedCompletionTypes.end() && Pos->second == C->Type)
            Priority /= CCF_ExactTypeMatch;
          else
            Priority /= CCF_SimilarTypeMatch;
        }
      }
    }
    
    // Adjust the completion string, if required.
    if (C->Kind == CXCursor_MacroDefinition &&
        Context.getKind() == CodeCompletionContext::CCC_MacroNameUse) {
      // Create a new code-completion string that just contains the
      // macro name, without its arguments.
      Completion = new CodeCompletionString;
      Completion->AddTypedTextChunk(C->Completion->getTypedText());
      StringsToDestroy.push_back(Completion);
      CursorKind = CXCursor_NotImplemented;
      Priority = CCP_CodePattern;
    }
    
    AllResults.push_back(Result(Completion, Priority, CursorKind, 
                                C->Availability));
  }
  
  // If we did not add any cached completion results, just forward the
  // results we were given to the next consumer.
  if (!AddedResult) {
    Next.ProcessCodeCompleteResults(S, Context, Results, NumResults);
    return;
  }
  
  Next.ProcessCodeCompleteResults(S, Context, AllResults.data(),
                                  AllResults.size());
  
  for (unsigned I = 0, N = StringsToDestroy.size(); I != N; ++I)
    delete StringsToDestroy[I];
}



void ASTUnit::CodeComplete(llvm::StringRef File, unsigned Line, unsigned Column,
                           RemappedFile *RemappedFiles, 
                           unsigned NumRemappedFiles,
                           bool IncludeMacros, 
                           bool IncludeCodePatterns,
                           CodeCompleteConsumer &Consumer,
                           Diagnostic &Diag, LangOptions &LangOpts,
                           SourceManager &SourceMgr, FileManager &FileMgr,
                   llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiagnostics,
             llvm::SmallVectorImpl<const llvm::MemoryBuffer *> &OwnedBuffers) {
  if (!Invocation.get())
    return;

  SimpleTimer CompletionTimer(WantTiming);
  CompletionTimer.setOutput("Code completion @ " + File + ":" +
                            llvm::Twine(Line) + ":" + llvm::Twine(Column));

  CompilerInvocation CCInvocation(*Invocation);
  FrontendOptions &FrontendOpts = CCInvocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts = CCInvocation.getPreprocessorOpts();

  FrontendOpts.ShowMacrosInCodeCompletion
    = IncludeMacros && CachedCompletionResults.empty();
  FrontendOpts.ShowCodePatternsInCodeCompletion = IncludeCodePatterns;
  FrontendOpts.ShowGlobalSymbolsInCodeCompletion
    = CachedCompletionResults.empty();
  FrontendOpts.CodeCompletionAt.FileName = File;
  FrontendOpts.CodeCompletionAt.Line = Line;
  FrontendOpts.CodeCompletionAt.Column = Column;

  // Set the language options appropriately.
  LangOpts = CCInvocation.getLangOpts();

  CompilerInstance Clang;
  Clang.setInvocation(&CCInvocation);
  OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;
    
  // Set up diagnostics, capturing any diagnostics produced.
  Clang.setDiagnostics(&Diag);
  ProcessWarningOptions(Diag, CCInvocation.getDiagnosticOpts());
  CaptureDroppedDiagnostics Capture(true, 
                                    Clang.getDiagnostics(), 
                                    StoredDiagnostics);
  
  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    Clang.takeInvocation();
    return;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());
  
  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_LLVM_IR &&
         "IR inputs not support here!");

  
  // Use the source and file managers that we were given.
  Clang.setFileManager(&FileMgr);
  Clang.setSourceManager(&SourceMgr);

  // Remap files.
  PreprocessorOpts.clearRemappedFiles();
  PreprocessorOpts.RetainRemappedFileBuffers = true;
  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    PreprocessorOpts.addRemappedFile(RemappedFiles[I].first,
                                     RemappedFiles[I].second);
    OwnedBuffers.push_back(RemappedFiles[I].second);
  }
  
  // Use the code completion consumer we were given, but adding any cached
  // code-completion results.
  AugmentedCodeCompleteConsumer *AugmentedConsumer
    = new AugmentedCodeCompleteConsumer(*this, Consumer, 
                                        FrontendOpts.ShowMacrosInCodeCompletion,
                                FrontendOpts.ShowCodePatternsInCodeCompletion,
                                FrontendOpts.ShowGlobalSymbolsInCodeCompletion);
  Clang.setCodeCompletionConsumer(AugmentedConsumer);

  // If we have a precompiled preamble, try to use it. We only allow
  // the use of the precompiled preamble if we're if the completion
  // point is within the main file, after the end of the precompiled
  // preamble.
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (!PreambleFile.empty()) {
    using llvm::sys::FileStatus;
    llvm::sys::PathWithStatus CompleteFilePath(File);
    llvm::sys::PathWithStatus MainPath(OriginalSourceFile);
    if (const FileStatus *CompleteFileStatus = CompleteFilePath.getFileStatus())
      if (const FileStatus *MainStatus = MainPath.getFileStatus())
        if (CompleteFileStatus->getUniqueID() == MainStatus->getUniqueID())
          OverrideMainBuffer
            = getMainBufferWithPrecompiledPreamble(CCInvocation, false, 
                                                   Line - 1);
  }

  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  StoredDiagnostics.insert(StoredDiagnostics.end(),
                           this->StoredDiagnostics.begin(),
             this->StoredDiagnostics.begin() + NumStoredDiagnosticsFromDriver);
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile, OverrideMainBuffer);
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PreprocessorOpts.ImplicitPCHInclude = PreambleFile;
    PreprocessorOpts.DisablePCHValidation = true;
    
    // The stored diagnostics have the old source manager. Copy them
    // to our output set of stored diagnostics, updating the source
    // manager to the one we were given.
    for (unsigned I = NumStoredDiagnosticsFromDriver, 
                  N = this->StoredDiagnostics.size(); 
         I < N; ++I) {
      StoredDiagnostics.push_back(this->StoredDiagnostics[I]);
      FullSourceLoc Loc(StoredDiagnostics[I].getLocation(), SourceMgr);
      StoredDiagnostics[I].setLocation(Loc);
    }

    OwnedBuffers.push_back(OverrideMainBuffer);
  } else {
    PreprocessorOpts.PrecompiledPreambleBytes.first = 0;
    PreprocessorOpts.PrecompiledPreambleBytes.second = false;
  }

  llvm::OwningPtr<SyntaxOnlyAction> Act;
  Act.reset(new SyntaxOnlyAction);
  if (Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                           Clang.getFrontendOpts().Inputs[0].first)) {
    Act->Execute();
    Act->EndSourceFile();
  }

  // Steal back our resources. 
  Clang.takeFileManager();
  Clang.takeSourceManager();
  Clang.takeInvocation();
}

bool ASTUnit::Save(llvm::StringRef File) {
  if (getDiagnostics().hasErrorOccurred())
    return true;
  
  // FIXME: Can we somehow regenerate the stat cache here, or do we need to 
  // unconditionally create a stat cache when we parse the file?
  std::string ErrorInfo;
  llvm::raw_fd_ostream Out(File.str().c_str(), ErrorInfo,
                           llvm::raw_fd_ostream::F_Binary);
  if (!ErrorInfo.empty() || Out.has_error())
    return true;
  
  std::vector<unsigned char> Buffer;
  llvm::BitstreamWriter Stream(Buffer);
  ASTWriter Writer(Stream);
  Writer.WriteAST(getSema(), 0, 0);
  
  // Write the generated bitstream to "Out".
  if (!Buffer.empty())
    Out.write((char *)&Buffer.front(), Buffer.size());  
  Out.close();
  return Out.has_error();
}
