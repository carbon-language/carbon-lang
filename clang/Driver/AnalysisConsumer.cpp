//===--- AnalysisConsumer.cpp - ASTConsumer for running Analyses ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// "Meta" ASTConsumer for running different source analyses.
//
//===----------------------------------------------------------------------===//

#include "ASTConsumers.h"
#include "clang/Driver/PathDiagnosticClients.h"
#include "clang/Driver/ManagerRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/OwningPtr.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/TranslationUnit.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include <vector>

using namespace clang;

static ExplodedNodeImpl::Auditor* CreateUbiViz();

// Analyzer options.
static llvm::cl::opt<bool>
PurgeDead("analyzer-purge-dead",
          llvm::cl::init(true),
          llvm::cl::desc("Remove dead symbols, bindings, and constraints before"
                         " processing a statement."));
static llvm::cl::opt<bool>
UseRanges("analyzer-range-constraints",
          llvm::cl::init(true),
          llvm::cl::desc("Use the range constraint manager instead of the basic"
                         " constraint manager"));
  
//===----------------------------------------------------------------------===//
// Basic type definitions.
//===----------------------------------------------------------------------===//

namespace {  
  class AnalysisManager;
  typedef void (*CodeAction)(AnalysisManager& Mgr);
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AnalysisConsumer declaration.
//===----------------------------------------------------------------------===//

namespace {

  class VISIBILITY_HIDDEN AnalysisConsumer : public ASTConsumer {
    typedef std::vector<CodeAction> Actions;
    Actions FunctionActions;
    Actions ObjCMethodActions;
    Actions ObjCImplementationActions;
    Actions TranslationUnitActions;
    
  public:
    const bool VisGraphviz;
    const bool VisUbigraph;
    const bool TrimGraph;
    const LangOptions& LOpts;    
    Diagnostic &Diags;
    ASTContext* Ctx;
    Preprocessor* PP;
    PreprocessorFactory* PPF;
    const std::string HTMLDir;
    const std::string FName;
    llvm::OwningPtr<PathDiagnosticClient> PD;
    bool AnalyzeAll;  
    AnalysisStores SM;
    AnalysisDiagClients DC;
    const bool DisplayProgress;

    AnalysisConsumer(Diagnostic &diags, Preprocessor* pp,
                     PreprocessorFactory* ppf,
                     const LangOptions& lopts,
                     const std::string& fname,
                     const std::string& htmldir,
                     AnalysisStores sm, AnalysisDiagClients dc,
                     bool visgraphviz, bool visubi, bool trim, bool analyzeAll,
                     bool displayProgress)
      : VisGraphviz(visgraphviz), VisUbigraph(visubi), TrimGraph(trim),
        LOpts(lopts), Diags(diags),
        Ctx(0), PP(pp), PPF(ppf),
        HTMLDir(htmldir),
        FName(fname),
        AnalyzeAll(analyzeAll), SM(sm), DC(dc),
        DisplayProgress(displayProgress) {}
    
    void addCodeAction(CodeAction action) {
      FunctionActions.push_back(action);
      ObjCMethodActions.push_back(action);
    }
    
    void addObjCImplementationAction(CodeAction action) {
      ObjCImplementationActions.push_back(action);
    }
    
    void addTranslationUnitAction(CodeAction action) {
      TranslationUnitActions.push_back(action);
    }
    
    virtual void Initialize(ASTContext &Context) {
      Ctx = &Context;
    }
    
    virtual void HandleTopLevelDecl(Decl *D);
    virtual void HandleTranslationUnit(TranslationUnit &TU);
    
    void HandleCode(Decl* D, Stmt* Body, Actions& actions);
  };
    
  
  class VISIBILITY_HIDDEN AnalysisManager : public BugReporterData {
    Decl* D; Stmt* Body; 
    TranslationUnit* TU;
    
    enum AnalysisScope { ScopeTU, ScopeDecl } AScope;
      
    AnalysisConsumer& C;
    bool DisplayedFunction;
    
    llvm::OwningPtr<CFG> cfg;
    llvm::OwningPtr<LiveVariables> liveness;
    llvm::OwningPtr<ParentMap> PM;

    // Configurable components creators.
    StoreManagerCreator CreateStoreMgr;
    ConstraintManagerCreator CreateConstraintMgr;

  public:
    AnalysisManager(AnalysisConsumer& c, Decl* d, Stmt* b, bool displayProgress) 
      : D(d), Body(b), TU(0), AScope(ScopeDecl), C(c), 
        DisplayedFunction(!displayProgress) {
      setManagerCreators();
    }
    
    AnalysisManager(AnalysisConsumer& c, TranslationUnit* tu,
                    bool displayProgress) 
      : D(0), Body(0), TU(tu), AScope(ScopeTU), C(c),
        DisplayedFunction(!displayProgress) {
      setManagerCreators();
    }
    
    Decl* getCodeDecl() const { 
      assert (AScope == ScopeDecl);
      return D;
    }
    
    Stmt* getBody() const {
      assert (AScope == ScopeDecl);
      return Body;
    }
    
    TranslationUnit* getTranslationUnit() const {
      assert (AScope == ScopeTU);
      return TU;
    }    
    
    StoreManagerCreator getStoreManagerCreator() {
      return CreateStoreMgr;
    };

    ConstraintManagerCreator getConstraintManagerCreator() {
      return CreateConstraintMgr;
    }
    
    virtual CFG* getCFG() {
      if (!cfg) cfg.reset(CFG::buildCFG(getBody()));
      return cfg.get();
    }
    
    virtual ParentMap& getParentMap() {
      if (!PM) 
        PM.reset(new ParentMap(getBody()));
      return *PM.get();
    }
    
    virtual ASTContext& getContext() {
      return *C.Ctx;
    }
    
    virtual SourceManager& getSourceManager() {
      return getContext().getSourceManager();
    }
    
    virtual Diagnostic& getDiagnostic() {
      return C.Diags;
    }
    
    const LangOptions& getLangOptions() const {
      return C.LOpts;
    }
    
    virtual PathDiagnosticClient* getPathDiagnosticClient() {
      if (C.PD.get() == 0 && !C.HTMLDir.empty()) {
        switch (C.DC) {
          default:
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATEFN, AUTOCREATE)\
case PD_##NAME: C.PD.reset(CREATEFN(C.HTMLDir, C.PP, C.PPF)); break;
#include "Analyses.def"
        }
      }
      return C.PD.get();      
    }
      
    virtual LiveVariables* getLiveVariables() {
      if (!liveness) {
        CFG* c = getCFG();
        if (!c) return 0;
        
        liveness.reset(new LiveVariables(getContext(), *c));
        liveness->runOnCFG(*c);
        liveness->runOnAllBlocks(*c, 0, true);
      }
      
      return liveness.get();
    }
    
    bool shouldVisualizeGraphviz() const {
      return C.VisGraphviz;
    }

    bool shouldVisualizeUbigraph() const {
      return C.VisUbigraph;
    }
    
    bool shouldVisualize() const {
      return C.VisGraphviz || C.VisUbigraph;
    }
    
    bool shouldTrimGraph() const {
      return C.TrimGraph;
    }
    
    void DisplayFunction() {
      
      if (DisplayedFunction)
        return;
      
      DisplayedFunction = true;
      
      // FIXME: Is getCodeDecl() always a named decl?
      if (isa<FunctionDecl>(getCodeDecl()) ||
          isa<ObjCMethodDecl>(getCodeDecl())) {
        NamedDecl *ND = cast<NamedDecl>(getCodeDecl());
        SourceManager &SM = getContext().getSourceManager();
        llvm::cerr << "ANALYZE: "
          << SM.getPresumedLoc(ND->getLocation()).getFilename()
          << ' ' << ND->getNameAsString() << '\n';
      }
    }

  private:
    /// Set configurable analyzer components creators. First check if there are
    /// components registered at runtime. Otherwise fall back to builtin
    /// components.
    void setManagerCreators() {
      if (ManagerRegistry::StoreMgrCreator != 0) {
        CreateStoreMgr = ManagerRegistry::StoreMgrCreator;
      }
      else {
        switch (C.SM) {
        default:
          assert(0 && "Unknown store manager.");
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC)     \
          case NAME##Model: CreateStoreMgr = Create##NAME##Manager; break;
#include "Analyses.def"
        }
      }

      if (ManagerRegistry::ConstraintMgrCreator != 0)
        CreateConstraintMgr = ManagerRegistry::ConstraintMgrCreator;
      else if (UseRanges)
        CreateConstraintMgr = CreateRangeConstraintManager;
      else
        CreateConstraintMgr = CreateBasicConstraintManager;
      
      // Some DiagnosticClients should be created all the time instead of
      // lazily.  Create those now.
      switch (C.DC) {
        default: break;
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATEFN, AUTOCREATE)\
case PD_##NAME: if (AUTOCREATE) getPathDiagnosticClient(); break;
#include "Analyses.def"
      }      
    }

  };

} // end anonymous namespace

namespace llvm {
  template <> struct FoldingSetTrait<CodeAction> {
    static inline void Profile(CodeAction X, FoldingSetNodeID& ID) {
      ID.AddPointer(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(X)));
    }
  };   
}

//===----------------------------------------------------------------------===//
// AnalysisConsumer implementation.
//===----------------------------------------------------------------------===//

void AnalysisConsumer::HandleTopLevelDecl(Decl *D) { 
  switch (D->getKind()) {
    case Decl::Function: {
      FunctionDecl* FD = cast<FunctionDecl>(D);

      if (FName.size() > 0 && FName != FD->getIdentifier()->getName())
        break;
      
      Stmt* Body = FD->getBody();
      if (Body) HandleCode(FD, Body, FunctionActions);
      break;
    }
      
    case Decl::ObjCMethod: {
      ObjCMethodDecl* MD = cast<ObjCMethodDecl>(D);
      
      if (FName.size() > 0 && FName != MD->getSelector().getAsString())
        return;
      
      Stmt* Body = MD->getBody();
      if (Body) HandleCode(MD, Body, ObjCMethodActions);
      break;
    }
      
    default:
      break;
  }
}

void AnalysisConsumer::HandleTranslationUnit(TranslationUnit& TU) {

  if(!TranslationUnitActions.empty()) {
    AnalysisManager mgr(*this, &TU, DisplayProgress);
    for (Actions::iterator I = TranslationUnitActions.begin(), 
         E = TranslationUnitActions.end(); I != E; ++I)
      (*I)(mgr);  
  }

  if (!ObjCImplementationActions.empty())
    for (TranslationUnit::iterator I = TU.begin(), E = TU.end(); I!=E; ++I)
      if (ObjCImplementationDecl* ID = dyn_cast<ObjCImplementationDecl>(*I))
        HandleCode(ID, 0, ObjCImplementationActions);
  
  // Delete the PathDiagnosticClient here just in case the AnalysisConsumer
  // object doesn't get released.  This will cause any side-effects in the
  // destructor of the PathDiagnosticClient to get executed.
  PD.reset();
}

void AnalysisConsumer::HandleCode(Decl* D, Stmt* Body, Actions& actions) {
  
  // Don't run the actions if an error has occured with parsing the file.
  if (Diags.hasErrorOccurred())
    return;

  // Don't run the actions on declarations in header files unless
  // otherwise specified.
  if (!AnalyzeAll && !Ctx->getSourceManager().isFromMainFile(D->getLocation()))
    return;  

  // Create an AnalysisManager that will manage the state for analyzing
  // this method/function.
  AnalysisManager mgr(*this, D, Body, DisplayProgress);
  
  // Dispatch on the actions.  
  for (Actions::iterator I = actions.begin(), E = actions.end(); I != E; ++I)
    (*I)(mgr);  
}

//===----------------------------------------------------------------------===//
// Analyses
//===----------------------------------------------------------------------===//

static void ActionWarnDeadStores(AnalysisManager& mgr) {
  if (LiveVariables* L = mgr.getLiveVariables()) {
    BugReporter BR(mgr);
    CheckDeadStores(*L, BR);
  }
}

static void ActionWarnUninitVals(AnalysisManager& mgr) {
  if (CFG* c = mgr.getCFG())
    CheckUninitializedValues(*c, mgr.getContext(), mgr.getDiagnostic());
}


static void ActionGRExprEngine(AnalysisManager& mgr, GRTransferFuncs* tf,
                               bool StandardWarnings = true) {
  
  
  llvm::OwningPtr<GRTransferFuncs> TF(tf);

  // Display progress.
  mgr.DisplayFunction();

  // Construct the analysis engine.
  LiveVariables* L = mgr.getLiveVariables();
  if (!L) return;

  GRExprEngine Eng(*mgr.getCFG(), *mgr.getCodeDecl(), mgr.getContext(), *L, mgr,
                   PurgeDead,
                   mgr.getStoreManagerCreator(), 
                   mgr.getConstraintManagerCreator());

  Eng.setTransferFunctions(tf);
  
  if (StandardWarnings) {
    Eng.RegisterInternalChecks();
    RegisterAppleChecks(Eng);
  }

  // Set the graph auditor.
  llvm::OwningPtr<ExplodedNodeImpl::Auditor> Auditor;
  if (mgr.shouldVisualizeUbigraph()) {
    Auditor.reset(CreateUbiViz());
    ExplodedNodeImpl::SetAuditor(Auditor.get());
  }
  
  // Execute the worklist algorithm.
  Eng.ExecuteWorkList();
  
  // Release the auditor (if any) so that it doesn't monitor the graph
  // created BugReporter.
  ExplodedNodeImpl::SetAuditor(0);
  
  // Display warnings.
  Eng.getBugReporter().FlushReports();
  
  // Visualize the exploded graph.
  if (mgr.shouldVisualizeGraphviz())
    Eng.ViewGraph(mgr.shouldTrimGraph());
}

static void ActionCheckerCFRefAux(AnalysisManager& mgr, bool GCEnabled,
                                  bool StandardWarnings) {
  
  GRTransferFuncs* TF = MakeCFRefCountTF(mgr.getContext(),
                                         GCEnabled,
                                         mgr.getLangOptions());
    
  ActionGRExprEngine(mgr, TF, StandardWarnings);
}

static void ActionCheckerCFRef(AnalysisManager& mgr) {
     
 switch (mgr.getLangOptions().getGCMode()) {
   default:
     assert (false && "Invalid GC mode.");
   case LangOptions::NonGC:
     ActionCheckerCFRefAux(mgr, false, true);
     break;
    
   case LangOptions::GCOnly:
     ActionCheckerCFRefAux(mgr, true, true);
     break;
     
   case LangOptions::HybridGC:
     ActionCheckerCFRefAux(mgr, false, true);
     ActionCheckerCFRefAux(mgr, true, false);
     break;
 }
}

static void ActionCheckerSimple(AnalysisManager& mgr) {
  ActionGRExprEngine(mgr, MakeGRSimpleValsTF());
}

static void ActionDisplayLiveVariables(AnalysisManager& mgr) {
  if (LiveVariables* L = mgr.getLiveVariables()) {
    mgr.DisplayFunction();  
    L->dumpBlockLiveness(mgr.getSourceManager());
  }
}

static void ActionCFGDump(AnalysisManager& mgr) {
  if (CFG* c = mgr.getCFG()) {
    mgr.DisplayFunction();
    c->dump();
  }
}

static void ActionCFGView(AnalysisManager& mgr) {
  if (CFG* c = mgr.getCFG()) {
    mgr.DisplayFunction();
    c->viewCFG();  
  }
}

static void ActionWarnObjCDealloc(AnalysisManager& mgr) {
  if (mgr.getLangOptions().getGCMode() == LangOptions::GCOnly)
    return;
      
  BugReporter BR(mgr);
  
  CheckObjCDealloc(cast<ObjCImplementationDecl>(mgr.getCodeDecl()), 
                   mgr.getLangOptions(), BR);  
}

static void ActionWarnObjCUnusedIvars(AnalysisManager& mgr) {
  BugReporter BR(mgr);
  CheckObjCUnusedIvar(cast<ObjCImplementationDecl>(mgr.getCodeDecl()), BR);  
}

static void ActionWarnObjCMethSigs(AnalysisManager& mgr) {
  BugReporter BR(mgr);
  
  CheckObjCInstMethSignature(cast<ObjCImplementationDecl>(mgr.getCodeDecl()),
                             BR);
}

//===----------------------------------------------------------------------===//
// AnalysisConsumer creation.
//===----------------------------------------------------------------------===//

ASTConsumer* clang::CreateAnalysisConsumer(Analyses* Beg, Analyses* End,
                                           AnalysisStores SM,
                                           AnalysisDiagClients DC,
                                           Diagnostic &diags, Preprocessor* pp,
                                           PreprocessorFactory* ppf,
                                           const LangOptions& lopts,
                                           const std::string& fname,
                                           const std::string& htmldir,
                                           bool VisGraphviz, bool VisUbi,
                                           bool trim,
                                           bool analyzeAll,
                                           bool displayProgress) {
  
  llvm::OwningPtr<AnalysisConsumer>
  C(new AnalysisConsumer(diags, pp, ppf, lopts, fname, htmldir, SM, DC,
                         VisGraphviz, VisUbi, trim, analyzeAll,
                         displayProgress));
  
  for ( ; Beg != End ; ++Beg)
    switch (*Beg) {
#define ANALYSIS(NAME, CMD, DESC, SCOPE)\
      case NAME:\
        C->add ## SCOPE ## Action(&Action ## NAME);\
        break;
#include "Analyses.def"
      default: break;
    }
  
  return C.take();
}

//===----------------------------------------------------------------------===//
// Ubigraph Visualization.  FIXME: Move to separate file.
//===----------------------------------------------------------------------===//

namespace {
  
class UbigraphViz : public ExplodedNodeImpl::Auditor {
  llvm::OwningPtr<llvm::raw_ostream> Out;
  llvm::sys::Path Dir, Filename;
  unsigned Cntr;

  typedef llvm::DenseMap<void*,unsigned> VMap;
  VMap M;
  
public:
  UbigraphViz(llvm::raw_ostream* out, llvm::sys::Path& dir,
              llvm::sys::Path& filename);
  
  ~UbigraphViz();
  
  virtual void AddEdge(ExplodedNodeImpl* Src, ExplodedNodeImpl* Dst);  
};
  
} // end anonymous namespace

static ExplodedNodeImpl::Auditor* CreateUbiViz() {
  std::string ErrMsg;
  
  llvm::sys::Path Dir = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);
  if (!ErrMsg.empty())
    return 0;

  llvm::sys::Path Filename = Dir;
  Filename.appendComponent("llvm_ubi");
  Filename.makeUnique(true,&ErrMsg);

  if (!ErrMsg.empty())
    return 0;

  llvm::cerr << "Writing '" << Filename << "'.\n";
  
  llvm::OwningPtr<llvm::raw_fd_ostream> Stream;
  std::string filename = Filename.toString();
  Stream.reset(new llvm::raw_fd_ostream(filename.c_str(), false, ErrMsg));

  if (!ErrMsg.empty())
    return 0;
  
  return new UbigraphViz(Stream.take(), Dir, Filename);
}

void UbigraphViz::AddEdge(ExplodedNodeImpl* Src, ExplodedNodeImpl* Dst) {
  
  assert (Src != Dst && "Self-edges are not allowed.");
  
  // Lookup the Src.  If it is a new node, it's a root.
  VMap::iterator SrcI= M.find(Src);
  unsigned SrcID;
  
  if (SrcI == M.end()) {
    M[Src] = SrcID = Cntr++;
    *Out << "('vertex', " << SrcID << ", ('color','#00ff00'))\n";
  }
  else
    SrcID = SrcI->second;
  
  // Lookup the Dst.
  VMap::iterator DstI= M.find(Dst);
  unsigned DstID;

  if (DstI == M.end()) {
    M[Dst] = DstID = Cntr++;
    *Out << "('vertex', " << DstID << ")\n";
  }
  else {
    // We have hit DstID before.  Change its style to reflect a cache hit.
    DstID = DstI->second;
    *Out << "('change_vertex_style', " << DstID << ", 1)\n";
  }

  // Add the edge.
  *Out << "('edge', " << SrcID << ", " << DstID 
       << ", ('arrow','true'), ('oriented', 'true'))\n";
}

UbigraphViz::UbigraphViz(llvm::raw_ostream* out, llvm::sys::Path& dir,
                         llvm::sys::Path& filename)
  : Out(out), Dir(dir), Filename(filename), Cntr(0) {

  *Out << "('vertex_style_attribute', 0, ('shape', 'icosahedron'))\n";
  *Out << "('vertex_style', 1, 0, ('shape', 'sphere'), ('color', '#ffcc66'),"
          " ('size', '1.5'))\n";
}

UbigraphViz::~UbigraphViz() {
  Out.reset(0);
  llvm::cerr << "Running 'ubiviz' program... ";
  std::string ErrMsg;
  llvm::sys::Path Ubiviz = llvm::sys::Program::FindProgramByName("ubiviz");
  std::vector<const char*> args;
  args.push_back(Ubiviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);
  
  if (llvm::sys::Program::ExecuteAndWait(Ubiviz, &args[0],0,0,0,0,&ErrMsg)) {
    llvm::cerr << "Error viewing graph: " << ErrMsg << "\n";
  }
  
  // Delete the directory.
  Dir.eraseFromDisk(true); 
}
