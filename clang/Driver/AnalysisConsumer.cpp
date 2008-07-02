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
#include "HTMLDiagnostics.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/OwningPtr.h"
#include "clang/AST/CFG.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/Streams.h"

using namespace clang;

  
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
    typedef llvm::ImmutableList<CodeAction> Actions;
    Actions FunctionActions;
    Actions ObjCMethodActions;

    Actions::Factory F;
    
  public:
    const bool Visualize;
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

    AnalysisConsumer(Diagnostic &diags, Preprocessor* pp,
                     PreprocessorFactory* ppf,
                     const LangOptions& lopts,
                     const std::string& fname,
                     const std::string& htmldir,
                     bool visualize, bool trim, bool analyzeAll) 
      : FunctionActions(F.GetEmptyList()), ObjCMethodActions(F.GetEmptyList()),
        Visualize(visualize), TrimGraph(trim), LOpts(lopts), Diags(diags),
        Ctx(0), PP(pp), PPF(ppf),
        HTMLDir(htmldir),
        FName(fname),
        AnalyzeAll(analyzeAll) {}
    
    void addCodeAction(CodeAction action) {
      FunctionActions = F.Concat(action, FunctionActions);
      ObjCMethodActions = F.Concat(action, ObjCMethodActions);      
    }
    
    virtual void Initialize(ASTContext &Context) {
      Ctx = &Context;
    }
    
    virtual void HandleTopLevelDecl(Decl *D);
    void HandleCode(Decl* D, Stmt* Body, Actions actions);
  };
    
  
  class VISIBILITY_HIDDEN AnalysisManager {
    Decl* D;
    Stmt* Body;    
    AnalysisConsumer& C;
    bool DisplayedFunction;
    
    llvm::OwningPtr<CFG> cfg;
    llvm::OwningPtr<LiveVariables> liveness;
    llvm::OwningPtr<ParentMap> PM;
    llvm::OwningPtr<PathDiagnosticClient> PD;

  public:
    AnalysisManager(AnalysisConsumer& c, Decl* d, Stmt* b) 
    : D(d), Body(b), C(c), DisplayedFunction(false) {}
    
    
    Decl* getCodeDecl() const { return D; }
    Stmt* getBody() const { return Body; }
    
    CFG* getCFG() {
      if (!cfg) cfg.reset(CFG::buildCFG(getBody()));
      return cfg.get();
    }
    
    ParentMap* getParentMap() {
      if (!PM) PM.reset(new ParentMap(getBody()));
      return PM.get();
    }
    
    ASTContext& getContext() {
      return *C.Ctx;
    }
    
    SourceManager& getSourceManager() {
      return getContext().getSourceManager();
    }
    
    Diagnostic& getDiagnostic() {
      return C.Diags;
    }
    
    const LangOptions& getLangOptions() const {
      return C.LOpts;
    }
    
    PathDiagnosticClient* getPathDiagnosticClient() {
      if (PD.get() == 0 && !C.HTMLDir.empty())
        PD.reset(CreateHTMLDiagnosticClient(C.HTMLDir, C.PP, C.PPF));
      
      return PD.get();      
    }
      
    LiveVariables* getLiveVariables() {
      if (!liveness) {
        liveness.reset(new LiveVariables(*getCFG()));
        liveness->runOnCFG(*getCFG());
      }
      return liveness.get();
    }
    
    bool shouldVisualize() const {
      return C.Visualize;
    }
    
    bool shouldTrimGraph() const {
      return C.TrimGraph;
    }
    
    void DisplayFunction() {
      
      if (DisplayedFunction)
        return;
      
      DisplayedFunction = true;
      
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(getCodeDecl())) {
        llvm::cerr << "ANALYZE: "
        << getContext().getSourceManager().getSourceName(FD->getLocation())
        << ' '
        << FD->getIdentifier()->getName()
        << '\n';
      }
      else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(getCodeDecl())) {
        llvm::cerr << "ANALYZE (ObjC Method): "
        << getContext().getSourceManager().getSourceName(MD->getLocation())
        << " '"
        << MD->getSelector().getName() << "'\n";
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
      
      if (FName.size() > 0 && FName != MD->getSelector().getName())
        return;
      
      Stmt* Body = MD->getBody();
      if (Body) HandleCode(MD, Body, ObjCMethodActions);
      break;
    }
      
    default:
      break;
  }
}

void AnalysisConsumer::HandleCode(Decl* D, Stmt* Body, Actions actions) {
  
  // Don't run the actions if an error has occured with parsing the file.
  if (Diags.hasErrorOccurred())
    return;
  
  SourceLocation Loc = D->getLocation();
  
  // Only run actions on declarations defined in actual source.
  if (!Loc.isFileID())
    return;
  
  // Don't run the actions on declarations in header files unless
  // otherwise specified.
  if (!AnalyzeAll && !Ctx->getSourceManager().isFromMainFile(Loc))
    return;  

  // Create an AnalysisManager that will manage the state for analyzing
  // this method/function.
  AnalysisManager mgr(*this, D, Body);
  
  // Dispatch on the actions.  
  for (Actions::iterator I = actions.begin(), 
                         E = actions.end(); I != E; ++I)
    ((*I).getHead())(mgr);  
}

//===----------------------------------------------------------------------===//
// Analyses
//===----------------------------------------------------------------------===//

static void ActionDeadStores(AnalysisManager& mgr) {
  CheckDeadStores(*mgr.getCFG(), mgr.getContext(), *mgr.getParentMap(),
                  mgr.getDiagnostic());
}

static void ActionUninitVals(AnalysisManager& mgr) {
  CheckUninitializedValues(*mgr.getCFG(), mgr.getContext(),
                           mgr.getDiagnostic());
}


static void ActionGRExprEngine(AnalysisManager& mgr, GRTransferFuncs* tf) {
  
  llvm::OwningPtr<GRTransferFuncs> TF(tf);
  
  // Display progress.
  if (!mgr.shouldVisualize())
    mgr.DisplayFunction();
  
  // Construct the analysis engine.
  GRExprEngine Eng(*mgr.getCFG(), *mgr.getCodeDecl(), mgr.getContext());  
  Eng.setTransferFunctions(tf);
  
  // Execute the worklist algorithm.
  Eng.ExecuteWorkList();
  
  // Display warnings.
  Eng.EmitWarnings(mgr.getDiagnostic(), mgr.getPathDiagnosticClient());
  
  // Visualize the exploded graph.
  if (mgr.shouldVisualize())
    Eng.ViewGraph(mgr.shouldTrimGraph());
}

static void ActionRefLeakCheckerAux(AnalysisManager& mgr, bool GCEnabled,
                                    bool StandardWarnings) {

  GRTransferFuncs* TF = MakeCFRefCountTF(mgr.getContext(),
                                         GCEnabled,
                                         StandardWarnings,
                                         mgr.getLangOptions());
    
  ActionGRExprEngine(mgr, TF);
}

static void ActionRefLeakChecker(AnalysisManager& mgr) {
     
 switch (mgr.getLangOptions().getGCMode()) {
   default:
     assert (false && "Invalid GC mode.");
   case LangOptions::NonGC:
     ActionRefLeakCheckerAux(mgr, false, true);
     break;
    
   case LangOptions::GCOnly:
     ActionRefLeakCheckerAux(mgr, true, true);
     break;
     
   case LangOptions::HybridGC:
     ActionRefLeakCheckerAux(mgr, false, true);
     ActionRefLeakCheckerAux(mgr, true, false);
     break;
 }
}

static void ActionSimpleChecks(AnalysisManager& mgr) {
  ActionGRExprEngine(mgr, MakeGRSimpleValsTF());
}

static void ActionLiveness(AnalysisManager& mgr) {
  mgr.DisplayFunction();
  mgr.getLiveVariables()->dumpBlockLiveness(mgr.getSourceManager());
}

//===----------------------------------------------------------------------===//
// AnalysisConsumer creation.
//===----------------------------------------------------------------------===//

ASTConsumer* clang::CreateAnalysisConsumer(Analyses* Beg, Analyses* End,
                                           Diagnostic &diags, Preprocessor* pp,
                                           PreprocessorFactory* ppf,
                                           const LangOptions& lopts,
                                           const std::string& fname,
                                           const std::string& htmldir,
                                           bool visualize, bool trim,
                                           bool analyzeAll) {
  
  llvm::OwningPtr<AnalysisConsumer>
  C(new AnalysisConsumer(diags, pp, ppf, lopts, fname, htmldir,
                         visualize, trim, analyzeAll));
  
  for ( ; Beg != End ; ++Beg)
    switch (*Beg) {
      case WarnDeadStores:
        C->addCodeAction(&ActionDeadStores);
        break;
        
      case WarnUninitVals:
        C->addCodeAction(&ActionUninitVals);
        break;
      
      case DisplayLiveVariables:
        C->addCodeAction(&ActionLiveness);
        break;
        
      case CheckerCFRef:
        C->addCodeAction(&ActionRefLeakChecker);
        break;
        
      case CheckerSimple:
        C->addCodeAction(&ActionSimpleChecks);
        break;
        
      default: break;
    }
  
  return C.take();
}

