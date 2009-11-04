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

#include "clang/Frontend/AnalysisConsumer.h"
#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Frontend/ManagerRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/PathSensitive/AnalysisManager.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include "llvm/ADT/OwningPtr.h"

using namespace clang;

static ExplodedNode::Auditor* CreateUbiViz();

//===----------------------------------------------------------------------===//
// Basic type definitions.
//===----------------------------------------------------------------------===//

namespace {
  typedef void (*CodeAction)(AnalysisManager& Mgr, Decl *D);
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Special PathDiagnosticClients.
//===----------------------------------------------------------------------===//

static PathDiagnosticClient*
CreatePlistHTMLDiagnosticClient(const std::string& prefix, Preprocessor* PP) {
  llvm::sys::Path F(prefix);
  PathDiagnosticClientFactory *PF =
    CreateHTMLDiagnosticClientFactory(F.getDirname(), PP);
  return CreatePlistDiagnosticClient(prefix, PP, PF);
}

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
    const LangOptions& LOpts;
    Diagnostic &Diags;
    ASTContext* Ctx;
    Preprocessor* PP;
    const std::string OutDir;
    AnalyzerOptions Opts;


    // PD is owned by AnalysisManager.
    PathDiagnosticClient *PD;

    StoreManagerCreator CreateStoreMgr;
    ConstraintManagerCreator CreateConstraintMgr;

    llvm::OwningPtr<AnalysisManager> Mgr;

    AnalysisConsumer(Diagnostic &diags, Preprocessor* pp,
                     const LangOptions& lopts,
                     const std::string& outdir,
                     const AnalyzerOptions& opts)
      : LOpts(lopts), Diags(diags), Ctx(0), PP(pp), OutDir(outdir),
        Opts(opts), PD(0) {
      DigestAnalyzerOptions();
    }

    void DigestAnalyzerOptions() {
      // Create the PathDiagnosticClient.
      if (!OutDir.empty()) {
        switch (Opts.AnalysisDiagOpt) {
        default:
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATEFN, AUTOCREATE) \
          case PD_##NAME: PD = CREATEFN(OutDir, PP); break;
#include "clang/Frontend/Analyses.def"
        }
      }

      // Create the analyzer component creators.
      if (ManagerRegistry::StoreMgrCreator != 0) {
        CreateStoreMgr = ManagerRegistry::StoreMgrCreator;
      }
      else {
        switch (Opts.AnalysisStoreOpt) {
        default:
          assert(0 && "Unknown store manager.");
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATEFN)           \
          case NAME##Model: CreateStoreMgr = CREATEFN; break;
#include "clang/Frontend/Analyses.def"
        }
      }

      if (ManagerRegistry::ConstraintMgrCreator != 0)
        CreateConstraintMgr = ManagerRegistry::ConstraintMgrCreator;
      else {
        switch (Opts.AnalysisConstraintsOpt) {
        default:
          assert(0 && "Unknown store manager.");
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATEFN)     \
          case NAME##Model: CreateConstraintMgr = CREATEFN; break;
#include "clang/Frontend/Analyses.def"
        }
      }
    }

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
      Mgr.reset(new AnalysisManager(*Ctx, Diags, LOpts, PD,
                                    CreateStoreMgr, CreateConstraintMgr,
                                    Opts.AnalyzerDisplayProgress,
                                    Opts.VisualizeEGDot, Opts.VisualizeEGUbi,
                                    Opts.PurgeDead, Opts.EagerlyAssume,
                                    Opts.TrimGraph));
    }

    virtual void HandleTopLevelDecl(DeclGroupRef D) {
      for (DeclGroupRef::iterator I = D.begin(), E = D.end(); I != E; ++I)
        HandleTopLevelSingleDecl(*I);
    }

    void HandleTopLevelSingleDecl(Decl *D);
    virtual void HandleTranslationUnit(ASTContext &C);

    void HandleCode(Decl* D, Stmt* Body, Actions& actions);
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

void AnalysisConsumer::HandleTopLevelSingleDecl(Decl *D) {
  switch (D->getKind()) {
    case Decl::Function: {
      FunctionDecl* FD = cast<FunctionDecl>(D);

      if (!Opts.AnalyzeSpecificFunction.empty() &&
          Opts.AnalyzeSpecificFunction != FD->getIdentifier()->getName())
        break;

      Stmt* Body = FD->getBody();
      if (Body) HandleCode(FD, Body, FunctionActions);
      break;
    }

    case Decl::ObjCMethod: {
      ObjCMethodDecl* MD = cast<ObjCMethodDecl>(D);

      if (Opts.AnalyzeSpecificFunction.size() > 0 &&
          Opts.AnalyzeSpecificFunction != MD->getSelector().getAsString())
        return;

      Stmt* Body = MD->getBody();
      if (Body) HandleCode(MD, Body, ObjCMethodActions);
      break;
    }

    default:
      break;
  }
}

void AnalysisConsumer::HandleTranslationUnit(ASTContext &C) {
  
  TranslationUnitDecl *TU = C.getTranslationUnitDecl();
  
  if (!TranslationUnitActions.empty()) {  
    // Find the entry function definition (if any).
    FunctionDecl *FD = 0;
    
    if (!Opts.AnalyzeSpecificFunction.empty()) {
      for (DeclContext::decl_iterator I=TU->decls_begin(), E=TU->decls_end();
           I != E; ++I) {
        if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*I))
          if (fd->isThisDeclarationADefinition() &&
              fd->getNameAsString() == Opts.AnalyzeSpecificFunction) {
            FD = fd;
            break;
          }
      }
    }

    for (Actions::iterator I = TranslationUnitActions.begin(), 
         E = TranslationUnitActions.end(); I != E; ++I)
      (*I)(*Mgr, FD);  
  }

  if (!ObjCImplementationActions.empty()) {
    for (DeclContext::decl_iterator I = TU->decls_begin(),
                                    E = TU->decls_end();
         I != E; ++I)
      if (ObjCImplementationDecl* ID = dyn_cast<ObjCImplementationDecl>(*I))
        HandleCode(ID, 0, ObjCImplementationActions);
  }

  // Explicitly destroy the PathDiagnosticClient.  This will flush its output.
  // FIXME: This should be replaced with something that doesn't rely on
  // side-effects in PathDiagnosticClient's destructor.
  Mgr.reset(NULL);
}

void AnalysisConsumer::HandleCode(Decl *D, Stmt* Body, Actions& actions) {

  // Don't run the actions if an error has occured with parsing the file.
  if (Diags.hasErrorOccurred())
    return;

  // Don't run the actions on declarations in header files unless
  // otherwise specified.
  if (!Opts.AnalyzeAll &&
      !Ctx->getSourceManager().isFromMainFile(D->getLocation()))
    return;

  // Clear the AnalysisManager of old AnalysisContexts.
  Mgr->ClearContexts();
  
  // Dispatch on the actions.
  for (Actions::iterator I = actions.begin(), E = actions.end(); I != E; ++I)
    (*I)(*Mgr, D);  
}

//===----------------------------------------------------------------------===//
// Analyses
//===----------------------------------------------------------------------===//

static void ActionWarnDeadStores(AnalysisManager& mgr, Decl *D) {
  if (LiveVariables *L = mgr.getLiveVariables(D)) {
    BugReporter BR(mgr);
    CheckDeadStores(*mgr.getCFG(D), *L, mgr.getParentMap(D), BR);
  }
}

static void ActionWarnUninitVals(AnalysisManager& mgr, Decl *D) {
  if (CFG* c = mgr.getCFG(D))
    CheckUninitializedValues(*c, mgr.getASTContext(), mgr.getDiagnostic());
}


static void ActionGRExprEngine(AnalysisManager& mgr, Decl *D, 
                               GRTransferFuncs* tf) {


  llvm::OwningPtr<GRTransferFuncs> TF(tf);

  // Display progress.
  mgr.DisplayFunction(D);

  // Construct the analysis engine.  We first query for the LiveVariables
  // information to see if the CFG is valid.
  // FIXME: Inter-procedural analysis will need to handle invalid CFGs.
  if (!mgr.getLiveVariables(D))
    return;  
  
  GRExprEngine Eng(mgr);

  Eng.setTransferFunctions(tf);
  Eng.RegisterInternalChecks(); // FIXME: Internal checks should just
                                // automatically register.
  RegisterAppleChecks(Eng, *D);


  // Set the graph auditor.
  llvm::OwningPtr<ExplodedNode::Auditor> Auditor;
  if (mgr.shouldVisualizeUbigraph()) {
    Auditor.reset(CreateUbiViz());
    ExplodedNode::SetAuditor(Auditor.get());
  }

  // Execute the worklist algorithm.
  Eng.ExecuteWorkList(mgr.getStackFrame(D));

  // Release the auditor (if any) so that it doesn't monitor the graph
  // created BugReporter.
  ExplodedNode::SetAuditor(0);

  // Visualize the exploded graph.
  if (mgr.shouldVisualizeGraphviz())
    Eng.ViewGraph(mgr.shouldTrimGraph());

  // Display warnings.
  Eng.getBugReporter().FlushReports();
}

static void ActionCheckerCFRefAux(AnalysisManager& mgr, Decl *D,
                                  bool GCEnabled) {

  GRTransferFuncs* TF = MakeCFRefCountTF(mgr.getASTContext(),
                                         GCEnabled,
                                         mgr.getLangOptions());

  ActionGRExprEngine(mgr, D, TF);
}

static void ActionCheckerCFRef(AnalysisManager& mgr, Decl *D) {

 switch (mgr.getLangOptions().getGCMode()) {
   default:
     assert (false && "Invalid GC mode.");
   case LangOptions::NonGC:
     ActionCheckerCFRefAux(mgr, D, false);
     break;

   case LangOptions::GCOnly:
     ActionCheckerCFRefAux(mgr, D, true);
     break;

   case LangOptions::HybridGC:
     ActionCheckerCFRefAux(mgr, D, false);
     ActionCheckerCFRefAux(mgr, D, true);
     break;
 }
}

static void ActionDisplayLiveVariables(AnalysisManager& mgr, Decl *D) {
  if (LiveVariables* L = mgr.getLiveVariables(D)) {
    mgr.DisplayFunction(D);
    L->dumpBlockLiveness(mgr.getSourceManager());
  }
}

static void ActionCFGDump(AnalysisManager& mgr, Decl *D) {
  if (CFG* c = mgr.getCFG(D)) {
    mgr.DisplayFunction(D);
    c->dump(mgr.getLangOptions());
  }
}

static void ActionCFGView(AnalysisManager& mgr, Decl *D) {
  if (CFG* c = mgr.getCFG(D)) {
    mgr.DisplayFunction(D);
    c->viewCFG(mgr.getLangOptions());
  }
}

static void ActionSecuritySyntacticChecks(AnalysisManager &mgr, Decl *D) {
  BugReporter BR(mgr);
  CheckSecuritySyntaxOnly(D, BR);
}

static void ActionWarnObjCDealloc(AnalysisManager& mgr, Decl *D) {
  if (mgr.getLangOptions().getGCMode() == LangOptions::GCOnly)
    return;

  BugReporter BR(mgr);
  CheckObjCDealloc(cast<ObjCImplementationDecl>(D), mgr.getLangOptions(), BR);  
}

static void ActionWarnObjCUnusedIvars(AnalysisManager& mgr, Decl *D) {
  BugReporter BR(mgr);
  CheckObjCUnusedIvar(cast<ObjCImplementationDecl>(D), BR);  
}

static void ActionWarnObjCMethSigs(AnalysisManager& mgr, Decl *D) {
  BugReporter BR(mgr);

  CheckObjCInstMethSignature(cast<ObjCImplementationDecl>(D), BR);
}

static void ActionInlineCall(AnalysisManager &mgr, Decl *D) {
  if (!D)
    return;

  llvm::OwningPtr<GRTransferFuncs> TF(CreateCallInliner(mgr.getASTContext()));

  // Construct the analysis engine.
  GRExprEngine Eng(mgr);

  Eng.setTransferFunctions(TF.get());
  
  Eng.RegisterInternalChecks();
  RegisterAppleChecks(Eng, *D);

  // Execute the worklist algorithm.
  Eng.ExecuteWorkList(mgr.getStackFrame(D));
  
  // Visualize the exploded graph.
  if (mgr.shouldVisualizeGraphviz())
    Eng.ViewGraph(mgr.shouldTrimGraph());
}

//===----------------------------------------------------------------------===//
// AnalysisConsumer creation.
//===----------------------------------------------------------------------===//

ASTConsumer* clang::CreateAnalysisConsumer(Diagnostic &diags, Preprocessor* pp,
                                           const LangOptions& lopts,
                                           const std::string& OutDir,
                                           const AnalyzerOptions& Opts) {

  llvm::OwningPtr<AnalysisConsumer> C(new AnalysisConsumer(diags, pp,
                                                           lopts, OutDir,
                                                           Opts));

  for (unsigned i = 0; i < Opts.AnalysisList.size(); ++i)
    switch (Opts.AnalysisList[i]) {
#define ANALYSIS(NAME, CMD, DESC, SCOPE)\
      case NAME:\
        C->add ## SCOPE ## Action(&Action ## NAME);\
        break;
#include "clang/Frontend/Analyses.def"
      default: break;
    }

  // Last, disable the effects of '-Werror' when using the AnalysisConsumer.
  diags.setWarningsAsErrors(false);

  return C.take();
}

//===----------------------------------------------------------------------===//
// Ubigraph Visualization.  FIXME: Move to separate file.
//===----------------------------------------------------------------------===//

namespace {

class UbigraphViz : public ExplodedNode::Auditor {
  llvm::OwningPtr<llvm::raw_ostream> Out;
  llvm::sys::Path Dir, Filename;
  unsigned Cntr;

  typedef llvm::DenseMap<void*,unsigned> VMap;
  VMap M;

public:
  UbigraphViz(llvm::raw_ostream* out, llvm::sys::Path& dir,
              llvm::sys::Path& filename);

  ~UbigraphViz();

  virtual void AddEdge(ExplodedNode* Src, ExplodedNode* Dst);
};

} // end anonymous namespace

static ExplodedNode::Auditor* CreateUbiViz() {
  std::string ErrMsg;

  llvm::sys::Path Dir = llvm::sys::Path::GetTemporaryDirectory(&ErrMsg);
  if (!ErrMsg.empty())
    return 0;

  llvm::sys::Path Filename = Dir;
  Filename.appendComponent("llvm_ubi");
  Filename.makeUnique(true,&ErrMsg);

  if (!ErrMsg.empty())
    return 0;

  llvm::errs() << "Writing '" << Filename.str() << "'.\n";

  llvm::OwningPtr<llvm::raw_fd_ostream> Stream;
  Stream.reset(new llvm::raw_fd_ostream(Filename.c_str(), ErrMsg));

  if (!ErrMsg.empty())
    return 0;

  return new UbigraphViz(Stream.take(), Dir, Filename);
}

void UbigraphViz::AddEdge(ExplodedNode* Src, ExplodedNode* Dst) {

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
  llvm::errs() << "Running 'ubiviz' program... ";
  std::string ErrMsg;
  llvm::sys::Path Ubiviz = llvm::sys::Program::FindProgramByName("ubiviz");
  std::vector<const char*> args;
  args.push_back(Ubiviz.c_str());
  args.push_back(Filename.c_str());
  args.push_back(0);

  if (llvm::sys::Program::ExecuteAndWait(Ubiviz, &args[0],0,0,0,0,&ErrMsg)) {
    llvm::errs() << "Error viewing graph: " << ErrMsg << "\n";
  }

  // Delete the directory.
  Dir.eraseFromDisk(true);
}
