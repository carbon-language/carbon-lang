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

#define DEBUG_TYPE "AnalysisConsumer"

#include "AnalysisConsumer.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Checkers/LocalCheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathDiagnosticConsumers.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/AnalyzerOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Timer.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Statistic.h"

#include <queue>

using namespace clang;
using namespace ento;
using llvm::SmallPtrSet;

static ExplodedNode::Auditor* CreateUbiViz();

STATISTIC(NumFunctionTopLevel, "The # of functions at top level.");
STATISTIC(NumFunctionsAnalyzed, "The # of functions analysed (as top level).");
STATISTIC(NumBlocksInAnalyzedFunctions,
                     "The # of basic blocks in the analyzed functions.");
STATISTIC(PercentReachableBlocks, "The % of reachable basic blocks.");

//===----------------------------------------------------------------------===//
// Special PathDiagnosticConsumers.
//===----------------------------------------------------------------------===//

static PathDiagnosticConsumer*
createPlistHTMLDiagnosticConsumer(const std::string& prefix,
                                const Preprocessor &PP) {
  PathDiagnosticConsumer *PD =
    createHTMLDiagnosticConsumer(llvm::sys::path::parent_path(prefix), PP);
  return createPlistDiagnosticConsumer(prefix, PP, PD);
}

//===----------------------------------------------------------------------===//
// AnalysisConsumer declaration.
//===----------------------------------------------------------------------===//

namespace {

class AnalysisConsumer : public ASTConsumer,
                         public RecursiveASTVisitor<AnalysisConsumer> {
  enum AnalysisMode {
    ANALYSIS_SYNTAX,
    ANALYSIS_PATH,
    ANALYSIS_ALL
  };

  /// Mode of the analyzes while recursively visiting Decls.
  AnalysisMode RecVisitorMode;
  /// Bug Reporter to use while recursively visiting Decls.
  BugReporter *RecVisitorBR;

public:
  ASTContext *Ctx;
  const Preprocessor &PP;
  const std::string OutDir;
  AnalyzerOptions Opts;
  ArrayRef<std::string> Plugins;

  // PD is owned by AnalysisManager.
  PathDiagnosticConsumer *PD;

  StoreManagerCreator CreateStoreMgr;
  ConstraintManagerCreator CreateConstraintMgr;

  OwningPtr<CheckerManager> checkerMgr;
  OwningPtr<AnalysisManager> Mgr;

  /// Time the analyzes time of each translation unit.
  static llvm::Timer* TUTotalTimer;

  /// The information about analyzed functions shared throughout the
  /// translation unit.
  FunctionSummariesTy FunctionSummaries;

  AnalysisConsumer(const Preprocessor& pp,
                   const std::string& outdir,
                   const AnalyzerOptions& opts,
                   ArrayRef<std::string> plugins)
    : RecVisitorMode(ANALYSIS_ALL), RecVisitorBR(0),
      Ctx(0), PP(pp), OutDir(outdir), Opts(opts), Plugins(plugins), PD(0) {
    DigestAnalyzerOptions();
    if (Opts.PrintStats) {
      llvm::EnableStatistics();
      TUTotalTimer = new llvm::Timer("Analyzer Total Time");
    }
  }

  ~AnalysisConsumer() {
    if (Opts.PrintStats)
      delete TUTotalTimer;
  }

  void DigestAnalyzerOptions() {
    // Create the PathDiagnosticConsumer.
    if (!OutDir.empty()) {
      switch (Opts.AnalysisDiagOpt) {
      default:
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATEFN, AUTOCREATE) \
        case PD_##NAME: PD = CREATEFN(OutDir, PP); break;
#include "clang/Frontend/Analyses.def"
      }
    } else if (Opts.AnalysisDiagOpt == PD_TEXT) {
      // Create the text client even without a specified output file since
      // it just uses diagnostic notes.
      PD = createTextPathDiagnosticConsumer("", PP);
    }

    // Create the analyzer component creators.
    switch (Opts.AnalysisStoreOpt) {
    default:
      llvm_unreachable("Unknown store manager.");
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATEFN)           \
      case NAME##Model: CreateStoreMgr = CREATEFN; break;
#include "clang/Frontend/Analyses.def"
    }

    switch (Opts.AnalysisConstraintsOpt) {
    default:
      llvm_unreachable("Unknown store manager.");
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATEFN)     \
      case NAME##Model: CreateConstraintMgr = CREATEFN; break;
#include "clang/Frontend/Analyses.def"
    }
  }

  void DisplayFunction(const Decl *D, AnalysisMode Mode) {
    if (!Opts.AnalyzerDisplayProgress)
      return;

    SourceManager &SM = Mgr->getASTContext().getSourceManager();
    PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
    if (Loc.isValid()) {
      llvm::errs() << "ANALYZE";
      switch (Mode) {
        case ANALYSIS_SYNTAX: llvm::errs() << "(Syntax)"; break;
        case ANALYSIS_PATH: llvm::errs() << "(Path Sensitive)"; break;
        case ANALYSIS_ALL: break;
      };
      llvm::errs() << ": " << Loc.getFilename();
      if (isa<FunctionDecl>(D) || isa<ObjCMethodDecl>(D)) {
        const NamedDecl *ND = cast<NamedDecl>(D);
        llvm::errs() << ' ' << *ND << '\n';
      }
      else if (isa<BlockDecl>(D)) {
        llvm::errs() << ' ' << "block(line:" << Loc.getLine() << ",col:"
                     << Loc.getColumn() << '\n';
      }
      else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
        Selector S = MD->getSelector();
        llvm::errs() << ' ' << S.getAsString();
      }
    }
  }

  virtual void Initialize(ASTContext &Context) {
    Ctx = &Context;
    checkerMgr.reset(createCheckerManager(Opts, PP.getLangOpts(), Plugins,
                                          PP.getDiagnostics()));
    Mgr.reset(new AnalysisManager(*Ctx, PP.getDiagnostics(),
                                  PP.getLangOpts(), PD,
                                  CreateStoreMgr, CreateConstraintMgr,
                                  checkerMgr.get(),
                                  /* Indexer */ 0, 
                                  Opts.MaxNodes, Opts.MaxLoop,
                                  Opts.VisualizeEGDot, Opts.VisualizeEGUbi,
                                  Opts.AnalysisPurgeOpt, Opts.EagerlyAssume,
                                  Opts.TrimGraph,
                                  Opts.UnoptimizedCFG, Opts.CFGAddImplicitDtors,
                                  Opts.CFGAddInitializers,
                                  Opts.EagerlyTrimEGraph,
                                  Opts.IPAMode,
                                  Opts.InlineMaxStackDepth,
                                  Opts.InlineMaxFunctionSize,
                                  Opts.InliningMode,
                                  Opts.NoRetryExhausted));
  }

  virtual void HandleTranslationUnit(ASTContext &C);

  /// \brief Build the call graph for the TU and use it to define the order
  /// in which the functions should be visited.
  void HandleDeclsGallGraph(TranslationUnitDecl *TU);

  /// \brief Run analyzes(syntax or path sensitive) on the given function.
  /// \param Mode - determines if we are requesting syntax only or path
  /// sensitive only analysis.
  /// \param VisitedCallees - The output parameter, which is populated with the
  /// set of functions which should be considered analyzed after analyzing the
  /// given root function.
  void HandleCode(Decl *D, AnalysisMode Mode, SetOfDecls *VisitedCallees = 0);

  /// \brief Check if we should skip (not analyze) the given function.
  bool skipFunction(Decl *D);

  void RunPathSensitiveChecks(Decl *D, SetOfDecls *VisitedCallees);
  void ActionExprEngine(Decl *D, bool ObjCGCEnabled, SetOfDecls *VisitedCallees);

  /// Visitors for the RecursiveASTVisitor.

  /// Handle callbacks for arbitrary Decls.
  bool VisitDecl(Decl *D) {
    checkerMgr->runCheckersOnASTDecl(D, *Mgr, *RecVisitorBR);
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    IdentifierInfo *II = FD->getIdentifier();
    if (II && II->getName().startswith("__inline"))
      return true;

    // We skip function template definitions, as their semantics is
    // only determined when they are instantiated.
    if (FD->isThisDeclarationADefinition() &&
        !FD->isDependentContext()) {
      HandleCode(FD, RecVisitorMode);
    }
    return true;
  }

  bool VisitObjCMethodDecl(ObjCMethodDecl *MD) {
    checkerMgr->runCheckersOnASTDecl(MD, *Mgr, *RecVisitorBR);
    if (MD->isThisDeclarationADefinition())
      HandleCode(MD, RecVisitorMode);
    return true;
  }
};
} // end anonymous namespace


//===----------------------------------------------------------------------===//
// AnalysisConsumer implementation.
//===----------------------------------------------------------------------===//
llvm::Timer* AnalysisConsumer::TUTotalTimer = 0;

void AnalysisConsumer::HandleDeclsGallGraph(TranslationUnitDecl *TU) {
  // Otherwise, use the Callgraph to derive the order.
  // Build the Call Graph.
  CallGraph CG;
  CG.addToCallGraph(TU);

  // Find the top level nodes - children of root + the unreachable (parentless)
  // nodes.
  llvm::SmallVector<CallGraphNode*, 24> TopLevelFunctions;
  for (CallGraph::nodes_iterator TI = CG.parentless_begin(),
                                 TE = CG.parentless_end(); TI != TE; ++TI) {
    TopLevelFunctions.push_back(*TI);
    NumFunctionTopLevel++;
  }
  CallGraphNode *Entry = CG.getRoot();
  for (CallGraphNode::iterator I = Entry->begin(),
                               E = Entry->end(); I != E; ++I) {
    TopLevelFunctions.push_back(*I);
    NumFunctionTopLevel++;
  }

  // Make sure the nodes are sorted in order reverse of their definition in the 
  // translation unit. This step is very important for performance. It ensures 
  // that we analyze the root functions before the externally available 
  // subroutines.
  std::queue<CallGraphNode*> BFSQueue;
  for (llvm::SmallVector<CallGraphNode*, 24>::reverse_iterator
         TI = TopLevelFunctions.rbegin(), TE = TopLevelFunctions.rend();
         TI != TE; ++TI)
    BFSQueue.push(*TI);

  // BFS over all of the functions, while skipping the ones inlined into
  // the previously processed functions. Use external Visited set, which is
  // also modified when we inline a function.
  SmallPtrSet<CallGraphNode*,24> Visited;
  while(!BFSQueue.empty()) {
    CallGraphNode *N = BFSQueue.front();
    BFSQueue.pop();

    // Skip the functions which have been processed already or previously
    // inlined.
    if (Visited.count(N))
      continue;

    // Analyze the function.
    SetOfDecls VisitedCallees;
    Decl *D = N->getDecl();
    assert(D);
    HandleCode(D, ANALYSIS_PATH,
               (Mgr->InliningMode == All ? 0 : &VisitedCallees));

    // Add the visited callees to the global visited set.
    for (SetOfDecls::const_iterator I = VisitedCallees.begin(),
                                    E = VisitedCallees.end(); I != E; ++I) {
      CallGraphNode *VN = CG.getNode(*I);
      if (VN)
        Visited.insert(VN);
    }
    Visited.insert(N);

    // Push the children into the queue.
    for (CallGraphNode::const_iterator CI = N->begin(),
                                       CE = N->end(); CI != CE; ++CI) {
      BFSQueue.push(*CI);
    }
  }
}

void AnalysisConsumer::HandleTranslationUnit(ASTContext &C) {
  // Don't run the actions if an error has occurred with parsing the file.
  DiagnosticsEngine &Diags = PP.getDiagnostics();
  if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
    return;

  {
    if (TUTotalTimer) TUTotalTimer->startTimer();

    // Introduce a scope to destroy BR before Mgr.
    BugReporter BR(*Mgr);
    TranslationUnitDecl *TU = C.getTranslationUnitDecl();
    checkerMgr->runCheckersOnASTDecl(TU, *Mgr, BR);

    // Run the AST-only checks using the order in which functions are defined.
    // If inlining is not turned on, use the simplest function order for path
    // sensitive analyzes as well.
    RecVisitorMode = (Mgr->shouldInlineCall() ? ANALYSIS_SYNTAX : ANALYSIS_ALL);
    RecVisitorBR = &BR;
    TraverseDecl(TU);

    if (Mgr->shouldInlineCall())
      HandleDeclsGallGraph(TU);

    // After all decls handled, run checkers on the entire TranslationUnit.
    checkerMgr->runCheckersOnEndOfTranslationUnit(TU, *Mgr, BR);

    RecVisitorBR = 0;
  }

  // Explicitly destroy the PathDiagnosticConsumer.  This will flush its output.
  // FIXME: This should be replaced with something that doesn't rely on
  // side-effects in PathDiagnosticConsumer's destructor. This is required when
  // used with option -disable-free.
  Mgr.reset(NULL);

  if (TUTotalTimer) TUTotalTimer->stopTimer();

  // Count how many basic blocks we have not covered.
  NumBlocksInAnalyzedFunctions = FunctionSummaries.getTotalNumBasicBlocks();
  if (NumBlocksInAnalyzedFunctions > 0)
    PercentReachableBlocks =
      (FunctionSummaries.getTotalNumVisitedBasicBlocks() * 100) /
        NumBlocksInAnalyzedFunctions;

}

static void FindBlocks(DeclContext *D, SmallVectorImpl<Decl*> &WL) {
  if (BlockDecl *BD = dyn_cast<BlockDecl>(D))
    WL.push_back(BD);

  for (DeclContext::decl_iterator I = D->decls_begin(), E = D->decls_end();
       I!=E; ++I)
    if (DeclContext *DC = dyn_cast<DeclContext>(*I))
      FindBlocks(DC, WL);
}

static std::string getFunctionName(const Decl *D) {
  if (const ObjCMethodDecl *ID = dyn_cast<ObjCMethodDecl>(D)) {
    return ID->getSelector().getAsString();
  }
  if (const FunctionDecl *ND = dyn_cast<FunctionDecl>(D)) {
    IdentifierInfo *II = ND->getIdentifier();
    if (II)
      return II->getName();
  }
  return "";
}

bool AnalysisConsumer::skipFunction(Decl *D) {
  if (!Opts.AnalyzeSpecificFunction.empty() &&
      getFunctionName(D) != Opts.AnalyzeSpecificFunction)
    return true;

  // Don't run the actions on declarations in header files unless
  // otherwise specified.
  SourceManager &SM = Ctx->getSourceManager();
  SourceLocation SL = SM.getExpansionLoc(D->getLocation());
  if (!Opts.AnalyzeAll && !SM.isFromMainFile(SL))
    return true;

  return false;
}

void AnalysisConsumer::HandleCode(Decl *D, AnalysisMode Mode,
                                  SetOfDecls *VisitedCallees) {
  if (skipFunction(D))
    return;

  DisplayFunction(D, Mode);

  // Clear the AnalysisManager of old AnalysisDeclContexts.
  Mgr->ClearContexts();

  // Dispatch on the actions.
  SmallVector<Decl*, 10> WL;
  WL.push_back(D);

  if (D->hasBody() && Opts.AnalyzeNestedBlocks)
    FindBlocks(cast<DeclContext>(D), WL);

  BugReporter BR(*Mgr);
  for (SmallVectorImpl<Decl*>::iterator WI=WL.begin(), WE=WL.end();
       WI != WE; ++WI)
    if ((*WI)->hasBody()) {
      if (Mode != ANALYSIS_PATH)
        checkerMgr->runCheckersOnASTBody(*WI, *Mgr, BR);
      if (Mode != ANALYSIS_SYNTAX && checkerMgr->hasPathSensitiveCheckers()) {
        RunPathSensitiveChecks(*WI, VisitedCallees);
        NumFunctionsAnalyzed++;
      }
    }
}

//===----------------------------------------------------------------------===//
// Path-sensitive checking.
//===----------------------------------------------------------------------===//

void AnalysisConsumer::ActionExprEngine(Decl *D, bool ObjCGCEnabled,
                                        SetOfDecls *VisitedCallees) {
  // Construct the analysis engine.  First check if the CFG is valid.
  // FIXME: Inter-procedural analysis will need to handle invalid CFGs.
  if (!Mgr->getCFG(D))
    return;

  ExprEngine Eng(*Mgr, ObjCGCEnabled, VisitedCallees, &FunctionSummaries);

  // Set the graph auditor.
  OwningPtr<ExplodedNode::Auditor> Auditor;
  if (Mgr->shouldVisualizeUbigraph()) {
    Auditor.reset(CreateUbiViz());
    ExplodedNode::SetAuditor(Auditor.get());
  }

  // Execute the worklist algorithm.
  Eng.ExecuteWorkList(Mgr->getAnalysisDeclContextManager().getStackFrame(D, 0),
                      Mgr->getMaxNodes());

  // Release the auditor (if any) so that it doesn't monitor the graph
  // created BugReporter.
  ExplodedNode::SetAuditor(0);

  // Visualize the exploded graph.
  if (Mgr->shouldVisualizeGraphviz())
    Eng.ViewGraph(Mgr->shouldTrimGraph());

  // Display warnings.
  Eng.getBugReporter().FlushReports();
}

void AnalysisConsumer::RunPathSensitiveChecks(Decl *D, SetOfDecls *Visited) {

  switch (Mgr->getLangOpts().getGC()) {
  case LangOptions::NonGC:
    ActionExprEngine(D, false, Visited);
    break;
  
  case LangOptions::GCOnly:
    ActionExprEngine(D, true, Visited);
    break;
  
  case LangOptions::HybridGC:
    ActionExprEngine(D, false, Visited);
    ActionExprEngine(D, true, Visited);
    break;
  }
}

//===----------------------------------------------------------------------===//
// AnalysisConsumer creation.
//===----------------------------------------------------------------------===//

ASTConsumer* ento::CreateAnalysisConsumer(const Preprocessor& pp,
                                          const std::string& outDir,
                                          const AnalyzerOptions& opts,
                                          ArrayRef<std::string> plugins) {
  // Disable the effects of '-Werror' when using the AnalysisConsumer.
  pp.getDiagnostics().setWarningsAsErrors(false);

  return new AnalysisConsumer(pp, outDir, opts, plugins);
}

//===----------------------------------------------------------------------===//
// Ubigraph Visualization.  FIXME: Move to separate file.
//===----------------------------------------------------------------------===//

namespace {

class UbigraphViz : public ExplodedNode::Auditor {
  OwningPtr<raw_ostream> Out;
  llvm::sys::Path Dir, Filename;
  unsigned Cntr;

  typedef llvm::DenseMap<void*,unsigned> VMap;
  VMap M;

public:
  UbigraphViz(raw_ostream *out, llvm::sys::Path& dir,
              llvm::sys::Path& filename);

  ~UbigraphViz();

  virtual void AddEdge(ExplodedNode *Src, ExplodedNode *Dst);
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

  OwningPtr<llvm::raw_fd_ostream> Stream;
  Stream.reset(new llvm::raw_fd_ostream(Filename.c_str(), ErrMsg));

  if (!ErrMsg.empty())
    return 0;

  return new UbigraphViz(Stream.take(), Dir, Filename);
}

void UbigraphViz::AddEdge(ExplodedNode *Src, ExplodedNode *Dst) {

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

UbigraphViz::UbigraphViz(raw_ostream *out, llvm::sys::Path& dir,
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
