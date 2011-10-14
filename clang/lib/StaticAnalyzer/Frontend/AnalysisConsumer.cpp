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

#include "AnalysisConsumer.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/CFG.h"
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
#include "llvm/ADT/OwningPtr.h"

using namespace clang;
using namespace ento;

static ExplodedNode::Auditor* CreateUbiViz();

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

class AnalysisConsumer : public ASTConsumer {
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

  llvm::OwningPtr<CheckerManager> checkerMgr;
  llvm::OwningPtr<AnalysisManager> Mgr;

  AnalysisConsumer(const Preprocessor& pp,
                   const std::string& outdir,
                   const AnalyzerOptions& opts,
                   ArrayRef<std::string> plugins)
    : Ctx(0), PP(pp), OutDir(outdir), Opts(opts), Plugins(plugins), PD(0) {
    DigestAnalyzerOptions();
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

  void DisplayFunction(const Decl *D) {
    if (!Opts.AnalyzerDisplayProgress)
      return;

    SourceManager &SM = Mgr->getASTContext().getSourceManager();
    PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
    if (Loc.isValid()) {
      llvm::errs() << "ANALYZE: " << Loc.getFilename();

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
    checkerMgr.reset(createCheckerManager(Opts, PP.getLangOptions(), Plugins,
                                          PP.getDiagnostics()));
    Mgr.reset(new AnalysisManager(*Ctx, PP.getDiagnostics(),
                                  PP.getLangOptions(), PD,
                                  CreateStoreMgr, CreateConstraintMgr,
                                  checkerMgr.get(),
                                  /* Indexer */ 0, 
                                  Opts.MaxNodes, Opts.MaxLoop,
                                  Opts.VisualizeEGDot, Opts.VisualizeEGUbi,
                                  Opts.AnalysisPurgeOpt, Opts.EagerlyAssume,
                                  Opts.TrimGraph, Opts.InlineCall,
                                  Opts.UnoptimizedCFG, Opts.CFGAddImplicitDtors,
                                  Opts.CFGAddInitializers,
                                  Opts.EagerlyTrimEGraph));
  }

  virtual void HandleTranslationUnit(ASTContext &C);
  void HandleDeclContext(ASTContext &C, DeclContext *dc);
  void HandleDeclContextDecl(ASTContext &C, Decl *D);

  void HandleCode(Decl *D);
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AnalysisConsumer implementation.
//===----------------------------------------------------------------------===//

void AnalysisConsumer::HandleDeclContext(ASTContext &C, DeclContext *dc) {
  for (DeclContext::decl_iterator I = dc->decls_begin(), E = dc->decls_end();
       I != E; ++I) {
    HandleDeclContextDecl(C, *I);
  }
}

void AnalysisConsumer::HandleDeclContextDecl(ASTContext &C, Decl *D) {
  { // Handle callbacks for arbitrary decls.
    BugReporter BR(*Mgr);
    checkerMgr->runCheckersOnASTDecl(D, *Mgr, BR);
  }

  switch (D->getKind()) {
    case Decl::Namespace: {
      HandleDeclContext(C, cast<NamespaceDecl>(D));
      break;
    }
    case Decl::CXXConstructor:
    case Decl::CXXDestructor:
    case Decl::CXXConversion:
    case Decl::CXXMethod:
    case Decl::Function: {
      FunctionDecl *FD = cast<FunctionDecl>(D);
      // We skip function template definitions, as their semantics is
      // only determined when they are instantiated.
      if (FD->isThisDeclarationADefinition() &&
          !FD->isDependentContext()) {
        if (!Opts.AnalyzeSpecificFunction.empty() &&
            FD->getDeclName().getAsString() != Opts.AnalyzeSpecificFunction)
          break;
        DisplayFunction(FD);
        HandleCode(FD);
      }
      break;
    }
     
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCImplementation: {
      ObjCImplDecl *ID = cast<ObjCImplDecl>(D);
      HandleCode(ID);
      
      for (ObjCContainerDecl::method_iterator MI = ID->meth_begin(), 
           ME = ID->meth_end(); MI != ME; ++MI) {
        BugReporter BR(*Mgr);
        checkerMgr->runCheckersOnASTDecl(*MI, *Mgr, BR);

        if ((*MI)->isThisDeclarationADefinition()) {
          if (!Opts.AnalyzeSpecificFunction.empty() &&
              Opts.AnalyzeSpecificFunction != 
                (*MI)->getSelector().getAsString())
            continue;
          DisplayFunction(*MI);
          HandleCode(*MI);
        }
      }
      break;
    }
      
    default:
      break;
  }
}

void AnalysisConsumer::HandleTranslationUnit(ASTContext &C) {
  BugReporter BR(*Mgr);
  TranslationUnitDecl *TU = C.getTranslationUnitDecl();
  checkerMgr->runCheckersOnASTDecl(TU, *Mgr, BR);
  HandleDeclContext(C, TU);

  // After all decls handled, run checkers on the entire TranslationUnit.
  checkerMgr->runCheckersOnEndOfTranslationUnit(TU, *Mgr, BR);

  // Explicitly destroy the PathDiagnosticConsumer.  This will flush its output.
  // FIXME: This should be replaced with something that doesn't rely on
  // side-effects in PathDiagnosticConsumer's destructor. This is required when
  // used with option -disable-free.
  Mgr.reset(NULL);
}

static void FindBlocks(DeclContext *D, SmallVectorImpl<Decl*> &WL) {
  if (BlockDecl *BD = dyn_cast<BlockDecl>(D))
    WL.push_back(BD);

  for (DeclContext::decl_iterator I = D->decls_begin(), E = D->decls_end();
       I!=E; ++I)
    if (DeclContext *DC = dyn_cast<DeclContext>(*I))
      FindBlocks(DC, WL);
}

static void RunPathSensitiveChecks(AnalysisConsumer &C, AnalysisManager &mgr,
                                   Decl *D);

void AnalysisConsumer::HandleCode(Decl *D) {

  // Don't run the actions if an error has occurred with parsing the file.
  DiagnosticsEngine &Diags = PP.getDiagnostics();
  if (Diags.hasErrorOccurred() || Diags.hasFatalErrorOccurred())
    return;

  // Don't run the actions on declarations in header files unless
  // otherwise specified.
  SourceManager &SM = Ctx->getSourceManager();
  SourceLocation SL = SM.getExpansionLoc(D->getLocation());
  if (!Opts.AnalyzeAll && !SM.isFromMainFile(SL))
    return;

  // Clear the AnalysisManager of old AnalysisContexts.
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
      checkerMgr->runCheckersOnASTBody(*WI, *Mgr, BR);
      if (checkerMgr->hasPathSensitiveCheckers())
        RunPathSensitiveChecks(*this, *Mgr, *WI);
    }
}

//===----------------------------------------------------------------------===//
// Path-sensitive checking.
//===----------------------------------------------------------------------===//

static void ActionExprEngine(AnalysisConsumer &C, AnalysisManager &mgr,
                             Decl *D, bool ObjCGCEnabled) {
  // Construct the analysis engine.  First check if the CFG is valid.
  // FIXME: Inter-procedural analysis will need to handle invalid CFGs.
  if (!mgr.getCFG(D))
    return;
  ExprEngine Eng(mgr, ObjCGCEnabled);

  // Set the graph auditor.
  llvm::OwningPtr<ExplodedNode::Auditor> Auditor;
  if (mgr.shouldVisualizeUbigraph()) {
    Auditor.reset(CreateUbiViz());
    ExplodedNode::SetAuditor(Auditor.get());
  }

  // Execute the worklist algorithm.
  Eng.ExecuteWorkList(mgr.getStackFrame(D, 0), mgr.getMaxNodes());

  // Release the auditor (if any) so that it doesn't monitor the graph
  // created BugReporter.
  ExplodedNode::SetAuditor(0);

  // Visualize the exploded graph.
  if (mgr.shouldVisualizeGraphviz())
    Eng.ViewGraph(mgr.shouldTrimGraph());

  // Display warnings.
  Eng.getBugReporter().FlushReports();
}

static void RunPathSensitiveChecks(AnalysisConsumer &C, AnalysisManager &mgr,
                                   Decl *D) {

  switch (mgr.getLangOptions().getGC()) {
  default:
    llvm_unreachable("Invalid GC mode.");
  case LangOptions::NonGC:
    ActionExprEngine(C, mgr, D, false);
    break;
  
  case LangOptions::GCOnly:
    ActionExprEngine(C, mgr, D, true);
    break;
  
  case LangOptions::HybridGC:
    ActionExprEngine(C, mgr, D, false);
    ActionExprEngine(C, mgr, D, true);
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
  llvm::OwningPtr<raw_ostream> Out;
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

  llvm::OwningPtr<llvm::raw_fd_ostream> Stream;
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
