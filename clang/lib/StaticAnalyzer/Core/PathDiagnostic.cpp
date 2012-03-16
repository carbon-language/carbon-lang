//===--- PathDiagnostic.cpp - Path-Specific Diagnostic Handling -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PathDiagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace ento;

bool PathDiagnosticMacroPiece::containsEvent() const {
  for (PathPieces::const_iterator I = subPieces.begin(), E = subPieces.end();
       I!=E; ++I) {
    if (isa<PathDiagnosticEventPiece>(*I))
      return true;
    if (PathDiagnosticMacroPiece *MP = dyn_cast<PathDiagnosticMacroPiece>(*I))
      if (MP->containsEvent())
        return true;
  }
  return false;
}

static StringRef StripTrailingDots(StringRef s) {
  for (StringRef::size_type i = s.size(); i != 0; --i)
    if (s[i - 1] != '.')
      return s.substr(0, i);
  return "";
}

PathDiagnosticPiece::PathDiagnosticPiece(StringRef s,
                                         Kind k, DisplayHint hint)
  : str(StripTrailingDots(s)), kind(k), Hint(hint) {}

PathDiagnosticPiece::PathDiagnosticPiece(Kind k, DisplayHint hint)
  : kind(k), Hint(hint) {}

PathDiagnosticPiece::~PathDiagnosticPiece() {}
PathDiagnosticEventPiece::~PathDiagnosticEventPiece() {}
PathDiagnosticCallPiece::~PathDiagnosticCallPiece() {}
PathDiagnosticControlFlowPiece::~PathDiagnosticControlFlowPiece() {}
PathDiagnosticMacroPiece::~PathDiagnosticMacroPiece() {}
PathDiagnostic::PathDiagnostic() : path(pathImpl) {}
PathPieces::~PathPieces() {}
PathDiagnostic::~PathDiagnostic() {}

PathDiagnostic::PathDiagnostic(StringRef bugtype, StringRef desc,
                               StringRef category)
  : BugType(StripTrailingDots(bugtype)),
    Desc(StripTrailingDots(desc)),
    Category(StripTrailingDots(category)),
    path(pathImpl) {}

void PathDiagnosticConsumer::anchor() { }

PathDiagnosticConsumer::~PathDiagnosticConsumer() {
  // Delete the contents of the FoldingSet if it isn't empty already.
  for (llvm::FoldingSet<PathDiagnostic>::iterator it =
       Diags.begin(), et = Diags.end() ; it != et ; ++it) {
    delete &*it;
  }
}

void PathDiagnosticConsumer::HandlePathDiagnostic(PathDiagnostic *D) {
  llvm::OwningPtr<PathDiagnostic> OwningD(D);
  
  if (!D || D->path.empty())
    return;
  
  // We need to flatten the locations (convert Stmt* to locations) because
  // the referenced statements may be freed by the time the diagnostics
  // are emitted.
  D->flattenLocations();

  // If the PathDiagnosticConsumer does not support diagnostics that
  // cross file boundaries, prune out such diagnostics now.
  if (!supportsCrossFileDiagnostics()) {
    // Verify that the entire path is from the same FileID.
    FileID FID;
    const SourceManager &SMgr = (*D->path.begin())->getLocation().getManager();
    llvm::SmallVector<const PathPieces *, 5> WorkList;
    WorkList.push_back(&D->path);

    while (!WorkList.empty()) {
      const PathPieces &path = *WorkList.back();
      WorkList.pop_back();

      for (PathPieces::const_iterator I = path.begin(), E = path.end();
           I != E; ++I) {
        const PathDiagnosticPiece *piece = I->getPtr();
        FullSourceLoc L = piece->getLocation().asLocation().getExpansionLoc();
      
        if (FID.isInvalid()) {
          FID = SMgr.getFileID(L);
        } else if (SMgr.getFileID(L) != FID)
          return; // FIXME: Emit a warning?
      
        // Check the source ranges.
        for (PathDiagnosticPiece::range_iterator RI = piece->ranges_begin(),
             RE = piece->ranges_end();
             RI != RE; ++RI) {
          SourceLocation L = SMgr.getExpansionLoc(RI->getBegin());
          if (!L.isFileID() || SMgr.getFileID(L) != FID)
            return; // FIXME: Emit a warning?
          L = SMgr.getExpansionLoc(RI->getEnd());
          if (!L.isFileID() || SMgr.getFileID(L) != FID)
            return; // FIXME: Emit a warning?
        }
        
        if (const PathDiagnosticCallPiece *call =
            dyn_cast<PathDiagnosticCallPiece>(piece)) {
          WorkList.push_back(&call->path);
        }
        else if (const PathDiagnosticMacroPiece *macro =
                 dyn_cast<PathDiagnosticMacroPiece>(piece)) {
          WorkList.push_back(&macro->subPieces);
        }
      }
    }
    
    if (FID.isInvalid())
      return; // FIXME: Emit a warning?
  }  

  // Profile the node to see if we already have something matching it
  llvm::FoldingSetNodeID profile;
  D->Profile(profile);
  void *InsertPos = 0;

  if (PathDiagnostic *orig = Diags.FindNodeOrInsertPos(profile, InsertPos)) {
    // Keep the PathDiagnostic with the shorter path.
    const unsigned orig_size = orig->full_size();
    const unsigned new_size = D->full_size();
    
    if (orig_size <= new_size) {
      bool shouldKeepOriginal = true;
      if (orig_size == new_size) {
        // Here we break ties in a fairly arbitrary, but deterministic, way.
        llvm::FoldingSetNodeID fullProfile, fullProfileOrig;
        D->FullProfile(fullProfile);
        orig->FullProfile(fullProfileOrig);
        if (fullProfile.ComputeHash() < fullProfileOrig.ComputeHash())
          shouldKeepOriginal = false;
      }

      if (shouldKeepOriginal)
        return;
    }
    Diags.RemoveNode(orig);
    delete orig;
  }
  
  Diags.InsertNode(OwningD.take());
}


namespace {
struct CompareDiagnostics {
  // Compare if 'X' is "<" than 'Y'.
  bool operator()(const PathDiagnostic *X, const PathDiagnostic *Y) const {
    // First compare by location
    const FullSourceLoc &XLoc = X->getLocation().asLocation();
    const FullSourceLoc &YLoc = Y->getLocation().asLocation();
    if (XLoc < YLoc)
      return true;
    if (XLoc != YLoc)
      return false;
    
    // Next, compare by bug type.
    StringRef XBugType = X->getBugType();
    StringRef YBugType = Y->getBugType();
    if (XBugType < YBugType)
      return true;
    if (XBugType != YBugType)
      return false;
    
    // Next, compare by bug description.
    StringRef XDesc = X->getDescription();
    StringRef YDesc = Y->getDescription();
    if (XDesc < YDesc)
      return true;
    if (XDesc != YDesc)
      return false;
    
    // FIXME: Further refine by comparing PathDiagnosticPieces?
    return false;    
  }  
};  
}

void
PathDiagnosticConsumer::FlushDiagnostics(SmallVectorImpl<std::string> *Files) {
  if (flushed)
    return;
  
  flushed = true;
  
  std::vector<const PathDiagnostic *> BatchDiags;
  for (llvm::FoldingSet<PathDiagnostic>::iterator it = Diags.begin(),
       et = Diags.end(); it != et; ++it) {
    BatchDiags.push_back(&*it);
  }
  
  // Clear out the FoldingSet.
  Diags.clear();

  // Sort the diagnostics so that they are always emitted in a deterministic
  // order.
  if (!BatchDiags.empty())
    std::sort(BatchDiags.begin(), BatchDiags.end(), CompareDiagnostics());
  
  FlushDiagnosticsImpl(BatchDiags, Files);

  // Delete the flushed diagnostics.
  for (std::vector<const PathDiagnostic *>::iterator it = BatchDiags.begin(),
       et = BatchDiags.end(); it != et; ++it) {
    const PathDiagnostic *D = *it;
    delete D;
  }
}

//===----------------------------------------------------------------------===//
// PathDiagnosticLocation methods.
//===----------------------------------------------------------------------===//

static SourceLocation getValidSourceLocation(const Stmt* S,
                                             LocationOrAnalysisDeclContext LAC) {
  SourceLocation L = S->getLocStart();
  assert(!LAC.isNull() && "A valid LocationContext or AnalysisDeclContext should "
                          "be passed to PathDiagnosticLocation upon creation.");

  // S might be a temporary statement that does not have a location in the
  // source code, so find an enclosing statement and use it's location.
  if (!L.isValid()) {

    ParentMap *PM = 0;
    if (LAC.is<const LocationContext*>())
      PM = &LAC.get<const LocationContext*>()->getParentMap();
    else
      PM = &LAC.get<AnalysisDeclContext*>()->getParentMap();

    while (!L.isValid()) {
      S = PM->getParent(S);
      L = S->getLocStart();
    }
  }

  return L;
}

PathDiagnosticLocation
  PathDiagnosticLocation::createBegin(const Decl *D,
                                      const SourceManager &SM) {
  return PathDiagnosticLocation(D->getLocStart(), SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createBegin(const Stmt *S,
                                      const SourceManager &SM,
                                      LocationOrAnalysisDeclContext LAC) {
  return PathDiagnosticLocation(getValidSourceLocation(S, LAC),
                                SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createOperatorLoc(const BinaryOperator *BO,
                                            const SourceManager &SM) {
  return PathDiagnosticLocation(BO->getOperatorLoc(), SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createMemberLoc(const MemberExpr *ME,
                                          const SourceManager &SM) {
  return PathDiagnosticLocation(ME->getMemberLoc(), SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createBeginBrace(const CompoundStmt *CS,
                                           const SourceManager &SM) {
  SourceLocation L = CS->getLBracLoc();
  return PathDiagnosticLocation(L, SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createEndBrace(const CompoundStmt *CS,
                                         const SourceManager &SM) {
  SourceLocation L = CS->getRBracLoc();
  return PathDiagnosticLocation(L, SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::createDeclBegin(const LocationContext *LC,
                                          const SourceManager &SM) {
  // FIXME: Should handle CXXTryStmt if analyser starts supporting C++.
  if (const CompoundStmt *CS =
        dyn_cast_or_null<CompoundStmt>(LC->getDecl()->getBody()))
    if (!CS->body_empty()) {
      SourceLocation Loc = (*CS->body_begin())->getLocStart();
      return PathDiagnosticLocation(Loc, SM, SingleLocK);
    }

  return PathDiagnosticLocation();
}

PathDiagnosticLocation
  PathDiagnosticLocation::createDeclEnd(const LocationContext *LC,
                                        const SourceManager &SM) {
  SourceLocation L = LC->getDecl()->getBodyRBrace();
  return PathDiagnosticLocation(L, SM, SingleLocK);
}

PathDiagnosticLocation
  PathDiagnosticLocation::create(const ProgramPoint& P,
                                 const SourceManager &SMng) {

  const Stmt* S = 0;
  if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
    const CFGBlock *BSrc = BE->getSrc();
    S = BSrc->getTerminatorCondition();
  }
  else if (const PostStmt *PS = dyn_cast<PostStmt>(&P)) {
    S = PS->getStmt();
  }

  return PathDiagnosticLocation(S, SMng, P.getLocationContext());
}

PathDiagnosticLocation
  PathDiagnosticLocation::createEndOfPath(const ExplodedNode* N,
                                          const SourceManager &SM) {
  assert(N && "Cannot create a location with a null node.");

  const ExplodedNode *NI = N;

  while (NI) {
    ProgramPoint P = NI->getLocation();
    const LocationContext *LC = P.getLocationContext();
    if (const StmtPoint *PS = dyn_cast<StmtPoint>(&P))
      return PathDiagnosticLocation(PS->getStmt(), SM, LC);
    else if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
      const Stmt *Term = BE->getSrc()->getTerminator();
      if (Term) {
        return PathDiagnosticLocation(Term, SM, LC);
      }
    }
    NI = NI->succ_empty() ? 0 : *(NI->succ_begin());
  }

  return createDeclEnd(N->getLocationContext(), SM);
}

PathDiagnosticLocation PathDiagnosticLocation::createSingleLocation(
                                           const PathDiagnosticLocation &PDL) {
  FullSourceLoc L = PDL.asLocation();
  return PathDiagnosticLocation(L, L.getManager(), SingleLocK);
}

FullSourceLoc
  PathDiagnosticLocation::genLocation(SourceLocation L,
                                      LocationOrAnalysisDeclContext LAC) const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
    case RangeK:
      break;
    case StmtK:
      // Defensive checking.
      if (!S)
        break;
      return FullSourceLoc(getValidSourceLocation(S, LAC),
                           const_cast<SourceManager&>(*SM));
    case DeclK:
      // Defensive checking.
      if (!D)
        break;
      return FullSourceLoc(D->getLocation(), const_cast<SourceManager&>(*SM));
  }

  return FullSourceLoc(L, const_cast<SourceManager&>(*SM));
}

PathDiagnosticRange
  PathDiagnosticLocation::genRange(LocationOrAnalysisDeclContext LAC) const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
      return PathDiagnosticRange(SourceRange(Loc,Loc), true);
    case RangeK:
      break;
    case StmtK: {
      const Stmt *S = asStmt();
      switch (S->getStmtClass()) {
        default:
          break;
        case Stmt::DeclStmtClass: {
          const DeclStmt *DS = cast<DeclStmt>(S);
          if (DS->isSingleDecl()) {
            // Should always be the case, but we'll be defensive.
            return SourceRange(DS->getLocStart(),
                               DS->getSingleDecl()->getLocation());
          }
          break;
        }
          // FIXME: Provide better range information for different
          //  terminators.
        case Stmt::IfStmtClass:
        case Stmt::WhileStmtClass:
        case Stmt::DoStmtClass:
        case Stmt::ForStmtClass:
        case Stmt::ChooseExprClass:
        case Stmt::IndirectGotoStmtClass:
        case Stmt::SwitchStmtClass:
        case Stmt::BinaryConditionalOperatorClass:
        case Stmt::ConditionalOperatorClass:
        case Stmt::ObjCForCollectionStmtClass: {
          SourceLocation L = getValidSourceLocation(S, LAC);
          return SourceRange(L, L);
        }
      }
      SourceRange R = S->getSourceRange();
      if (R.isValid())
        return R;
      break;  
    }
    case DeclK:
      if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
        return MD->getSourceRange();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        if (Stmt *Body = FD->getBody())
          return Body->getSourceRange();
      }
      else {
        SourceLocation L = D->getLocation();
        return PathDiagnosticRange(SourceRange(L, L), true);
      }
  }

  return SourceRange(Loc,Loc);
}

void PathDiagnosticLocation::flatten() {
  if (K == StmtK) {
    K = RangeK;
    S = 0;
    D = 0;
  }
  else if (K == DeclK) {
    K = SingleLocK;
    S = 0;
    D = 0;
  }
}

PathDiagnosticLocation PathDiagnostic::getLocation() const {
  assert(path.size() > 0 &&
         "getLocation() requires a non-empty PathDiagnostic.");
  
  PathDiagnosticPiece *p = path.rbegin()->getPtr();
  
  while (true) {
    if (PathDiagnosticCallPiece *cp = dyn_cast<PathDiagnosticCallPiece>(p)) {
      assert(!cp->path.empty());
      p = cp->path.rbegin()->getPtr();
      continue;
    }
    break;
  }
  
  return p->getLocation();
}

//===----------------------------------------------------------------------===//
// Manipulation of PathDiagnosticCallPieces.
//===----------------------------------------------------------------------===//

static PathDiagnosticLocation getLastStmtLoc(const ExplodedNode *N,
                                             const SourceManager &SM) {
  while (N) {
    ProgramPoint PP = N->getLocation();
    if (const StmtPoint *SP = dyn_cast<StmtPoint>(&PP))
      return PathDiagnosticLocation(SP->getStmt(), SM, PP.getLocationContext());
    if (N->pred_empty())
      break;
    N = *N->pred_begin();
  }
  return PathDiagnosticLocation();
}

PathDiagnosticCallPiece *
PathDiagnosticCallPiece::construct(const ExplodedNode *N,
                                   const CallExit &CE,
                                   const SourceManager &SM) {
  const Decl *caller = CE.getLocationContext()->getParent()->getDecl();
  PathDiagnosticLocation pos = getLastStmtLoc(N, SM);
  return new PathDiagnosticCallPiece(caller, pos);
}

PathDiagnosticCallPiece *
PathDiagnosticCallPiece::construct(PathPieces &path,
                                   const Decl *caller) {
  PathDiagnosticCallPiece *C = new PathDiagnosticCallPiece(path, caller);
  path.clear();
  path.push_front(C);
  return C;
}

void PathDiagnosticCallPiece::setCallee(const CallEnter &CE,
                                        const SourceManager &SM) {
  const Decl *D = CE.getCalleeContext()->getDecl();
  Callee = D;
  callEnter = PathDiagnosticLocation(CE.getCallExpr(), SM,
                                     CE.getLocationContext());
  callEnterWithin = PathDiagnosticLocation::createBegin(D, SM);
}

IntrusiveRefCntPtr<PathDiagnosticEventPiece>
PathDiagnosticCallPiece::getCallEnterEvent() const {
  if (!Callee)
    return 0;  
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  if (isa<BlockDecl>(Callee))
    Out << "Calling anonymous block";
  else if (const NamedDecl *ND = dyn_cast<NamedDecl>(Callee))
    Out << "Calling '" << *ND << "'";
  StringRef msg = Out.str();
  if (msg.empty())
    return 0;
  return new PathDiagnosticEventPiece(callEnter, msg);
}

IntrusiveRefCntPtr<PathDiagnosticEventPiece>
PathDiagnosticCallPiece::getCallEnterWithinCallerEvent() const {
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(Caller))
    Out << "Entered call from '" << *ND << "'";
  else
    Out << "Entered call";
  StringRef msg = Out.str();
  if (msg.empty())
    return 0;
  return new PathDiagnosticEventPiece(callEnterWithin, msg);
}

IntrusiveRefCntPtr<PathDiagnosticEventPiece>
PathDiagnosticCallPiece::getCallExitEvent() const {
  if (NoExit)
    return 0;
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  if (!CallStackMessage.empty())
    Out << CallStackMessage;
  else if (const NamedDecl *ND = dyn_cast_or_null<NamedDecl>(Callee))
    Out << "Returning from '" << *ND << "'";
  else
    Out << "Returning to caller";
  return new PathDiagnosticEventPiece(callReturn, Out.str());
}

static void compute_path_size(const PathPieces &pieces, unsigned &size) {
  for (PathPieces::const_iterator it = pieces.begin(),
                                  et = pieces.end(); it != et; ++it) {
    const PathDiagnosticPiece *piece = it->getPtr();
    if (const PathDiagnosticCallPiece *cp = 
        dyn_cast<PathDiagnosticCallPiece>(piece)) {
      compute_path_size(cp->path, size);
    }
    else
      ++size;
  }
}

unsigned PathDiagnostic::full_size() {
  unsigned size = 0;
  compute_path_size(path, size);
  return size;
}

//===----------------------------------------------------------------------===//
// FoldingSet profiling methods.
//===----------------------------------------------------------------------===//

void PathDiagnosticLocation::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger(Range.getBegin().getRawEncoding());
  ID.AddInteger(Range.getEnd().getRawEncoding());
  ID.AddInteger(Loc.getRawEncoding());
  return;
}

void PathDiagnosticPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger((unsigned) getKind());
  ID.AddString(str);
  // FIXME: Add profiling support for code hints.
  ID.AddInteger((unsigned) getDisplayHint());
  for (range_iterator I = ranges_begin(), E = ranges_end(); I != E; ++I) {
    ID.AddInteger(I->getBegin().getRawEncoding());
    ID.AddInteger(I->getEnd().getRawEncoding());
  }  
}

void PathDiagnosticCallPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  for (PathPieces::const_iterator it = path.begin(), 
       et = path.end(); it != et; ++it) {
    ID.Add(**it);
  }
}

void PathDiagnosticSpotPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  ID.Add(Pos);
}

void PathDiagnosticControlFlowPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    ID.Add(*I);
}

void PathDiagnosticMacroPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticSpotPiece::Profile(ID);
  for (PathPieces::const_iterator I = subPieces.begin(), E = subPieces.end();
       I != E; ++I)
    ID.Add(**I);
}

void PathDiagnostic::Profile(llvm::FoldingSetNodeID &ID) const {
  if (!path.empty())
    getLocation().Profile(ID);
  ID.AddString(BugType);
  ID.AddString(Desc);
  ID.AddString(Category);
}

void PathDiagnostic::FullProfile(llvm::FoldingSetNodeID &ID) const {
  Profile(ID);
  for (PathPieces::const_iterator I = path.begin(), E = path.end(); I != E; ++I)
    ID.Add(**I);
  for (meta_iterator I = meta_begin(), E = meta_end(); I != E; ++I)
    ID.AddString(*I);
}

StackHintGenerator::~StackHintGenerator() {}

std::string StackHintGeneratorForSymbol::getMessage(const ExplodedNode *N){
  ProgramPoint P = N->getLocation();
  const CallExit *CExit = dyn_cast<CallExit>(&P);
  assert(CExit && "Stack Hints should be constructed at CallExit points.");

  const CallExpr *CE = dyn_cast_or_null<CallExpr>(CExit->getStmt());
  if (!CE)
    return "";

  // Get the successor node to make sure the return statement is evaluated and
  // CE is set to the result value.
  N = *N->succ_begin();
  if (!N)
    return getMessageForSymbolNotFound();

  // Check if one of the parameters are set to the interesting symbol.
  ProgramStateRef State = N->getState();
  const LocationContext *LCtx = N->getLocationContext();
  unsigned ArgIndex = 0;
  for (CallExpr::const_arg_iterator I = CE->arg_begin(),
                                    E = CE->arg_end(); I != E; ++I, ++ArgIndex){
    SVal SV = State->getSVal(*I, LCtx);

    // Check if the variable corresponding to the symbol is passed by value.
    SymbolRef AS = SV.getAsLocSymbol();
    if (AS == Sym) {
      return getMessageForArg(*I, ArgIndex);
    }

    // Check if the parameter is a pointer to the symbol.
    if (const loc::MemRegionVal *Reg = dyn_cast<loc::MemRegionVal>(&SV)) {
      SVal PSV = State->getSVal(Reg->getRegion());
      SymbolRef AS = PSV.getAsLocSymbol();
      if (AS == Sym) {
        return getMessageForArg(*I, ArgIndex);
      }
    }
  }

  // Check if we are returning the interesting symbol.
  SVal SV = State->getSVal(CE, LCtx);
  SymbolRef RetSym = SV.getAsLocSymbol();
  if (RetSym == Sym) {
    return getMessageForReturn(CE);
  }

  return getMessageForSymbolNotFound();
}

/// TODO: This is copied from clang diagnostics. Maybe we could just move it to
/// some common place. (Same as HandleOrdinalModifier.)
void StackHintGeneratorForSymbol::printOrdinal(unsigned ValNo,
                                               llvm::raw_svector_ostream &Out) {
  assert(ValNo != 0 && "ValNo must be strictly positive!");

  // We could use text forms for the first N ordinals, but the numeric
  // forms are actually nicer in diagnostics because they stand out.
  Out << ValNo;

  // It is critically important that we do this perfectly for
  // user-written sequences with over 100 elements.
  switch (ValNo % 100) {
  case 11:
  case 12:
  case 13:
    Out << "th"; return;
  default:
    switch (ValNo % 10) {
    case 1: Out << "st"; return;
    case 2: Out << "nd"; return;
    case 3: Out << "rd"; return;
    default: Out << "th"; return;
    }
  }
}

std::string StackHintGeneratorForSymbol::getMessageForArg(const Expr *ArgE,
                                                        unsigned ArgIndex) {
  SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);

  os << Msg << " via ";
  // Printed parameters start at 1, not 0.
  printOrdinal(++ArgIndex, os);
  os << " parameter";

  return os.str();
}
