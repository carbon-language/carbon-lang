//===--- JumpDiagnostics.cpp - Analyze Jump Targets for VLA issues --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JumpScopeChecker class, which is used to diagnose
// jumps that enter a VLA scope in an invalid way.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "Sema.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
using namespace clang;

namespace {

/// JumpScopeChecker - This object is used by Sema to diagnose invalid jumps
/// into VLA and other protected scopes.  For example, this rejects:
///    goto L;
///    int a[n];
///  L:
///
class JumpScopeChecker {
  Sema &S;

  /// GotoScope - This is a record that we use to keep track of all of the
  /// scopes that are introduced by VLAs and other things that scope jumps like
  /// gotos.  This scope tree has nothing to do with the source scope tree,
  /// because you can have multiple VLA scopes per compound statement, and most
  /// compound statements don't introduce any scopes.
  struct GotoScope {
    /// ParentScope - The index in ScopeMap of the parent scope.  This is 0 for
    /// the parent scope is the function body.
    unsigned ParentScope;

    /// InDiag - The diagnostic to emit if there is a jump into this scope.
    unsigned InDiag;

    /// OutDiag - The diagnostic to emit if there is an indirect jump out
    /// of this scope.  Direct jumps always clean up their current scope
    /// in an orderly way.
    unsigned OutDiag;

    /// Loc - Location to emit the diagnostic.
    SourceLocation Loc;

    GotoScope(unsigned parentScope, unsigned InDiag, unsigned OutDiag,
              SourceLocation L)
      : ParentScope(parentScope), InDiag(InDiag), OutDiag(OutDiag), Loc(L) {}
  };

  llvm::SmallVector<GotoScope, 48> Scopes;
  llvm::DenseMap<Stmt*, unsigned> LabelAndGotoScopes;
  llvm::SmallVector<Stmt*, 16> Jumps;

  llvm::SmallVector<IndirectGotoStmt*, 4> IndirectJumps;
  llvm::SmallVector<LabelStmt*, 4> IndirectJumpTargets;
public:
  JumpScopeChecker(Stmt *Body, Sema &S);
private:
  void BuildScopeInformation(Stmt *S, unsigned ParentScope);
  void VerifyJumps();
  void VerifyIndirectJumps();
  void DiagnoseIndirectJump(IndirectGotoStmt *IG, unsigned IGScope,
                            LabelStmt *Target, unsigned TargetScope);
  void CheckJump(Stmt *From, Stmt *To,
                 SourceLocation DiagLoc, unsigned JumpDiag);
};
} // end anonymous namespace


JumpScopeChecker::JumpScopeChecker(Stmt *Body, Sema &s) : S(s) {
  // Add a scope entry for function scope.
  Scopes.push_back(GotoScope(~0U, ~0U, ~0U, SourceLocation()));

  // Build information for the top level compound statement, so that we have a
  // defined scope record for every "goto" and label.
  BuildScopeInformation(Body, 0);

  // Check that all jumps we saw are kosher.
  VerifyJumps();
  VerifyIndirectJumps();
}

/// GetDiagForGotoScopeDecl - If this decl induces a new goto scope, return a
/// diagnostic that should be emitted if control goes over it. If not, return 0.
static std::pair<unsigned,unsigned>
    GetDiagForGotoScopeDecl(const Decl *D, bool isCPlusPlus) {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    unsigned InDiag = 0, OutDiag = 0;
    if (VD->getType()->isVariablyModifiedType())
      InDiag = diag::note_protected_by_vla;

    if (VD->hasAttr<BlocksAttr>()) {
      InDiag = diag::note_protected_by___block;
      OutDiag = diag::note_exits___block;
    } else if (VD->hasAttr<CleanupAttr>()) {
      InDiag = diag::note_protected_by_cleanup;
      OutDiag = diag::note_exits_cleanup;
    } else if (isCPlusPlus) {
      // FIXME: In C++0x, we have to check more conditions than "did we
      // just give it an initializer?". See 6.7p3.
      if (VD->hasLocalStorage() && VD->hasInit())
        InDiag = diag::note_protected_by_variable_init;

      CanQualType T = VD->getType()->getCanonicalTypeUnqualified();
      while (CanQual<ArrayType> AT = T->getAs<ArrayType>())
        T = AT->getElementType();
      if (CanQual<RecordType> RT = T->getAs<RecordType>())
        if (!cast<CXXRecordDecl>(RT->getDecl())->hasTrivialDestructor())
          OutDiag = diag::note_exits_dtor;
    }
    
    return std::make_pair(InDiag, OutDiag);    
  }

  if (const TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (TD->getUnderlyingType()->isVariablyModifiedType())
      return std::make_pair((unsigned) diag::note_protected_by_vla_typedef, 0);
  }

  return std::make_pair(0U, 0U);
}


/// BuildScopeInformation - The statements from CI to CE are known to form a
/// coherent VLA scope with a specified parent node.  Walk through the
/// statements, adding any labels or gotos to LabelAndGotoScopes and recursively
/// walking the AST as needed.
void JumpScopeChecker::BuildScopeInformation(Stmt *S, unsigned ParentScope) {

  // If we found a label, remember that it is in ParentScope scope.
  switch (S->getStmtClass()) {
  case Stmt::LabelStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
    LabelAndGotoScopes[S] = ParentScope;
    break;

  case Stmt::AddrLabelExprClass:
    IndirectJumpTargets.push_back(cast<AddrLabelExpr>(S)->getLabel());
    break;

  case Stmt::IndirectGotoStmtClass:
    LabelAndGotoScopes[S] = ParentScope;
    IndirectJumps.push_back(cast<IndirectGotoStmt>(S));
    break;

  case Stmt::GotoStmtClass:
  case Stmt::SwitchStmtClass:
    // Remember both what scope a goto is in as well as the fact that we have
    // it.  This makes the second scan not have to walk the AST again.
    LabelAndGotoScopes[S] = ParentScope;
    Jumps.push_back(S);
    break;

  default:
    break;
  }

  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end(); CI != E;
       ++CI) {
    Stmt *SubStmt = *CI;
    if (SubStmt == 0) continue;

    bool isCPlusPlus = this->S.getLangOptions().CPlusPlus;

    // If this is a declstmt with a VLA definition, it defines a scope from here
    // to the end of the containing context.
    if (DeclStmt *DS = dyn_cast<DeclStmt>(SubStmt)) {
      // The decl statement creates a scope if any of the decls in it are VLAs
      // or have the cleanup attribute.
      for (DeclStmt::decl_iterator I = DS->decl_begin(), E = DS->decl_end();
           I != E; ++I) {
        // If this decl causes a new scope, push and switch to it.
        std::pair<unsigned,unsigned> Diags
          = GetDiagForGotoScopeDecl(*I, isCPlusPlus);
        if (Diags.first || Diags.second) {
          Scopes.push_back(GotoScope(ParentScope, Diags.first, Diags.second,
                                     (*I)->getLocation()));
          ParentScope = Scopes.size()-1;
        }

        // If the decl has an initializer, walk it with the potentially new
        // scope we just installed.
        if (VarDecl *VD = dyn_cast<VarDecl>(*I))
          if (Expr *Init = VD->getInit())
            BuildScopeInformation(Init, ParentScope);
      }
      continue;
    }

    // Disallow jumps into any part of an @try statement by pushing a scope and
    // walking all sub-stmts in that scope.
    if (ObjCAtTryStmt *AT = dyn_cast<ObjCAtTryStmt>(SubStmt)) {
      // Recursively walk the AST for the @try part.
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_objc_try,
                                 diag::note_exits_objc_try,
                                 AT->getAtTryLoc()));
      if (Stmt *TryPart = AT->getTryBody())
        BuildScopeInformation(TryPart, Scopes.size()-1);

      // Jump from the catch to the finally or try is not valid.
      for (unsigned I = 0, N = AT->getNumCatchStmts(); I != N; ++I) {
        ObjCAtCatchStmt *AC = AT->getCatchStmt(I);
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_objc_catch,
                                   diag::note_exits_objc_catch,
                                   AC->getAtCatchLoc()));
        // @catches are nested and it isn't
        BuildScopeInformation(AC->getCatchBody(), Scopes.size()-1);
      }

      // Jump from the finally to the try or catch is not valid.
      if (ObjCAtFinallyStmt *AF = AT->getFinallyStmt()) {
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_objc_finally,
                                   diag::note_exits_objc_finally,
                                   AF->getAtFinallyLoc()));
        BuildScopeInformation(AF, Scopes.size()-1);
      }

      continue;
    }

    // Disallow jumps into the protected statement of an @synchronized, but
    // allow jumps into the object expression it protects.
    if (ObjCAtSynchronizedStmt *AS = dyn_cast<ObjCAtSynchronizedStmt>(SubStmt)){
      // Recursively walk the AST for the @synchronized object expr, it is
      // evaluated in the normal scope.
      BuildScopeInformation(AS->getSynchExpr(), ParentScope);

      // Recursively walk the AST for the @synchronized part, protected by a new
      // scope.
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_objc_synchronized,
                                 diag::note_exits_objc_synchronized,
                                 AS->getAtSynchronizedLoc()));
      BuildScopeInformation(AS->getSynchBody(), Scopes.size()-1);
      continue;
    }

    // Disallow jumps into any part of a C++ try statement. This is pretty
    // much the same as for Obj-C.
    if (CXXTryStmt *TS = dyn_cast<CXXTryStmt>(SubStmt)) {
      Scopes.push_back(GotoScope(ParentScope,
                                 diag::note_protected_by_cxx_try,
                                 diag::note_exits_cxx_try,
                                 TS->getSourceRange().getBegin()));
      if (Stmt *TryBlock = TS->getTryBlock())
        BuildScopeInformation(TryBlock, Scopes.size()-1);

      // Jump from the catch into the try is not allowed either.
      for (unsigned I = 0, E = TS->getNumHandlers(); I != E; ++I) {
        CXXCatchStmt *CS = TS->getHandler(I);
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_cxx_catch,
                                   diag::note_exits_cxx_catch,
                                   CS->getSourceRange().getBegin()));
        BuildScopeInformation(CS->getHandlerBlock(), Scopes.size()-1);
      }

      continue;
    }

    // Recursively walk the AST.
    BuildScopeInformation(SubStmt, ParentScope);
  }
}

/// VerifyJumps - Verify each element of the Jumps array to see if they are
/// valid, emitting diagnostics if not.
void JumpScopeChecker::VerifyJumps() {
  while (!Jumps.empty()) {
    Stmt *Jump = Jumps.pop_back_val();

    // With a goto,
    if (GotoStmt *GS = dyn_cast<GotoStmt>(Jump)) {
      CheckJump(GS, GS->getLabel(), GS->getGotoLoc(),
                diag::err_goto_into_protected_scope);
      continue;
    }

    SwitchStmt *SS = cast<SwitchStmt>(Jump);
    for (SwitchCase *SC = SS->getSwitchCaseList(); SC;
         SC = SC->getNextSwitchCase()) {
      assert(LabelAndGotoScopes.count(SC) && "Case not visited?");
      CheckJump(SS, SC, SC->getLocStart(),
                diag::err_switch_into_protected_scope);
    }
  }
}

/// VerifyIndirectJumps - Verify whether any possible indirect jump might
/// cross a protection boundary.
///
/// An indirect jump is "trivial" if it bypasses no initializations
/// and no teardowns.  More formally, the jump from A to B is trivial
/// if the path out from A to DCA(A,B) is trivial and the path in from
/// DCA(A,B) to B is trivial, where DCA(A,B) is the deepest common
/// ancestor of A and B.
/// A path in is trivial if none of the entered scopes have an InDiag.
/// A path out is trivial is none of the exited scopes have an OutDiag.
/// Jump-triviality is transitive but asymmetric.
void JumpScopeChecker::VerifyIndirectJumps() {
  if (IndirectJumps.empty()) return;

  // If there aren't any address-of-label expressions in this function,
  // complain about the first indirect goto.
  if (IndirectJumpTargets.empty()) {
    S.Diag(IndirectJumps[0]->getGotoLoc(),
           diag::err_indirect_goto_without_addrlabel);
    return;
  }

  // Build a vector of source scopes.  This serves to unique source
  // scopes as well as to eliminate redundant lookups into
  // LabelAndGotoScopes.
  typedef std::pair<unsigned, IndirectGotoStmt*> JumpScope;
  llvm::SmallVector<JumpScope, 32> JumpScopes;
  {
    llvm::DenseMap<unsigned, IndirectGotoStmt*> JumpScopesMap;
    for (llvm::SmallVectorImpl<IndirectGotoStmt*>::iterator
           I = IndirectJumps.begin(), E = IndirectJumps.end(); I != E; ++I) {
      IndirectGotoStmt *IG = *I;
      assert(LabelAndGotoScopes.count(IG) &&
             "indirect jump didn't get added to scopes?");
      unsigned IGScope = LabelAndGotoScopes[IG];
      IndirectGotoStmt *&Entry = JumpScopesMap[IGScope];
      if (!Entry) Entry = IG;
    }
    JumpScopes.reserve(JumpScopesMap.size());
    for (llvm::DenseMap<unsigned, IndirectGotoStmt*>::iterator
           I = JumpScopesMap.begin(), E = JumpScopesMap.end(); I != E; ++I)
      JumpScopes.push_back(*I);
  }

  // Find a representative label from each protection scope.
  llvm::DenseMap<unsigned, LabelStmt*> TargetScopes;
  for (llvm::SmallVectorImpl<LabelStmt*>::iterator
         I = IndirectJumpTargets.begin(), E = IndirectJumpTargets.end();
       I != E; ++I) {
    LabelStmt *TheLabel = *I;
    assert(LabelAndGotoScopes.count(TheLabel) &&
           "Referenced label didn't get added to scopes?");
    unsigned LabelScope = LabelAndGotoScopes[TheLabel];
    LabelStmt *&Target = TargetScopes[LabelScope];
    if (!Target) Target = TheLabel;
  }

  llvm::BitVector Reachable(Scopes.size(), false);
  for (llvm::DenseMap<unsigned,LabelStmt*>::iterator
         TI = TargetScopes.begin(), TE = TargetScopes.end(); TI != TE; ++TI) {
    unsigned TargetScope = TI->first;
    LabelStmt *TargetLabel = TI->second;

    Reachable.reset();

    // Mark all the enclosing scopes from which you can safely jump
    // into the target scope.
    unsigned Min = TargetScope;
    while (true) {
      Reachable.set(Min);

      // Don't go beyond the outermost scope.
      if (Min == 0) break;

      // Don't go further if we couldn't trivially enter this scope.
      if (Scopes[Min].InDiag) break;

      Min = Scopes[Min].ParentScope;
    }

    // Walk through all the jump sites, checking that they can trivially
    // reach this label scope.
    for (llvm::SmallVectorImpl<JumpScope>::iterator
           I = JumpScopes.begin(), E = JumpScopes.end(); I != E; ++I) {
      unsigned Scope = I->first;

      // Walk out the "scope chain" for this scope, looking for a scope
      // we've marked reachable.
      bool IsReachable = false;
      while (true) {
        if (Reachable.test(Scope)) {
          // If we find something reachable, mark all the scopes we just
          // walked through as reachable.
          for (unsigned S = I->first; S != Scope; S = Scopes[S].ParentScope)
            Reachable.set(S);
          IsReachable = true;
          break;
        }

        // Don't walk out if we've reached the top-level scope or we've
        // gotten shallower than the shallowest reachable scope.
        if (Scope == 0 || Scope < Min) break;

        // Don't walk out through an out-diagnostic.
        if (Scopes[Scope].OutDiag) break;

        Scope = Scopes[Scope].ParentScope;
      }

      // Only diagnose if we didn't find something.
      if (IsReachable) continue;

      DiagnoseIndirectJump(I->second, I->first, TargetLabel, TargetScope);
    }
  }
}

void JumpScopeChecker::DiagnoseIndirectJump(IndirectGotoStmt *Jump,
                                            unsigned JumpScope,
                                            LabelStmt *Target,
                                            unsigned TargetScope) {
  assert(JumpScope != TargetScope);

  S.Diag(Jump->getGotoLoc(), diag::warn_indirect_goto_in_protected_scope);
  S.Diag(Target->getIdentLoc(), diag::note_indirect_goto_target);

  // Collect everything in the target scope chain.
  llvm::DenseSet<unsigned> TargetScopeChain;
  for (unsigned SI = TargetScope; SI != 0; SI = Scopes[SI].ParentScope)
    TargetScopeChain.insert(SI);
  TargetScopeChain.insert(0);

  // Walk out the scopes containing the indirect goto until we find a
  // common ancestor with the target label.
  unsigned Common = JumpScope;
  while (!TargetScopeChain.count(Common)) {
    // FIXME: this isn't necessarily a problem!  Not every protected
    // scope requires destruction.
    S.Diag(Scopes[Common].Loc, Scopes[Common].OutDiag);
    Common = Scopes[Common].ParentScope;
  }

  // Now walk into the scopes containing the label whose address was taken.
  for (unsigned SI = TargetScope; SI != Common; SI = Scopes[SI].ParentScope)
    S.Diag(Scopes[SI].Loc, Scopes[SI].InDiag);
}

/// CheckJump - Validate that the specified jump statement is valid: that it is
/// jumping within or out of its current scope, not into a deeper one.
void JumpScopeChecker::CheckJump(Stmt *From, Stmt *To,
                                 SourceLocation DiagLoc, unsigned JumpDiag) {
  assert(LabelAndGotoScopes.count(From) && "Jump didn't get added to scopes?");
  unsigned FromScope = LabelAndGotoScopes[From];

  assert(LabelAndGotoScopes.count(To) && "Jump didn't get added to scopes?");
  unsigned ToScope = LabelAndGotoScopes[To];

  // Common case: exactly the same scope, which is fine.
  if (FromScope == ToScope) return;

  // The only valid mismatch jump case happens when the jump is more deeply
  // nested inside the jump target.  Do a quick scan to see if the jump is valid
  // because valid code is more common than invalid code.
  unsigned TestScope = Scopes[FromScope].ParentScope;
  while (TestScope != ~0U) {
    // If we found the jump target, then we're jumping out of our current scope,
    // which is perfectly fine.
    if (TestScope == ToScope) return;

    // Otherwise, scan up the hierarchy.
    TestScope = Scopes[TestScope].ParentScope;
  }

  // If we get here, then either we have invalid code or we're jumping in
  // past some cleanup blocks.  It may seem strange to have a declaration
  // with a trivial constructor and a non-trivial destructor, but it's
  // possible.

  // Eliminate the common prefix of the jump and the target.  Start by
  // linearizing both scopes, reversing them as we go.
  std::vector<unsigned> FromScopes, ToScopes;
  for (TestScope = FromScope; TestScope != ~0U;
       TestScope = Scopes[TestScope].ParentScope)
    FromScopes.push_back(TestScope);
  for (TestScope = ToScope; TestScope != ~0U;
       TestScope = Scopes[TestScope].ParentScope)
    ToScopes.push_back(TestScope);

  // Remove any common entries (such as the top-level function scope).
  while (!FromScopes.empty() && (FromScopes.back() == ToScopes.back())) {
    FromScopes.pop_back();
    ToScopes.pop_back();
  }

  // Ignore any cleanup blocks on the way in.
  while (!ToScopes.empty()) {
    if (Scopes[ToScopes.back()].InDiag) break;
    ToScopes.pop_back();
  }
  if (ToScopes.empty()) return;

  S.Diag(DiagLoc, JumpDiag);

  // Emit diagnostics for whatever is left in ToScopes.
  for (unsigned i = 0, e = ToScopes.size(); i != e; ++i)
    S.Diag(Scopes[ToScopes[i]].Loc, Scopes[ToScopes[i]].InDiag);
}

void Sema::DiagnoseInvalidJumps(Stmt *Body) {
  (void)JumpScopeChecker(Body, *this);
}
