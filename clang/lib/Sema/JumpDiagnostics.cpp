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

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/BitVector.h"
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
  llvm::SmallVector<LabelDecl*, 4> IndirectJumpTargets;
public:
  JumpScopeChecker(Stmt *Body, Sema &S);
private:
  void BuildScopeInformation(Decl *D, unsigned &ParentScope);
  void BuildScopeInformation(Stmt *S, unsigned ParentScope);
  void VerifyJumps();
  void VerifyIndirectJumps();
  void DiagnoseIndirectJump(IndirectGotoStmt *IG, unsigned IGScope,
                            LabelDecl *Target, unsigned TargetScope);
  void CheckJump(Stmt *From, Stmt *To,
                 SourceLocation DiagLoc, unsigned JumpDiag);

  unsigned GetDeepestCommonScope(unsigned A, unsigned B);
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

/// GetDeepestCommonScope - Finds the innermost scope enclosing the
/// two scopes.
unsigned JumpScopeChecker::GetDeepestCommonScope(unsigned A, unsigned B) {
  while (A != B) {
    // Inner scopes are created after outer scopes and therefore have
    // higher indices.
    if (A < B) {
      assert(Scopes[B].ParentScope < B);
      B = Scopes[B].ParentScope;
    } else {
      assert(Scopes[A].ParentScope < A);
      A = Scopes[A].ParentScope;
    }
  }
  return A;
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
      if (!VD->hasLocalStorage())
        return std::make_pair(InDiag, OutDiag);
      
      ASTContext &Context = D->getASTContext();
      QualType T = Context.getBaseElementType(VD->getType());
      if (!T->isDependentType()) {
        // C++0x [stmt.dcl]p3:
        //   A program that jumps from a point where a variable with automatic
        //   storage duration is not in scope to a point where it is in scope
        //   is ill-formed unless the variable has scalar type, class type with
        //   a trivial default constructor and a trivial destructor, a 
        //   cv-qualified version of one of these types, or an array of one of
        //   the preceding types and is declared without an initializer (8.5).
        // Check whether this is a C++ class.
        CXXRecordDecl *Record = T->getAsCXXRecordDecl();
        
        if (const Expr *Init = VD->getInit()) {
          bool CallsTrivialConstructor = false;
          if (Record) {
            // FIXME: With generalized initializer lists, this may
            // classify "X x{};" as having no initializer.
            if (const CXXConstructExpr *Construct 
                                        = dyn_cast<CXXConstructExpr>(Init))
              if (const CXXConstructorDecl *Constructor
                                                = Construct->getConstructor())
                if ((Context.getLangOptions().CPlusPlus0x
                       ? Record->hasTrivialDefaultConstructor()
                       : Record->isPOD()) &&
                    Constructor->isDefaultConstructor())
                  CallsTrivialConstructor = true;
          }
          
          if (!CallsTrivialConstructor)
            InDiag = diag::note_protected_by_variable_init;
        }
        
        // Note whether we have a class with a non-trivial destructor.
        if (Record && !Record->hasTrivialDestructor())
          OutDiag = diag::note_exits_dtor;
      }
    }
    
    return std::make_pair(InDiag, OutDiag);    
  }

  if (const TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (TD->getUnderlyingType()->isVariablyModifiedType())
      return std::make_pair((unsigned) diag::note_protected_by_vla_typedef, 0);
  }

  if (const TypeAliasDecl *TD = dyn_cast<TypeAliasDecl>(D)) {
    if (TD->getUnderlyingType()->isVariablyModifiedType())
      return std::make_pair((unsigned) diag::note_protected_by_vla_type_alias, 0);
  }

  return std::make_pair(0U, 0U);
}

/// \brief Build scope information for a declaration that is part of a DeclStmt.
void JumpScopeChecker::BuildScopeInformation(Decl *D, unsigned &ParentScope) {
  bool isCPlusPlus = this->S.getLangOptions().CPlusPlus;
  
  // If this decl causes a new scope, push and switch to it.
  std::pair<unsigned,unsigned> Diags
    = GetDiagForGotoScopeDecl(D, isCPlusPlus);
  if (Diags.first || Diags.second) {
    Scopes.push_back(GotoScope(ParentScope, Diags.first, Diags.second,
                               D->getLocation()));
    ParentScope = Scopes.size()-1;
  }
  
  // If the decl has an initializer, walk it with the potentially new
  // scope we just installed.
  if (VarDecl *VD = dyn_cast<VarDecl>(D))
    if (Expr *Init = VD->getInit())
      BuildScopeInformation(Init, ParentScope);
}

/// BuildScopeInformation - The statements from CI to CE are known to form a
/// coherent VLA scope with a specified parent node.  Walk through the
/// statements, adding any labels or gotos to LabelAndGotoScopes and recursively
/// walking the AST as needed.
void JumpScopeChecker::BuildScopeInformation(Stmt *S, unsigned ParentScope) {
  bool SkipFirstSubStmt = false;
  
  // If we found a label, remember that it is in ParentScope scope.
  switch (S->getStmtClass()) {
  case Stmt::AddrLabelExprClass:
    IndirectJumpTargets.push_back(cast<AddrLabelExpr>(S)->getLabel());
    break;

  case Stmt::IndirectGotoStmtClass:
    // "goto *&&lbl;" is a special case which we treat as equivalent
    // to a normal goto.  In addition, we don't calculate scope in the
    // operand (to avoid recording the address-of-label use), which
    // works only because of the restricted set of expressions which
    // we detect as constant targets.
    if (cast<IndirectGotoStmt>(S)->getConstantTarget()) {
      LabelAndGotoScopes[S] = ParentScope;
      Jumps.push_back(S);
      return;
    }

    LabelAndGotoScopes[S] = ParentScope;
    IndirectJumps.push_back(cast<IndirectGotoStmt>(S));
    break;

  case Stmt::SwitchStmtClass:
    // Evaluate the condition variable before entering the scope of the switch
    // statement.
    if (VarDecl *Var = cast<SwitchStmt>(S)->getConditionVariable()) {
      BuildScopeInformation(Var, ParentScope);
      SkipFirstSubStmt = true;
    }
    // Fall through
      
  case Stmt::GotoStmtClass:
    // Remember both what scope a goto is in as well as the fact that we have
    // it.  This makes the second scan not have to walk the AST again.
    LabelAndGotoScopes[S] = ParentScope;
    Jumps.push_back(S);
    break;

  default:
    break;
  }

  for (Stmt::child_range CI = S->children(); CI; ++CI) {
    if (SkipFirstSubStmt) {
      SkipFirstSubStmt = false;
      continue;
    }
    
    Stmt *SubStmt = *CI;
    if (SubStmt == 0) continue;

    // Cases, labels, and defaults aren't "scope parents".  It's also
    // important to handle these iteratively instead of recursively in
    // order to avoid blowing out the stack.
    while (true) {
      Stmt *Next;
      if (CaseStmt *CS = dyn_cast<CaseStmt>(SubStmt))
        Next = CS->getSubStmt();
      else if (DefaultStmt *DS = dyn_cast<DefaultStmt>(SubStmt))
        Next = DS->getSubStmt();
      else if (LabelStmt *LS = dyn_cast<LabelStmt>(SubStmt))
        Next = LS->getSubStmt();
      else
        break;

      LabelAndGotoScopes[SubStmt] = ParentScope;
      SubStmt = Next;
    }

    // If this is a declstmt with a VLA definition, it defines a scope from here
    // to the end of the containing context.
    if (DeclStmt *DS = dyn_cast<DeclStmt>(SubStmt)) {
      // The decl statement creates a scope if any of the decls in it are VLAs
      // or have the cleanup attribute.
      for (DeclStmt::decl_iterator I = DS->decl_begin(), E = DS->decl_end();
           I != E; ++I)
        BuildScopeInformation(*I, ParentScope);
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
      CheckJump(GS, GS->getLabel()->getStmt(), GS->getGotoLoc(),
                diag::err_goto_into_protected_scope);
      continue;
    }

    // We only get indirect gotos here when they have a constant target.
    if (IndirectGotoStmt *IGS = dyn_cast<IndirectGotoStmt>(Jump)) {
      LabelDecl *Target = IGS->getConstantTarget();
      CheckJump(IGS, Target->getStmt(), IGS->getGotoLoc(),
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

/// VerifyIndirectJumps - Verify whether any possible indirect jump
/// might cross a protection boundary.  Unlike direct jumps, indirect
/// jumps count cleanups as protection boundaries:  since there's no
/// way to know where the jump is going, we can't implicitly run the
/// right cleanups the way we can with direct jumps.
///
/// Thus, an indirect jump is "trivial" if it bypasses no
/// initializations and no teardowns.  More formally, an indirect jump
/// from A to B is trivial if the path out from A to DCA(A,B) is
/// trivial and the path in from DCA(A,B) to B is trivial, where
/// DCA(A,B) is the deepest common ancestor of A and B.
/// Jump-triviality is transitive but asymmetric.
///
/// A path in is trivial if none of the entered scopes have an InDiag.
/// A path out is trivial is none of the exited scopes have an OutDiag.
///
/// Under these definitions, this function checks that the indirect
/// jump between A and B is trivial for every indirect goto statement A
/// and every label B whose address was taken in the function.
void JumpScopeChecker::VerifyIndirectJumps() {
  if (IndirectJumps.empty()) return;

  // If there aren't any address-of-label expressions in this function,
  // complain about the first indirect goto.
  if (IndirectJumpTargets.empty()) {
    S.Diag(IndirectJumps[0]->getGotoLoc(),
           diag::err_indirect_goto_without_addrlabel);
    return;
  }

  // Collect a single representative of every scope containing an
  // indirect goto.  For most code bases, this substantially cuts
  // down on the number of jump sites we'll have to consider later.
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

  // Collect a single representative of every scope containing a
  // label whose address was taken somewhere in the function.
  // For most code bases, there will be only one such scope.
  llvm::DenseMap<unsigned, LabelDecl*> TargetScopes;
  for (llvm::SmallVectorImpl<LabelDecl*>::iterator
         I = IndirectJumpTargets.begin(), E = IndirectJumpTargets.end();
       I != E; ++I) {
    LabelDecl *TheLabel = *I;
    assert(LabelAndGotoScopes.count(TheLabel->getStmt()) &&
           "Referenced label didn't get added to scopes?");
    unsigned LabelScope = LabelAndGotoScopes[TheLabel->getStmt()];
    LabelDecl *&Target = TargetScopes[LabelScope];
    if (!Target) Target = TheLabel;
  }

  // For each target scope, make sure it's trivially reachable from
  // every scope containing a jump site.
  //
  // A path between scopes always consists of exitting zero or more
  // scopes, then entering zero or more scopes.  We build a set of
  // of scopes S from which the target scope can be trivially
  // entered, then verify that every jump scope can be trivially
  // exitted to reach a scope in S.
  llvm::BitVector Reachable(Scopes.size(), false);
  for (llvm::DenseMap<unsigned,LabelDecl*>::iterator
         TI = TargetScopes.begin(), TE = TargetScopes.end(); TI != TE; ++TI) {
    unsigned TargetScope = TI->first;
    LabelDecl *TargetLabel = TI->second;

    Reachable.reset();

    // Mark all the enclosing scopes from which you can safely jump
    // into the target scope.  'Min' will end up being the index of
    // the shallowest such scope.
    unsigned Min = TargetScope;
    while (true) {
      Reachable.set(Min);

      // Don't go beyond the outermost scope.
      if (Min == 0) break;

      // Stop if we can't trivially enter the current scope.
      if (Scopes[Min].InDiag) break;

      Min = Scopes[Min].ParentScope;
    }

    // Walk through all the jump sites, checking that they can trivially
    // reach this label scope.
    for (llvm::SmallVectorImpl<JumpScope>::iterator
           I = JumpScopes.begin(), E = JumpScopes.end(); I != E; ++I) {
      unsigned Scope = I->first;

      // Walk out the "scope chain" for this scope, looking for a scope
      // we've marked reachable.  For well-formed code this amortizes
      // to O(JumpScopes.size() / Scopes.size()):  we only iterate
      // when we see something unmarked, and in well-formed code we
      // mark everything we iterate past.
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

/// Diagnose an indirect jump which is known to cross scopes.
void JumpScopeChecker::DiagnoseIndirectJump(IndirectGotoStmt *Jump,
                                            unsigned JumpScope,
                                            LabelDecl *Target,
                                            unsigned TargetScope) {
  assert(JumpScope != TargetScope);

  S.Diag(Jump->getGotoLoc(), diag::err_indirect_goto_in_protected_scope);
  S.Diag(Target->getStmt()->getIdentLoc(), diag::note_indirect_goto_target);

  unsigned Common = GetDeepestCommonScope(JumpScope, TargetScope);

  // Walk out the scope chain until we reach the common ancestor.
  for (unsigned I = JumpScope; I != Common; I = Scopes[I].ParentScope)
    if (Scopes[I].OutDiag)
      S.Diag(Scopes[I].Loc, Scopes[I].OutDiag);

  // Now walk into the scopes containing the label whose address was taken.
  for (unsigned I = TargetScope; I != Common; I = Scopes[I].ParentScope)
    if (Scopes[I].InDiag)
      S.Diag(Scopes[I].Loc, Scopes[I].InDiag);
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

  unsigned CommonScope = GetDeepestCommonScope(FromScope, ToScope);

  // It's okay to jump out from a nested scope.
  if (CommonScope == ToScope) return;

  // Pull out (and reverse) any scopes we might need to diagnose skipping.
  llvm::SmallVector<unsigned, 10> ToScopes;
  for (unsigned I = ToScope; I != CommonScope; I = Scopes[I].ParentScope)
    if (Scopes[I].InDiag)
      ToScopes.push_back(I);

  // If the only scopes present are cleanup scopes, we're okay.
  if (ToScopes.empty()) return;

  S.Diag(DiagLoc, JumpDiag);

  // Emit diagnostics for whatever is left in ToScopes.
  for (unsigned i = 0, e = ToScopes.size(); i != e; ++i)
    S.Diag(Scopes[ToScopes[i]].Loc, Scopes[ToScopes[i]].InDiag);
}

void Sema::DiagnoseInvalidJumps(Stmt *Body) {
  (void)JumpScopeChecker(Body, *this);
}
