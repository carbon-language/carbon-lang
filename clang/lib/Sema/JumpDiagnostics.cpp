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
    
    /// Diag - The diagnostic to emit if there is a jump into this scope.
    unsigned Diag;
    
    /// Loc - Location to emit the diagnostic.
    SourceLocation Loc;
    
    GotoScope(unsigned parentScope, unsigned diag, SourceLocation L)
    : ParentScope(parentScope), Diag(diag), Loc(L) {}
  };
 
  llvm::SmallVector<GotoScope, 48> Scopes;
  llvm::DenseMap<Stmt*, unsigned> LabelAndGotoScopes;
  llvm::SmallVector<Stmt*, 16> Jumps;
public:
  JumpScopeChecker(Stmt *Body, Sema &S);
private:
  void BuildScopeInformation(Stmt *S, unsigned ParentScope);
  void VerifyJumps();
  void CheckJump(Stmt *From, Stmt *To,
                 SourceLocation DiagLoc, unsigned JumpDiag);
};
} // end anonymous namespace


JumpScopeChecker::JumpScopeChecker(Stmt *Body, Sema &s) : S(s) {
  // Add a scope entry for function scope.
  Scopes.push_back(GotoScope(~0U, ~0U, SourceLocation()));
  
  // Build information for the top level compound statement, so that we have a
  // defined scope record for every "goto" and label.
  BuildScopeInformation(Body, 0);
  
  // Check that all jumps we saw are kosher.
  VerifyJumps();
}
  
/// GetDiagForGotoScopeDecl - If this decl induces a new goto scope, return a
/// diagnostic that should be emitted if control goes over it. If not, return 0.
static unsigned GetDiagForGotoScopeDecl(const Decl *D) {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->getType()->isVariablyModifiedType())
      return diag::note_protected_by_vla;
    if (VD->hasAttr<CleanupAttr>())
      return diag::note_protected_by_cleanup;
  } else if (const TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    if (TD->getUnderlyingType()->isVariablyModifiedType())
      return diag::note_protected_by_vla_typedef;
  }
  
  return 0;
}


/// BuildScopeInformation - The statements from CI to CE are known to form a
/// coherent VLA scope with a specified parent node.  Walk through the
/// statements, adding any labels or gotos to LabelAndGotoScopes and recursively
/// walking the AST as needed.
void JumpScopeChecker::BuildScopeInformation(Stmt *S, unsigned ParentScope) {
  
  // If we found a label, remember that it is in ParentScope scope.
  if (isa<LabelStmt>(S) || isa<DefaultStmt>(S) || isa<CaseStmt>(S)) {
    LabelAndGotoScopes[S] = ParentScope;
  } else if (isa<GotoStmt>(S) || isa<SwitchStmt>(S) ||
             isa<IndirectGotoStmt>(S) || isa<AddrLabelExpr>(S)) {
    // Remember both what scope a goto is in as well as the fact that we have
    // it.  This makes the second scan not have to walk the AST again.
    LabelAndGotoScopes[S] = ParentScope;
    Jumps.push_back(S);
  }
  
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end(); CI != E;
       ++CI) {
    Stmt *SubStmt = *CI;
    if (SubStmt == 0) continue;
    
    // FIXME: diagnose jumps past initialization: required in C++, warning in C.
    //   goto L; int X = 4;   L: ;

    // If this is a declstmt with a VLA definition, it defines a scope from here
    // to the end of the containing context.
    if (DeclStmt *DS = dyn_cast<DeclStmt>(SubStmt)) {
      // The decl statement creates a scope if any of the decls in it are VLAs or
      // have the cleanup attribute.
      for (DeclStmt::decl_iterator I = DS->decl_begin(), E = DS->decl_end();
           I != E; ++I) {
        // If this decl causes a new scope, push and switch to it.
        if (unsigned Diag = GetDiagForGotoScopeDecl(*I)) {
          Scopes.push_back(GotoScope(ParentScope, Diag, (*I)->getLocation()));
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
      Scopes.push_back(GotoScope(ParentScope,diag::note_protected_by_objc_try,
                                 AT->getAtTryLoc()));
      if (Stmt *TryPart = AT->getTryBody())
        BuildScopeInformation(TryPart, Scopes.size()-1);

      // Jump from the catch to the finally or try is not valid.
      for (ObjCAtCatchStmt *AC = AT->getCatchStmts(); AC;
           AC = AC->getNextCatchStmt()) {
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_objc_catch,
                                   AC->getAtCatchLoc()));
        // @catches are nested and it isn't 
        BuildScopeInformation(AC->getCatchBody(), Scopes.size()-1);
      }
      
      // Jump from the finally to the try or catch is not valid.
      if (ObjCAtFinallyStmt *AF = AT->getFinallyStmt()) {
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_objc_finally,
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
                                 AS->getAtSynchronizedLoc()));
      BuildScopeInformation(AS->getSynchBody(), Scopes.size()-1);
      continue;
    }

    // Disallow jumps into any part of a C++ try statement. This is pretty
    // much the same as for Obj-C.
    if (CXXTryStmt *TS = dyn_cast<CXXTryStmt>(SubStmt)) {
      Scopes.push_back(GotoScope(ParentScope, diag::note_protected_by_cxx_try,
                                 TS->getSourceRange().getBegin()));
      if (Stmt *TryBlock = TS->getTryBlock())
        BuildScopeInformation(TryBlock, Scopes.size()-1);

      // Jump from the catch into the try is not allowed either.
      for(unsigned I = 0, E = TS->getNumHandlers(); I != E; ++I) {
        CXXCatchStmt *CS = TS->getHandler(I);
        Scopes.push_back(GotoScope(ParentScope,
                                   diag::note_protected_by_cxx_catch,
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
    
    if (SwitchStmt *SS = dyn_cast<SwitchStmt>(Jump)) {
      for (SwitchCase *SC = SS->getSwitchCaseList(); SC;
           SC = SC->getNextSwitchCase()) {
        assert(LabelAndGotoScopes.count(SC) && "Case not visited?");
        CheckJump(SS, SC, SC->getLocStart(),
                  diag::err_switch_into_protected_scope);
      }
      continue;
    }

    unsigned DiagnosticScope;
    
    // We don't know where an indirect goto goes, require that it be at the
    // top level of scoping.
    if (IndirectGotoStmt *IG = dyn_cast<IndirectGotoStmt>(Jump)) {
      assert(LabelAndGotoScopes.count(Jump) &&
             "Jump didn't get added to scopes?");
      unsigned GotoScope = LabelAndGotoScopes[IG];
      if (GotoScope == 0) continue;  // indirect jump is ok.
      S.Diag(IG->getGotoLoc(), diag::err_indirect_goto_in_protected_scope);
      DiagnosticScope = GotoScope;
    } else {
      // We model &&Label as a jump for purposes of scope tracking.  We actually
      // don't care *where* the address of label is, but we require the *label
      // itself* to be in scope 0.  If it is nested inside of a VLA scope, then
      // it is possible for an indirect goto to illegally enter the VLA scope by
      // indirectly jumping to the label.
      assert(isa<AddrLabelExpr>(Jump) && "Unknown jump type");
      LabelStmt *TheLabel = cast<AddrLabelExpr>(Jump)->getLabel();
    
      assert(LabelAndGotoScopes.count(TheLabel) &&
             "Referenced label didn't get added to scopes?");
      unsigned LabelScope = LabelAndGotoScopes[TheLabel];
      if (LabelScope == 0) continue; // Addr of label is ok.
    
      S.Diag(Jump->getLocStart(), diag::err_addr_of_label_in_protected_scope);
      DiagnosticScope = LabelScope;
    }

    // Report all the things that would be skipped over by this &&label or
    // indirect goto.
    while (DiagnosticScope != 0) {
      S.Diag(Scopes[DiagnosticScope].Loc, Scopes[DiagnosticScope].Diag);
      DiagnosticScope = Scopes[DiagnosticScope].ParentScope;
    }
  }
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
  
  // If we get here, then we know we have invalid code.  Diagnose the bad jump,
  // and then emit a note at each VLA being jumped out of.
  S.Diag(DiagLoc, JumpDiag);

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
  while (!FromScopes.empty() && FromScopes.back() == ToScopes.back()) {
    FromScopes.pop_back();
    ToScopes.pop_back();
  }
  
  // Emit diagnostics for whatever is left in ToScopes.
  for (unsigned i = 0, e = ToScopes.size(); i != e; ++i)
    S.Diag(Scopes[ToScopes[i]].Loc, Scopes[ToScopes[i]].Diag);
}

void Sema::DiagnoseInvalidJumps(Stmt *Body) {
  JumpScopeChecker(Body, *this);
}
