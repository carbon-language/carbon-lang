//===--- Scope.h - Scope interface ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Scope interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_SCOPE_H
#define LLVM_CLANG_PARSE_SCOPE_H

#include "clang/Parse/Action.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {

/// Scope - A scope is a transient data structure that is used while parsing the
/// program.  It assists with resolving identifiers to the appropriate
/// declaration.
///
class Scope {
public:
  /// ScopeFlags - These are bitfields that are or'd together when creating a
  /// scope, which defines the sorts of things the scope contains.
  enum ScopeFlags {
    /// FnScope - This indicates that the scope corresponds to a function, which
    /// means that labels are set here.
    FnScope       = 0x01,

    /// BreakScope - This is a while,do,switch,for, etc that can have break
    /// stmts embedded into it.
    BreakScope    = 0x02,

    /// ContinueScope - This is a while,do,for, which can have continue
    /// stmt embedded into it.
    ContinueScope = 0x04,

    /// DeclScope - This is a scope that can contain a declaration.  Some scopes
    /// just contain loop constructs but don't contain decls.
    DeclScope = 0x08,

    /// ControlScope - The controlling scope in a if/switch/while/for statement.
    ControlScope = 0x10,

    /// ClassScope - The scope of a struct/union/class definition.
    ClassScope = 0x20,

    /// BlockScope - This is a scope that corresponds to a block object.
    /// Blocks serve as top-level scopes for some objects like labels, they
    /// also prevent things like break and continue.  BlockScopes have the
    /// other flags set as well.
    BlockScope = 0x40,

    /// TemplateParamScope - This is a scope that corresponds to the
    /// template parameters of a C++ template. Template parameter
    /// scope starts at the 'template' keyword and ends when the
    /// template declaration ends.
    TemplateParamScope = 0x80,

    /// FunctionPrototypeScope - This is a scope that corresponds to the
    /// parameters within a function prototype.
    FunctionPrototypeScope = 0x100,

    /// AtCatchScope - This is a scope that corresponds to the Objective-C
    /// @catch statement.
    AtCatchScope = 0x200
  };
private:
  /// The parent scope for this scope.  This is null for the translation-unit
  /// scope.
  Scope *AnyParent;

  /// Depth - This is the depth of this scope.  The translation-unit scope has
  /// depth 0.
  unsigned Depth : 16;

  /// Flags - This contains a set of ScopeFlags, which indicates how the scope
  /// interrelates with other control flow statements.
  unsigned Flags : 10;

  /// WithinElse - Whether this scope is part of the "else" branch in
  /// its parent ControlScope.
  bool WithinElse : 1;

  /// FnParent - If this scope has a parent scope that is a function body, this
  /// pointer is non-null and points to it.  This is used for label processing.
  Scope *FnParent;

  /// BreakParent/ContinueParent - This is a direct link to the immediately
  /// preceeding BreakParent/ContinueParent if this scope is not one, or null if
  /// there is no containing break/continue scope.
  Scope *BreakParent, *ContinueParent;

  /// ControlParent - This is a direct link to the immediately
  /// preceeding ControlParent if this scope is not one, or null if
  /// there is no containing control scope.
  Scope *ControlParent;

  /// BlockParent - This is a direct link to the immediately containing
  /// BlockScope if this scope is not one, or null if there is none.
  Scope *BlockParent;

  /// TemplateParamParent - This is a direct link to the
  /// immediately containing template parameter scope. In the
  /// case of nested templates, template parameter scopes can have
  /// other template parameter scopes as parents.
  Scope *TemplateParamParent;

  /// DeclsInScope - This keeps track of all declarations in this scope.  When
  /// the declaration is added to the scope, it is set as the current
  /// declaration for the identifier in the IdentifierTable.  When the scope is
  /// popped, these declarations are removed from the IdentifierTable's notion
  /// of current declaration.  It is up to the current Action implementation to
  /// implement these semantics.
  typedef llvm::SmallPtrSet<Action::DeclPtrTy, 32> DeclSetTy;
  DeclSetTy DeclsInScope;

  /// Entity - The entity with which this scope is associated. For
  /// example, the entity of a class scope is the class itself, the
  /// entity of a function scope is a function, etc. This field is
  /// maintained by the Action implementation.
  void *Entity;

  typedef llvm::SmallVector<Action::DeclPtrTy, 2> UsingDirectivesTy;
  UsingDirectivesTy UsingDirectives;

public:
  Scope(Scope *Parent, unsigned ScopeFlags) {
    Init(Parent, ScopeFlags);
  }

  /// getFlags - Return the flags for this scope.
  ///
  unsigned getFlags() const { return Flags; }

  /// isBlockScope - Return true if this scope does not correspond to a
  /// closure.
  bool isBlockScope() const { return Flags & BlockScope; }

  /// getParent - Return the scope that this is nested in.
  ///
  const Scope *getParent() const { return AnyParent; }
  Scope *getParent() { return AnyParent; }

  /// getFnParent - Return the closest scope that is a function body.
  ///
  const Scope *getFnParent() const { return FnParent; }
  Scope *getFnParent() { return FnParent; }

  /// getContinueParent - Return the closest scope that a continue statement
  /// would be affected by.  If the closest scope is a closure scope, we know
  /// that there is no loop *inside* the closure.
  Scope *getContinueParent() {
    if (ContinueParent && !ContinueParent->isBlockScope())
      return ContinueParent;
    return 0;
  }

  const Scope *getContinueParent() const {
    return const_cast<Scope*>(this)->getContinueParent();
  }

  /// getBreakParent - Return the closest scope that a break statement
  /// would be affected by.  If the closest scope is a block scope, we know
  /// that there is no loop *inside* the block.
  Scope *getBreakParent() {
    if (BreakParent && !BreakParent->isBlockScope())
      return BreakParent;
    return 0;
  }
  const Scope *getBreakParent() const {
    return const_cast<Scope*>(this)->getBreakParent();
  }

  Scope *getControlParent() { return ControlParent; }
  const Scope *getControlParent() const { return ControlParent; }

  Scope *getBlockParent() { return BlockParent; }
  const Scope *getBlockParent() const { return BlockParent; }

  Scope *getTemplateParamParent() { return TemplateParamParent; }
  const Scope *getTemplateParamParent() const { return TemplateParamParent; }

  typedef DeclSetTy::iterator decl_iterator;
  decl_iterator decl_begin() const { return DeclsInScope.begin(); }
  decl_iterator decl_end()   const { return DeclsInScope.end(); }
  bool decl_empty()          const { return DeclsInScope.empty(); }

  void AddDecl(Action::DeclPtrTy D) {
    DeclsInScope.insert(D);
  }

  void RemoveDecl(Action::DeclPtrTy D) {
    DeclsInScope.erase(D);
  }

  /// isDeclScope - Return true if this is the scope that the specified decl is
  /// declared in.
  bool isDeclScope(Action::DeclPtrTy D) {
    return DeclsInScope.count(D) != 0;
  }

  void* getEntity() const { return Entity; }
  void setEntity(void *E) { Entity = E; }

  /// isClassScope - Return true if this scope is a class/struct/union scope.
  bool isClassScope() const {
    return (getFlags() & Scope::ClassScope);
  }

  /// isInCXXInlineMethodScope - Return true if this scope is a C++ inline
  /// method scope or is inside one.
  bool isInCXXInlineMethodScope() const {
    if (const Scope *FnS = getFnParent()) {
      assert(FnS->getParent() && "TUScope not created?");
      return FnS->getParent()->isClassScope();
    }
    return false;
  }

  /// isTemplateParamScope - Return true if this scope is a C++
  /// template parameter scope.
  bool isTemplateParamScope() const {
    return getFlags() & Scope::TemplateParamScope;
  }

  /// isFunctionPrototypeScope - Return true if this scope is a
  /// function prototype scope.
  bool isFunctionPrototypeScope() const {
    return getFlags() & Scope::FunctionPrototypeScope;
  }

  /// isAtCatchScope - Return true if this scope is @catch.
  bool isAtCatchScope() const {
    return getFlags() & Scope::AtCatchScope;
  }

  /// isWithinElse - Whether we are within the "else" of the
  /// ControlParent (if any).
  bool isWithinElse() const { return WithinElse; }

  void setWithinElse(bool WE) { WithinElse = WE; }

  typedef UsingDirectivesTy::iterator udir_iterator;
  typedef UsingDirectivesTy::const_iterator const_udir_iterator;

  void PushUsingDirective(Action::DeclPtrTy UDir) {
    UsingDirectives.push_back(UDir);
  }

  udir_iterator using_directives_begin() {
    return UsingDirectives.begin();
  }

  udir_iterator using_directives_end() {
    return UsingDirectives.end();
  }

  const_udir_iterator using_directives_begin() const {
    return UsingDirectives.begin();
  }

  const_udir_iterator using_directives_end() const {
    return UsingDirectives.end();
  }

  /// Init - This is used by the parser to implement scope caching.
  ///
  void Init(Scope *Parent, unsigned ScopeFlags) {
    AnyParent = Parent;
    Depth = AnyParent ? AnyParent->Depth+1 : 0;
    Flags = ScopeFlags;

    if (AnyParent) {
      FnParent       = AnyParent->FnParent;
      BreakParent    = AnyParent->BreakParent;
      ContinueParent = AnyParent->ContinueParent;
      ControlParent = AnyParent->ControlParent;
      BlockParent  = AnyParent->BlockParent;
      TemplateParamParent = AnyParent->TemplateParamParent;
      WithinElse = AnyParent->WithinElse;

    } else {
      FnParent = BreakParent = ContinueParent = BlockParent = 0;
      ControlParent = 0;
      TemplateParamParent = 0;
      WithinElse = false;
    }

    // If this scope is a function or contains breaks/continues, remember it.
    if (Flags & FnScope)            FnParent = this;
    if (Flags & BreakScope)         BreakParent = this;
    if (Flags & ContinueScope)      ContinueParent = this;
    if (Flags & ControlScope)       ControlParent = this;
    if (Flags & BlockScope)         BlockParent = this;
    if (Flags & TemplateParamScope) TemplateParamParent = this;
    DeclsInScope.clear();
    UsingDirectives.clear();
    Entity = 0;
  }
};

}  // end namespace clang

#endif
