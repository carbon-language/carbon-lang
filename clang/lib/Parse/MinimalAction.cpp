//===--- MinimalAction.cpp - Implement the MinimalAction class ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the MinimalAction interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

///  Out-of-line virtual destructor to provide home for ActionBase class.
ActionBase::~ActionBase() {}

///  Out-of-line virtual destructor to provide home for Action class.
Action::~Action() {}

Action::ObjCMessageKind Action::getObjCMessageKind(Scope *S,
                                                   IdentifierInfo *Name,
                                                   SourceLocation NameLoc,
                                                   bool IsSuper,
                                                   bool HasTrailingDot,
                                                   TypeTy *&ReceiverType) {
  ReceiverType = 0;

  if (IsSuper && !HasTrailingDot && S->isInObjcMethodScope())
    return ObjCSuperMessage;
      
  if (TypeTy *TyName = getTypeName(*Name, NameLoc, S)) {
    DeclSpec DS;
    const char *PrevSpec = 0;
    unsigned DiagID = 0;
    if (!DS.SetTypeSpecType(DeclSpec::TST_typename, NameLoc, PrevSpec,
                            DiagID, TyName)) {
      DS.SetRangeEnd(NameLoc);
      Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
      TypeResult Ty = ActOnTypeName(S, DeclaratorInfo);
      if (!Ty.isInvalid())
        ReceiverType = Ty.get();
    }
    return ObjCClassMessage;
  }
      
  return ObjCInstanceMessage;
}

// Defined out-of-line here because of dependecy on AttributeList
Action::DeclPtrTy Action::ActOnUsingDirective(Scope *CurScope,
                                              SourceLocation UsingLoc,
                                              SourceLocation NamespcLoc,
                                              CXXScopeSpec &SS,
                                              SourceLocation IdentLoc,
                                              IdentifierInfo *NamespcName,
                                              AttributeList *AttrList) {

  // FIXME: Parser seems to assume that Action::ActOn* takes ownership over
  // passed AttributeList, however other actions don't free it, is it
  // temporary state or bug?
  delete AttrList;
  return DeclPtrTy();
}

// Defined out-of-line here because of dependency on AttributeList
Action::DeclPtrTy Action::ActOnUsingDeclaration(Scope *CurScope,
                                                AccessSpecifier AS,
                                                bool HasUsingKeyword,
                                                SourceLocation UsingLoc,
                                                CXXScopeSpec &SS,
                                                UnqualifiedId &Name,
                                                AttributeList *AttrList,
                                                bool IsTypeName,
                                                SourceLocation TypenameLoc) {

  // FIXME: Parser seems to assume that Action::ActOn* takes ownership over
  // passed AttributeList, however other actions don't free it, is it
  // temporary state or bug?
  delete AttrList;
  return DeclPtrTy();
}


void PrettyStackTraceActionsDecl::print(llvm::raw_ostream &OS) const {
  if (Loc.isValid()) {
    Loc.print(OS, SM);
    OS << ": ";
  }
  OS << Message;

  std::string Name = Actions.getDeclName(TheDecl);
  if (!Name.empty())
    OS << " '" << Name << '\'';

  OS << '\n';
}

/// TypeNameInfo - A link exists here for each scope that an identifier is
/// defined.
namespace {
  struct TypeNameInfo {
    TypeNameInfo *Prev;
    bool isTypeName;

    TypeNameInfo(bool istypename, TypeNameInfo *prev) {
      isTypeName = istypename;
      Prev = prev;
    }
  };

  struct TypeNameInfoTable {
    llvm::RecyclingAllocator<llvm::BumpPtrAllocator, TypeNameInfo> Allocator;

    void AddEntry(bool isTypename, IdentifierInfo *II) {
      TypeNameInfo *TI = Allocator.Allocate<TypeNameInfo>();
      new (TI) TypeNameInfo(isTypename, II->getFETokenInfo<TypeNameInfo>());
      II->setFETokenInfo(TI);
    }

    void DeleteEntry(TypeNameInfo *Entry) {
      Entry->~TypeNameInfo();
      Allocator.Deallocate(Entry);
    }
  };
}

static TypeNameInfoTable *getTable(void *TP) {
  return static_cast<TypeNameInfoTable*>(TP);
}

MinimalAction::MinimalAction(Preprocessor &pp)
  : Idents(pp.getIdentifierTable()), PP(pp) {
  TypeNameInfoTablePtr = new TypeNameInfoTable();
}

MinimalAction::~MinimalAction() {
  delete getTable(TypeNameInfoTablePtr);
}

void MinimalAction::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;

  TypeNameInfoTable &TNIT = *getTable(TypeNameInfoTablePtr);

  if (PP.getTargetInfo().getPointerWidth(0) >= 64) {
    // Install [u]int128_t for 64-bit targets.
    TNIT.AddEntry(true, &Idents.get("__int128_t"));
    TNIT.AddEntry(true, &Idents.get("__uint128_t"));
  }

  if (PP.getLangOptions().ObjC1) {
    // Recognize the ObjC built-in type identifiers as types.
    TNIT.AddEntry(true, &Idents.get("id"));
    TNIT.AddEntry(true, &Idents.get("SEL"));
    TNIT.AddEntry(true, &Idents.get("Class"));
    TNIT.AddEntry(true, &Idents.get("Protocol"));
  }
}

/// isTypeName - This looks at the IdentifierInfo::FETokenInfo field to
/// determine whether the name is a type name (objc class name or typedef) or
/// not in this scope.
///
/// FIXME: Use the passed CXXScopeSpec for accurate C++ type checking.
Action::TypeTy *
MinimalAction::getTypeName(IdentifierInfo &II, SourceLocation Loc,
                           Scope *S, CXXScopeSpec *SS,
                           bool isClassName, TypeTy *ObjectType) {
  if (TypeNameInfo *TI = II.getFETokenInfo<TypeNameInfo>())
    if (TI->isTypeName)
      return TI;
  return 0;
}

/// isCurrentClassName - Always returns false, because MinimalAction
/// does not support C++ classes with constructors.
bool MinimalAction::isCurrentClassName(const IdentifierInfo &, Scope *,
                                       const CXXScopeSpec *) {
  return false;
}

TemplateNameKind
MinimalAction::isTemplateName(Scope *S,
                              CXXScopeSpec &SS,
                              UnqualifiedId &Name,
                              TypeTy *ObjectType,
                              bool EnteringScope,
                              TemplateTy &TemplateDecl) {
  return TNK_Non_template;
}

/// ActOnDeclarator - If this is a typedef declarator, we modify the
/// IdentifierInfo::FETokenInfo field to keep track of this fact, until S is
/// popped.
Action::DeclPtrTy
MinimalAction::ActOnDeclarator(Scope *S, Declarator &D) {
  IdentifierInfo *II = D.getIdentifier();

  // If there is no identifier associated with this declarator, bail out.
  if (II == 0) return DeclPtrTy();

  TypeNameInfo *weCurrentlyHaveTypeInfo = II->getFETokenInfo<TypeNameInfo>();
  bool isTypeName =
    D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef;

  // this check avoids creating TypeNameInfo objects for the common case.
  // It does need to handle the uncommon case of shadowing a typedef name with a
  // non-typedef name. e.g. { typedef int a; a xx; { int a; } }
  if (weCurrentlyHaveTypeInfo || isTypeName) {
    // Allocate and add the 'TypeNameInfo' "decl".
    getTable(TypeNameInfoTablePtr)->AddEntry(isTypeName, II);

    // Remember that this needs to be removed when the scope is popped.
    S->AddDecl(DeclPtrTy::make(II));
  }
  return DeclPtrTy();
}

Action::DeclPtrTy
MinimalAction::ActOnStartClassInterface(SourceLocation AtInterfaceLoc,
                                        IdentifierInfo *ClassName,
                                        SourceLocation ClassLoc,
                                        IdentifierInfo *SuperName,
                                        SourceLocation SuperLoc,
                                        const DeclPtrTy *ProtoRefs,
                                        unsigned NumProtocols,
                                        const SourceLocation *ProtoLocs,
                                        SourceLocation EndProtoLoc,
                                        AttributeList *AttrList) {
  // Allocate and add the 'TypeNameInfo' "decl".
  getTable(TypeNameInfoTablePtr)->AddEntry(true, ClassName);
  return DeclPtrTy();
}

/// ActOnForwardClassDeclaration -
/// Scope will always be top level file scope.
Action::DeclPtrTy
MinimalAction::ActOnForwardClassDeclaration(SourceLocation AtClassLoc,
                                            IdentifierInfo **IdentList,
                                            SourceLocation *IdentLocs,
                                            unsigned NumElts) {
  for (unsigned i = 0; i != NumElts; ++i) {
    // Allocate and add the 'TypeNameInfo' "decl".
    getTable(TypeNameInfoTablePtr)->AddEntry(true, IdentList[i]);

    // Remember that this needs to be removed when the scope is popped.
    TUScope->AddDecl(DeclPtrTy::make(IdentList[i]));
  }
  return DeclPtrTy();
}

/// ActOnPopScope - When a scope is popped, if any typedefs are now
/// out-of-scope, they are removed from the IdentifierInfo::FETokenInfo field.
void MinimalAction::ActOnPopScope(SourceLocation Loc, Scope *S) {
  TypeNameInfoTable &Table = *getTable(TypeNameInfoTablePtr);

  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    IdentifierInfo &II = *(*I).getAs<IdentifierInfo>();
    TypeNameInfo *TI = II.getFETokenInfo<TypeNameInfo>();
    assert(TI && "This decl didn't get pushed??");

    if (TI) {
      TypeNameInfo *Next = TI->Prev;
      Table.DeleteEntry(TI);

      II.setFETokenInfo(Next);
    }
  }
}
