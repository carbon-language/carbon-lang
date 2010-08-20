//===--- Action.cpp - Implement the Action class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Action interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

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
