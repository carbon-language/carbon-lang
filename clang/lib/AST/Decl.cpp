//===--- Decl.cpp - Declaration AST Node Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// NamedDecl Implementation
//===----------------------------------------------------------------------===//

/// \brief Get the most restrictive linkage for the types in the given
/// template parameter list.
static Linkage 
getLinkageForTemplateParameterList(const TemplateParameterList *Params) {
  Linkage L = ExternalLinkage;
  for (TemplateParameterList::const_iterator P = Params->begin(),
                                          PEnd = Params->end();
       P != PEnd; ++P) {
    if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(*P))
      if (!NTTP->getType()->isDependentType()) {
        L = minLinkage(L, NTTP->getType()->getLinkage());
        continue;
      }

    if (TemplateTemplateParmDecl *TTP
                                   = dyn_cast<TemplateTemplateParmDecl>(*P)) {
      L = minLinkage(L, 
            getLinkageForTemplateParameterList(TTP->getTemplateParameters()));
    }
  }

  return L;
}

/// \brief Get the most restrictive linkage for the types and
/// declarations in the given template argument list.
static Linkage getLinkageForTemplateArgumentList(const TemplateArgument *Args,
                                                 unsigned NumArgs) {
  Linkage L = ExternalLinkage;

  for (unsigned I = 0; I != NumArgs; ++I) {
    switch (Args[I].getKind()) {
    case TemplateArgument::Null:
    case TemplateArgument::Integral:
    case TemplateArgument::Expression:
      break;
      
    case TemplateArgument::Type:
      L = minLinkage(L, Args[I].getAsType()->getLinkage());
      break;

    case TemplateArgument::Declaration:
      if (NamedDecl *ND = dyn_cast<NamedDecl>(Args[I].getAsDecl()))
        L = minLinkage(L, ND->getLinkage());
      if (ValueDecl *VD = dyn_cast<ValueDecl>(Args[I].getAsDecl()))
        L = minLinkage(L, VD->getType()->getLinkage());
      break;

    case TemplateArgument::Template:
      if (TemplateDecl *Template
                                = Args[I].getAsTemplate().getAsTemplateDecl())
        L = minLinkage(L, Template->getLinkage());
      break;

    case TemplateArgument::Pack:
      L = minLinkage(L, 
                     getLinkageForTemplateArgumentList(Args[I].pack_begin(),
                                                       Args[I].pack_size()));
      break;
    }
  }

  return L;
}

static Linkage getLinkageForNamespaceScopeDecl(const NamedDecl *D) {
  assert(D->getDeclContext()->getLookupContext()->isFileContext() &&
         "Not a name having namespace scope");
  ASTContext &Context = D->getASTContext();

  // C++ [basic.link]p3:
  //   A name having namespace scope (3.3.6) has internal linkage if it
  //   is the name of
  //     - an object, reference, function or function template that is
  //       explicitly declared static; or,
  // (This bullet corresponds to C99 6.2.2p3.)
  if (const VarDecl *Var = dyn_cast<VarDecl>(D)) {
    // Explicitly declared static.
    if (Var->getStorageClass() == VarDecl::Static)
      return InternalLinkage;

    // - an object or reference that is explicitly declared const
    //   and neither explicitly declared extern nor previously
    //   declared to have external linkage; or
    // (there is no equivalent in C99)
    if (Context.getLangOptions().CPlusPlus &&
        Var->getType().isConstant(Context) && 
        Var->getStorageClass() != VarDecl::Extern &&
        Var->getStorageClass() != VarDecl::PrivateExtern) {
      bool FoundExtern = false;
      for (const VarDecl *PrevVar = Var->getPreviousDeclaration();
           PrevVar && !FoundExtern; 
           PrevVar = PrevVar->getPreviousDeclaration())
        if (isExternalLinkage(PrevVar->getLinkage()))
          FoundExtern = true;
      
      if (!FoundExtern)
        return InternalLinkage;
    }
  } else if (isa<FunctionDecl>(D) || isa<FunctionTemplateDecl>(D)) {
    // C++ [temp]p4:
    //   A non-member function template can have internal linkage; any
    //   other template name shall have external linkage.
    const FunctionDecl *Function = 0;
    if (const FunctionTemplateDecl *FunTmpl
                                        = dyn_cast<FunctionTemplateDecl>(D))
      Function = FunTmpl->getTemplatedDecl();
    else
      Function = cast<FunctionDecl>(D);

    // Explicitly declared static.
    if (Function->getStorageClass() == FunctionDecl::Static)
      return InternalLinkage;
  } else if (const FieldDecl *Field = dyn_cast<FieldDecl>(D)) {
    //   - a data member of an anonymous union.
    if (cast<RecordDecl>(Field->getDeclContext())->isAnonymousStructOrUnion())
      return InternalLinkage;
  }

  // C++ [basic.link]p4:
  
  //   A name having namespace scope has external linkage if it is the
  //   name of
  //
  //     - an object or reference, unless it has internal linkage; or
  if (const VarDecl *Var = dyn_cast<VarDecl>(D)) {
    if (!Context.getLangOptions().CPlusPlus &&
        (Var->getStorageClass() == VarDecl::Extern ||
         Var->getStorageClass() == VarDecl::PrivateExtern)) {
      // C99 6.2.2p4:
      //   For an identifier declared with the storage-class specifier
      //   extern in a scope in which a prior declaration of that
      //   identifier is visible, if the prior declaration specifies
      //   internal or external linkage, the linkage of the identifier
      //   at the later declaration is the same as the linkage
      //   specified at the prior declaration. If no prior declaration
      //   is visible, or if the prior declaration specifies no
      //   linkage, then the identifier has external linkage.
      if (const VarDecl *PrevVar = Var->getPreviousDeclaration()) {
        if (Linkage L = PrevVar->getLinkage())
          return L;
      }
    }

    // C99 6.2.2p5:
    //   If the declaration of an identifier for an object has file
    //   scope and no storage-class specifier, its linkage is
    //   external.
    if (Var->isInAnonymousNamespace())
      return UniqueExternalLinkage;

    return ExternalLinkage;
  }

  //     - a function, unless it has internal linkage; or
  if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    // C99 6.2.2p5:
    //   If the declaration of an identifier for a function has no
    //   storage-class specifier, its linkage is determined exactly
    //   as if it were declared with the storage-class specifier
    //   extern.
    if (!Context.getLangOptions().CPlusPlus &&
        (Function->getStorageClass() == FunctionDecl::Extern ||
         Function->getStorageClass() == FunctionDecl::PrivateExtern ||
         Function->getStorageClass() == FunctionDecl::None)) {
      // C99 6.2.2p4:
      //   For an identifier declared with the storage-class specifier
      //   extern in a scope in which a prior declaration of that
      //   identifier is visible, if the prior declaration specifies
      //   internal or external linkage, the linkage of the identifier
      //   at the later declaration is the same as the linkage
      //   specified at the prior declaration. If no prior declaration
      //   is visible, or if the prior declaration specifies no
      //   linkage, then the identifier has external linkage.
      if (const FunctionDecl *PrevFunc = Function->getPreviousDeclaration()) {
        if (Linkage L = PrevFunc->getLinkage())
          return L;
      }
    }

    if (Function->isInAnonymousNamespace())
      return UniqueExternalLinkage;

    if (FunctionTemplateSpecializationInfo *SpecInfo
                               = Function->getTemplateSpecializationInfo()) {
      Linkage L = SpecInfo->getTemplate()->getLinkage();
      const TemplateArgumentList &TemplateArgs = *SpecInfo->TemplateArguments;
      L = minLinkage(L, 
                     getLinkageForTemplateArgumentList(
                                          TemplateArgs.getFlatArgumentList(), 
                                          TemplateArgs.flat_size()));
      return L;
    }

    return ExternalLinkage;
  }

  //     - a named class (Clause 9), or an unnamed class defined in a
  //       typedef declaration in which the class has the typedef name
  //       for linkage purposes (7.1.3); or
  //     - a named enumeration (7.2), or an unnamed enumeration
  //       defined in a typedef declaration in which the enumeration
  //       has the typedef name for linkage purposes (7.1.3); or
  if (const TagDecl *Tag = dyn_cast<TagDecl>(D))
    if (Tag->getDeclName() || Tag->getTypedefForAnonDecl()) {
      if (Tag->isInAnonymousNamespace())
        return UniqueExternalLinkage;

      // If this is a class template specialization, consider the
      // linkage of the template and template arguments.
      if (const ClassTemplateSpecializationDecl *Spec
            = dyn_cast<ClassTemplateSpecializationDecl>(Tag)) {
        const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
        Linkage L = getLinkageForTemplateArgumentList(
                                          TemplateArgs.getFlatArgumentList(),
                                                 TemplateArgs.flat_size());
        return minLinkage(L, Spec->getSpecializedTemplate()->getLinkage());
      }

      return ExternalLinkage;
    }

  //     - an enumerator belonging to an enumeration with external linkage;
  if (isa<EnumConstantDecl>(D)) {
    Linkage L = cast<NamedDecl>(D->getDeclContext())->getLinkage();
    if (isExternalLinkage(L))
      return L;
  }

  //     - a template, unless it is a function template that has
  //       internal linkage (Clause 14);
  if (const TemplateDecl *Template = dyn_cast<TemplateDecl>(D)) {
    if (D->isInAnonymousNamespace())
      return UniqueExternalLinkage;

    return getLinkageForTemplateParameterList(
                                         Template->getTemplateParameters());
  }

  //     - a namespace (7.3), unless it is declared within an unnamed
  //       namespace.
  if (isa<NamespaceDecl>(D) && !D->isInAnonymousNamespace())
    return ExternalLinkage;

  return NoLinkage;
}

Linkage NamedDecl::getLinkage() const {

  // Objective-C: treat all Objective-C declarations as having external
  // linkage.
  switch (getKind()) {
    default:
      break;
    case Decl::ObjCAtDefsField:
    case Decl::ObjCCategory:
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCClass:
    case Decl::ObjCCompatibleAlias:
    case Decl::ObjCForwardProtocol:
    case Decl::ObjCImplementation:
    case Decl::ObjCInterface:
    case Decl::ObjCIvar:
    case Decl::ObjCMethod:
    case Decl::ObjCProperty:
    case Decl::ObjCPropertyImpl:
    case Decl::ObjCProtocol:
      return ExternalLinkage;
  }

  // Handle linkage for namespace-scope names.
  if (getDeclContext()->getLookupContext()->isFileContext())
    if (Linkage L = getLinkageForNamespaceScopeDecl(this))
      return L;
  
  // C++ [basic.link]p5:
  //   In addition, a member function, static data member, a named
  //   class or enumeration of class scope, or an unnamed class or
  //   enumeration defined in a class-scope typedef declaration such
  //   that the class or enumeration has the typedef name for linkage
  //   purposes (7.1.3), has external linkage if the name of the class
  //   has external linkage.
  if (getDeclContext()->isRecord() &&
      (isa<CXXMethodDecl>(this) || isa<VarDecl>(this) ||
       (isa<TagDecl>(this) &&
        (getDeclName() || cast<TagDecl>(this)->getTypedefForAnonDecl())))) {
    Linkage L = cast<RecordDecl>(getDeclContext())->getLinkage();
    if (isExternalLinkage(L))
      return L;
  }

  // C++ [basic.link]p6:
  //   The name of a function declared in block scope and the name of
  //   an object declared by a block scope extern declaration have
  //   linkage. If there is a visible declaration of an entity with
  //   linkage having the same name and type, ignoring entities
  //   declared outside the innermost enclosing namespace scope, the
  //   block scope declaration declares that same entity and receives
  //   the linkage of the previous declaration. If there is more than
  //   one such matching entity, the program is ill-formed. Otherwise,
  //   if no matching entity is found, the block scope entity receives
  //   external linkage.
  if (getLexicalDeclContext()->isFunctionOrMethod()) {
    if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(this)) {
      if (Function->getPreviousDeclaration())
        if (Linkage L = Function->getPreviousDeclaration()->getLinkage())
          return L;

      if (Function->isInAnonymousNamespace())
        return UniqueExternalLinkage;

      return ExternalLinkage;
    }

    if (const VarDecl *Var = dyn_cast<VarDecl>(this))
      if (Var->getStorageClass() == VarDecl::Extern ||
          Var->getStorageClass() == VarDecl::PrivateExtern) {
        if (Var->getPreviousDeclaration())
          if (Linkage L = Var->getPreviousDeclaration()->getLinkage())
            return L;

        if (Var->isInAnonymousNamespace())
          return UniqueExternalLinkage;

        return ExternalLinkage;
      }
  }

  // C++ [basic.link]p6:
  //   Names not covered by these rules have no linkage.
  return NoLinkage;
  }

std::string NamedDecl::getQualifiedNameAsString() const {
  return getQualifiedNameAsString(getASTContext().getLangOptions());
}

std::string NamedDecl::getQualifiedNameAsString(const PrintingPolicy &P) const {
  const DeclContext *Ctx = getDeclContext();

  if (Ctx->isFunctionOrMethod())
    return getNameAsString();

  typedef llvm::SmallVector<const DeclContext *, 8> ContextsTy;
  ContextsTy Contexts;

  // Collect contexts.
  while (Ctx && isa<NamedDecl>(Ctx)) {
    Contexts.push_back(Ctx);
    Ctx = Ctx->getParent();
  };

  std::string QualName;
  llvm::raw_string_ostream OS(QualName);

  for (ContextsTy::reverse_iterator I = Contexts.rbegin(), E = Contexts.rend();
       I != E; ++I) {
    if (const ClassTemplateSpecializationDecl *Spec
          = dyn_cast<ClassTemplateSpecializationDecl>(*I)) {
      const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                           TemplateArgs.getFlatArgumentList(),
                                           TemplateArgs.flat_size(),
                                           P);
      OS << Spec->getName() << TemplateArgsStr;
    } else if (const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(*I)) {
      if (ND->isAnonymousNamespace())
        OS << "<anonymous namespace>";
      else
        OS << ND;
    } else if (const RecordDecl *RD = dyn_cast<RecordDecl>(*I)) {
      if (!RD->getIdentifier())
        OS << "<anonymous " << RD->getKindName() << '>';
      else
        OS << RD;
    } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(*I)) {
      const FunctionProtoType *FT = 0;
      if (FD->hasWrittenPrototype())
        FT = dyn_cast<FunctionProtoType>(FD->getType()->getAs<FunctionType>());

      OS << FD << '(';
      if (FT) {
        unsigned NumParams = FD->getNumParams();
        for (unsigned i = 0; i < NumParams; ++i) {
          if (i)
            OS << ", ";
          std::string Param;
          FD->getParamDecl(i)->getType().getAsStringInternal(Param, P);
          OS << Param;
        }

        if (FT->isVariadic()) {
          if (NumParams > 0)
            OS << ", ";
          OS << "...";
        }
      }
      OS << ')';
    } else {
      OS << cast<NamedDecl>(*I);
    }
    OS << "::";
  }

  if (getDeclName())
    OS << this;
  else
    OS << "<anonymous>";

  return OS.str();
}

bool NamedDecl::declarationReplaces(NamedDecl *OldD) const {
  assert(getDeclName() == OldD->getDeclName() && "Declaration name mismatch");

  // UsingDirectiveDecl's are not really NamedDecl's, and all have same name.
  // We want to keep it, unless it nominates same namespace.
  if (getKind() == Decl::UsingDirective) {
    return cast<UsingDirectiveDecl>(this)->getNominatedNamespace() ==
           cast<UsingDirectiveDecl>(OldD)->getNominatedNamespace();
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(this))
    // For function declarations, we keep track of redeclarations.
    return FD->getPreviousDeclaration() == OldD;

  // For function templates, the underlying function declarations are linked.
  if (const FunctionTemplateDecl *FunctionTemplate
        = dyn_cast<FunctionTemplateDecl>(this))
    if (const FunctionTemplateDecl *OldFunctionTemplate
          = dyn_cast<FunctionTemplateDecl>(OldD))
      return FunctionTemplate->getTemplatedDecl()
               ->declarationReplaces(OldFunctionTemplate->getTemplatedDecl());

  // For method declarations, we keep track of redeclarations.
  if (isa<ObjCMethodDecl>(this))
    return false;

  if (isa<ObjCInterfaceDecl>(this) && isa<ObjCCompatibleAliasDecl>(OldD))
    return true;

  if (isa<UsingShadowDecl>(this) && isa<UsingShadowDecl>(OldD))
    return cast<UsingShadowDecl>(this)->getTargetDecl() ==
           cast<UsingShadowDecl>(OldD)->getTargetDecl();

  // For non-function declarations, if the declarations are of the
  // same kind then this must be a redeclaration, or semantic analysis
  // would not have given us the new declaration.
  return this->getKind() == OldD->getKind();
}

bool NamedDecl::hasLinkage() const {
  return getLinkage() != NoLinkage;
}

NamedDecl *NamedDecl::getUnderlyingDecl() {
  NamedDecl *ND = this;
  while (true) {
    if (UsingShadowDecl *UD = dyn_cast<UsingShadowDecl>(ND))
      ND = UD->getTargetDecl();
    else if (ObjCCompatibleAliasDecl *AD
              = dyn_cast<ObjCCompatibleAliasDecl>(ND))
      return AD->getClassInterface();
    else
      return ND;
  }
}

bool NamedDecl::isCXXInstanceMember() const {
  assert(isCXXClassMember() &&
         "checking whether non-member is instance member");

  const NamedDecl *D = this;
  if (isa<UsingShadowDecl>(D))
    D = cast<UsingShadowDecl>(D)->getTargetDecl();

  if (isa<FieldDecl>(D))
    return true;
  if (isa<CXXMethodDecl>(D))
    return cast<CXXMethodDecl>(D)->isInstance();
  if (isa<FunctionTemplateDecl>(D))
    return cast<CXXMethodDecl>(cast<FunctionTemplateDecl>(D)
                                 ->getTemplatedDecl())->isInstance();
  return false;
}

//===----------------------------------------------------------------------===//
// DeclaratorDecl Implementation
//===----------------------------------------------------------------------===//

template <typename DeclT>
static SourceLocation getTemplateOrInnerLocStart(const DeclT *decl) {
  if (decl->getNumTemplateParameterLists() > 0)
    return decl->getTemplateParameterList(0)->getTemplateLoc();
  else
    return decl->getInnerLocStart();
}

SourceLocation DeclaratorDecl::getTypeSpecStartLoc() const {
  TypeSourceInfo *TSI = getTypeSourceInfo();
  if (TSI) return TSI->getTypeLoc().getBeginLoc();
  return SourceLocation();
}

void DeclaratorDecl::setQualifierInfo(NestedNameSpecifier *Qualifier,
                                      SourceRange QualifierRange) {
  if (Qualifier) {
    // Make sure the extended decl info is allocated.
    if (!hasExtInfo()) {
      // Save (non-extended) type source info pointer.
      TypeSourceInfo *savedTInfo = DeclInfo.get<TypeSourceInfo*>();
      // Allocate external info struct.
      DeclInfo = new (getASTContext()) ExtInfo;
      // Restore savedTInfo into (extended) decl info.
      getExtInfo()->TInfo = savedTInfo;
    }
    // Set qualifier info.
    getExtInfo()->NNS = Qualifier;
    getExtInfo()->NNSRange = QualifierRange;
  }
  else {
    // Here Qualifier == 0, i.e., we are removing the qualifier (if any).
    assert(QualifierRange.isInvalid());
    if (hasExtInfo()) {
      // Save type source info pointer.
      TypeSourceInfo *savedTInfo = getExtInfo()->TInfo;
      // Deallocate the extended decl info.
      getASTContext().Deallocate(getExtInfo());
      // Restore savedTInfo into (non-extended) decl info.
      DeclInfo = savedTInfo;
    }
  }
}

SourceLocation DeclaratorDecl::getOuterLocStart() const {
  return getTemplateOrInnerLocStart(this);
}

void
QualifierInfo::setTemplateParameterListsInfo(ASTContext &Context,
                                             unsigned NumTPLists,
                                             TemplateParameterList **TPLists) {
  assert((NumTPLists == 0 || TPLists != 0) &&
         "Empty array of template parameters with positive size!");
  assert((NumTPLists == 0 || NNS) &&
         "Nonempty array of template parameters with no qualifier!");

  // Free previous template parameters (if any).
  if (NumTemplParamLists > 0) {
    Context.Deallocate(TemplParamLists);
    TemplParamLists = 0;
    NumTemplParamLists = 0;
  }
  // Set info on matched template parameter lists (if any).
  if (NumTPLists > 0) {
    TemplParamLists = new (Context) TemplateParameterList*[NumTPLists];
    NumTemplParamLists = NumTPLists;
    for (unsigned i = NumTPLists; i-- > 0; )
      TemplParamLists[i] = TPLists[i];
  }
}

//===----------------------------------------------------------------------===//
// VarDecl Implementation
//===----------------------------------------------------------------------===//

const char *VarDecl::getStorageClassSpecifierString(StorageClass SC) {
  switch (SC) {
  case VarDecl::None:          break;
  case VarDecl::Auto:          return "auto"; break;
  case VarDecl::Extern:        return "extern"; break;
  case VarDecl::PrivateExtern: return "__private_extern__"; break;
  case VarDecl::Register:      return "register"; break;
  case VarDecl::Static:        return "static"; break;
  }

  assert(0 && "Invalid storage class");
  return 0;
}

VarDecl *VarDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                         IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
                         StorageClass S, StorageClass SCAsWritten) {
  return new (C) VarDecl(Var, DC, L, Id, T, TInfo, S, SCAsWritten);
}

SourceLocation VarDecl::getInnerLocStart() const {
  SourceLocation Start = getTypeSpecStartLoc();
  if (Start.isInvalid())
    Start = getLocation();
  return Start;
}

SourceRange VarDecl::getSourceRange() const {
  if (getInit())
    return SourceRange(getOuterLocStart(), getInit()->getLocEnd());
  return SourceRange(getOuterLocStart(), getLocation());
}

bool VarDecl::isExternC() const {
  ASTContext &Context = getASTContext();
  if (!Context.getLangOptions().CPlusPlus)
    return (getDeclContext()->isTranslationUnit() &&
            getStorageClass() != Static) ||
      (getDeclContext()->isFunctionOrMethod() && hasExternalStorage());

  for (const DeclContext *DC = getDeclContext(); !DC->isTranslationUnit();
       DC = DC->getParent()) {
    if (const LinkageSpecDecl *Linkage = dyn_cast<LinkageSpecDecl>(DC))  {
      if (Linkage->getLanguage() == LinkageSpecDecl::lang_c)
        return getStorageClass() != Static;

      break;
    }

    if (DC->isFunctionOrMethod())
      return false;
  }

  return false;
}

VarDecl *VarDecl::getCanonicalDecl() {
  return getFirstDeclaration();
}

VarDecl::DefinitionKind VarDecl::isThisDeclarationADefinition() const {
  // C++ [basic.def]p2:
  //   A declaration is a definition unless [...] it contains the 'extern'
  //   specifier or a linkage-specification and neither an initializer [...],
  //   it declares a static data member in a class declaration [...].
  // C++ [temp.expl.spec]p15:
  //   An explicit specialization of a static data member of a template is a
  //   definition if the declaration includes an initializer; otherwise, it is
  //   a declaration.
  if (isStaticDataMember()) {
    if (isOutOfLine() && (hasInit() ||
          getTemplateSpecializationKind() != TSK_ExplicitSpecialization))
      return Definition;
    else
      return DeclarationOnly;
  }
  // C99 6.7p5:
  //   A definition of an identifier is a declaration for that identifier that
  //   [...] causes storage to be reserved for that object.
  // Note: that applies for all non-file-scope objects.
  // C99 6.9.2p1:
  //   If the declaration of an identifier for an object has file scope and an
  //   initializer, the declaration is an external definition for the identifier
  if (hasInit())
    return Definition;
  // AST for 'extern "C" int foo;' is annotated with 'extern'.
  if (hasExternalStorage())
    return DeclarationOnly;
  
  if (getStorageClassAsWritten() == Extern ||
       getStorageClassAsWritten() == PrivateExtern) {
    for (const VarDecl *PrevVar = getPreviousDeclaration();
         PrevVar; PrevVar = PrevVar->getPreviousDeclaration()) {
      if (PrevVar->getLinkage() == InternalLinkage && PrevVar->hasInit())
        return DeclarationOnly;
    }
  }
  // C99 6.9.2p2:
  //   A declaration of an object that has file scope without an initializer,
  //   and without a storage class specifier or the scs 'static', constitutes
  //   a tentative definition.
  // No such thing in C++.
  if (!getASTContext().getLangOptions().CPlusPlus && isFileVarDecl())
    return TentativeDefinition;

  // What's left is (in C, block-scope) declarations without initializers or
  // external storage. These are definitions.
  return Definition;
}

VarDecl *VarDecl::getActingDefinition() {
  DefinitionKind Kind = isThisDeclarationADefinition();
  if (Kind != TentativeDefinition)
    return 0;

  VarDecl *LastTentative = 0;
  VarDecl *First = getFirstDeclaration();
  for (redecl_iterator I = First->redecls_begin(), E = First->redecls_end();
       I != E; ++I) {
    Kind = (*I)->isThisDeclarationADefinition();
    if (Kind == Definition)
      return 0;
    else if (Kind == TentativeDefinition)
      LastTentative = *I;
  }
  return LastTentative;
}

bool VarDecl::isTentativeDefinitionNow() const {
  DefinitionKind Kind = isThisDeclarationADefinition();
  if (Kind != TentativeDefinition)
    return false;

  for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I) {
    if ((*I)->isThisDeclarationADefinition() == Definition)
      return false;
  }
  return true;
}

VarDecl *VarDecl::getDefinition() {
  VarDecl *First = getFirstDeclaration();
  for (redecl_iterator I = First->redecls_begin(), E = First->redecls_end();
       I != E; ++I) {
    if ((*I)->isThisDeclarationADefinition() == Definition)
      return *I;
  }
  return 0;
}

const Expr *VarDecl::getAnyInitializer(const VarDecl *&D) const {
  redecl_iterator I = redecls_begin(), E = redecls_end();
  while (I != E && !I->getInit())
    ++I;

  if (I != E) {
    D = *I;
    return I->getInit();
  }
  return 0;
}

bool VarDecl::isOutOfLine() const {
  if (Decl::isOutOfLine())
    return true;

  if (!isStaticDataMember())
    return false;

  // If this static data member was instantiated from a static data member of
  // a class template, check whether that static data member was defined 
  // out-of-line.
  if (VarDecl *VD = getInstantiatedFromStaticDataMember())
    return VD->isOutOfLine();
  
  return false;
}

VarDecl *VarDecl::getOutOfLineDefinition() {
  if (!isStaticDataMember())
    return 0;
  
  for (VarDecl::redecl_iterator RD = redecls_begin(), RDEnd = redecls_end();
       RD != RDEnd; ++RD) {
    if (RD->getLexicalDeclContext()->isFileContext())
      return *RD;
  }
  
  return 0;
}

void VarDecl::setInit(Expr *I) {
  if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>()) {
    Eval->~EvaluatedStmt();
    getASTContext().Deallocate(Eval);
  }

  Init = I;
}

VarDecl *VarDecl::getInstantiatedFromStaticDataMember() const {
  if (MemberSpecializationInfo *MSI = getMemberSpecializationInfo())
    return cast<VarDecl>(MSI->getInstantiatedFrom());
  
  return 0;
}

TemplateSpecializationKind VarDecl::getTemplateSpecializationKind() const {
  if (MemberSpecializationInfo *MSI = getMemberSpecializationInfo())
    return MSI->getTemplateSpecializationKind();
  
  return TSK_Undeclared;
}

MemberSpecializationInfo *VarDecl::getMemberSpecializationInfo() const {
  return getASTContext().getInstantiatedFromStaticDataMember(this);
}

void VarDecl::setTemplateSpecializationKind(TemplateSpecializationKind TSK,
                                         SourceLocation PointOfInstantiation) {
  MemberSpecializationInfo *MSI = getMemberSpecializationInfo();
  assert(MSI && "Not an instantiated static data member?");
  MSI->setTemplateSpecializationKind(TSK);
  if (TSK != TSK_ExplicitSpecialization &&
      PointOfInstantiation.isValid() &&
      MSI->getPointOfInstantiation().isInvalid())
    MSI->setPointOfInstantiation(PointOfInstantiation);
}

//===----------------------------------------------------------------------===//
// ParmVarDecl Implementation
//===----------------------------------------------------------------------===//

ParmVarDecl *ParmVarDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 QualType T, TypeSourceInfo *TInfo,
                                 StorageClass S, StorageClass SCAsWritten,
                                 Expr *DefArg) {
  return new (C) ParmVarDecl(ParmVar, DC, L, Id, T, TInfo,
                             S, SCAsWritten, DefArg);
}

Expr *ParmVarDecl::getDefaultArg() {
  assert(!hasUnparsedDefaultArg() && "Default argument is not yet parsed!");
  assert(!hasUninstantiatedDefaultArg() &&
         "Default argument is not yet instantiated!");
  
  Expr *Arg = getInit();
  if (CXXExprWithTemporaries *E = dyn_cast_or_null<CXXExprWithTemporaries>(Arg))
    return E->getSubExpr();

  return Arg;
}

unsigned ParmVarDecl::getNumDefaultArgTemporaries() const {
  if (const CXXExprWithTemporaries *E = 
        dyn_cast<CXXExprWithTemporaries>(getInit()))
    return E->getNumTemporaries();

  return 0;
}

CXXTemporary *ParmVarDecl::getDefaultArgTemporary(unsigned i) {
  assert(getNumDefaultArgTemporaries() && 
         "Default arguments does not have any temporaries!");

  CXXExprWithTemporaries *E = cast<CXXExprWithTemporaries>(getInit());
  return E->getTemporary(i);
}

SourceRange ParmVarDecl::getDefaultArgRange() const {
  if (const Expr *E = getInit())
    return E->getSourceRange();

  if (hasUninstantiatedDefaultArg())
    return getUninstantiatedDefaultArg()->getSourceRange();

  return SourceRange();
}

//===----------------------------------------------------------------------===//
// FunctionDecl Implementation
//===----------------------------------------------------------------------===//

void FunctionDecl::getNameForDiagnostic(std::string &S,
                                        const PrintingPolicy &Policy,
                                        bool Qualified) const {
  NamedDecl::getNameForDiagnostic(S, Policy, Qualified);
  const TemplateArgumentList *TemplateArgs = getTemplateSpecializationArgs();
  if (TemplateArgs)
    S += TemplateSpecializationType::PrintTemplateArgumentList(
                                         TemplateArgs->getFlatArgumentList(),
                                         TemplateArgs->flat_size(),
                                                               Policy);
    
}

bool FunctionDecl::isVariadic() const {
  if (const FunctionProtoType *FT = getType()->getAs<FunctionProtoType>())
    return FT->isVariadic();
  return false;
}

bool FunctionDecl::hasBody(const FunctionDecl *&Definition) const {
  for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I) {
    if (I->Body) {
      Definition = *I;
      return true;
    }
  }

  return false;
}

Stmt *FunctionDecl::getBody(const FunctionDecl *&Definition) const {
  for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I) {
    if (I->Body) {
      Definition = *I;
      return I->Body.get(getASTContext().getExternalSource());
    }
  }

  return 0;
}

void FunctionDecl::setBody(Stmt *B) {
  Body = B;
  if (B)
    EndRangeLoc = B->getLocEnd();
}

bool FunctionDecl::isMain() const {
  ASTContext &Context = getASTContext();
  return !Context.getLangOptions().Freestanding &&
    getDeclContext()->getLookupContext()->isTranslationUnit() &&
    getIdentifier() && getIdentifier()->isStr("main");
}

bool FunctionDecl::isExternC() const {
  ASTContext &Context = getASTContext();
  // In C, any non-static, non-overloadable function has external
  // linkage.
  if (!Context.getLangOptions().CPlusPlus)
    return getStorageClass() != Static && !getAttr<OverloadableAttr>();

  for (const DeclContext *DC = getDeclContext(); !DC->isTranslationUnit();
       DC = DC->getParent()) {
    if (const LinkageSpecDecl *Linkage = dyn_cast<LinkageSpecDecl>(DC))  {
      if (Linkage->getLanguage() == LinkageSpecDecl::lang_c)
        return getStorageClass() != Static &&
               !getAttr<OverloadableAttr>();

      break;
    }
  }

  return false;
}

bool FunctionDecl::isGlobal() const {
  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(this))
    return Method->isStatic();

  if (getStorageClass() == Static)
    return false;

  for (const DeclContext *DC = getDeclContext();
       DC->isNamespace();
       DC = DC->getParent()) {
    if (const NamespaceDecl *Namespace = cast<NamespaceDecl>(DC)) {
      if (!Namespace->getDeclName())
        return false;
      break;
    }
  }

  return true;
}

void
FunctionDecl::setPreviousDeclaration(FunctionDecl *PrevDecl) {
  redeclarable_base::setPreviousDeclaration(PrevDecl);

  if (FunctionTemplateDecl *FunTmpl = getDescribedFunctionTemplate()) {
    FunctionTemplateDecl *PrevFunTmpl
      = PrevDecl? PrevDecl->getDescribedFunctionTemplate() : 0;
    assert((!PrevDecl || PrevFunTmpl) && "Function/function template mismatch");
    FunTmpl->setPreviousDeclaration(PrevFunTmpl);
  }
}

const FunctionDecl *FunctionDecl::getCanonicalDecl() const {
  return getFirstDeclaration();
}

FunctionDecl *FunctionDecl::getCanonicalDecl() {
  return getFirstDeclaration();
}

/// \brief Returns a value indicating whether this function
/// corresponds to a builtin function.
///
/// The function corresponds to a built-in function if it is
/// declared at translation scope or within an extern "C" block and
/// its name matches with the name of a builtin. The returned value
/// will be 0 for functions that do not correspond to a builtin, a
/// value of type \c Builtin::ID if in the target-independent range
/// \c [1,Builtin::First), or a target-specific builtin value.
unsigned FunctionDecl::getBuiltinID() const {
  ASTContext &Context = getASTContext();
  if (!getIdentifier() || !getIdentifier()->getBuiltinID())
    return 0;

  unsigned BuiltinID = getIdentifier()->getBuiltinID();
  if (!Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return BuiltinID;

  // This function has the name of a known C library
  // function. Determine whether it actually refers to the C library
  // function or whether it just has the same name.

  // If this is a static function, it's not a builtin.
  if (getStorageClass() == Static)
    return 0;

  // If this function is at translation-unit scope and we're not in
  // C++, it refers to the C library function.
  if (!Context.getLangOptions().CPlusPlus &&
      getDeclContext()->isTranslationUnit())
    return BuiltinID;

  // If the function is in an extern "C" linkage specification and is
  // not marked "overloadable", it's the real function.
  if (isa<LinkageSpecDecl>(getDeclContext()) &&
      cast<LinkageSpecDecl>(getDeclContext())->getLanguage()
        == LinkageSpecDecl::lang_c &&
      !getAttr<OverloadableAttr>())
    return BuiltinID;

  // Not a builtin
  return 0;
}


/// getNumParams - Return the number of parameters this function must have
/// based on its FunctionType.  This is the length of the PararmInfo array
/// after it has been created.
unsigned FunctionDecl::getNumParams() const {
  const FunctionType *FT = getType()->getAs<FunctionType>();
  if (isa<FunctionNoProtoType>(FT))
    return 0;
  return cast<FunctionProtoType>(FT)->getNumArgs();

}

void FunctionDecl::setParams(ParmVarDecl **NewParamInfo, unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");
  assert(NumParams == getNumParams() && "Parameter count mismatch!");

  // Zero params -> null pointer.
  if (NumParams) {
    void *Mem = getASTContext().Allocate(sizeof(ParmVarDecl*)*NumParams);
    ParamInfo = new (Mem) ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);

    // Update source range. The check below allows us to set EndRangeLoc before
    // setting the parameters.
    if (EndRangeLoc.isInvalid() || EndRangeLoc == getLocation())
      EndRangeLoc = NewParamInfo[NumParams-1]->getLocEnd();
  }
}

/// getMinRequiredArguments - Returns the minimum number of arguments
/// needed to call this function. This may be fewer than the number of
/// function parameters, if some of the parameters have default
/// arguments (in C++).
unsigned FunctionDecl::getMinRequiredArguments() const {
  unsigned NumRequiredArgs = getNumParams();
  while (NumRequiredArgs > 0
         && getParamDecl(NumRequiredArgs-1)->hasDefaultArg())
    --NumRequiredArgs;

  return NumRequiredArgs;
}

bool FunctionDecl::isInlined() const {
  // FIXME: This is not enough. Consider:
  //
  // inline void f();
  // void f() { }
  //
  // f is inlined, but does not have inline specified.
  // To fix this we should add an 'inline' flag to FunctionDecl.
  if (isInlineSpecified())
    return true;
  
  if (isa<CXXMethodDecl>(this)) {
    if (!isOutOfLine() || getCanonicalDecl()->isInlineSpecified())
      return true;
  }

  switch (getTemplateSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
    return false;

  case TSK_ImplicitInstantiation:
  case TSK_ExplicitInstantiationDeclaration:
  case TSK_ExplicitInstantiationDefinition:
    // Handle below.
    break;
  }

  const FunctionDecl *PatternDecl = getTemplateInstantiationPattern();
  bool HasPattern = false;
  if (PatternDecl)
    HasPattern = PatternDecl->hasBody(PatternDecl);
  
  if (HasPattern && PatternDecl)
    return PatternDecl->isInlined();
  
  return false;
}

/// \brief For an inline function definition in C or C++, determine whether the 
/// definition will be externally visible.
///
/// Inline function definitions are always available for inlining optimizations.
/// However, depending on the language dialect, declaration specifiers, and
/// attributes, the definition of an inline function may or may not be
/// "externally" visible to other translation units in the program.
///
/// In C99, inline definitions are not externally visible by default. However,
/// if even one of the global-scope declarations is marked "extern inline", the
/// inline definition becomes externally visible (C99 6.7.4p6).
///
/// In GNU89 mode, or if the gnu_inline attribute is attached to the function
/// definition, we use the GNU semantics for inline, which are nearly the 
/// opposite of C99 semantics. In particular, "inline" by itself will create 
/// an externally visible symbol, but "extern inline" will not create an 
/// externally visible symbol.
bool FunctionDecl::isInlineDefinitionExternallyVisible() const {
  assert(isThisDeclarationADefinition() && "Must have the function definition");
  assert(isInlined() && "Function must be inline");
  ASTContext &Context = getASTContext();
  
  if (!Context.getLangOptions().C99 || hasAttr<GNUInlineAttr>()) {
    // GNU inline semantics. Based on a number of examples, we came up with the
    // following heuristic: if the "inline" keyword is present on a
    // declaration of the function but "extern" is not present on that
    // declaration, then the symbol is externally visible. Otherwise, the GNU
    // "extern inline" semantics applies and the symbol is not externally
    // visible.
    for (redecl_iterator Redecl = redecls_begin(), RedeclEnd = redecls_end();
         Redecl != RedeclEnd;
         ++Redecl) {
      if (Redecl->isInlineSpecified() && Redecl->getStorageClass() != Extern)
        return true;
    }
    
    // GNU "extern inline" semantics; no externally visible symbol.
    return false;
  }
  
  // C99 6.7.4p6:
  //   [...] If all of the file scope declarations for a function in a 
  //   translation unit include the inline function specifier without extern, 
  //   then the definition in that translation unit is an inline definition.
  for (redecl_iterator Redecl = redecls_begin(), RedeclEnd = redecls_end();
       Redecl != RedeclEnd;
       ++Redecl) {
    // Only consider file-scope declarations in this test.
    if (!Redecl->getLexicalDeclContext()->isTranslationUnit())
      continue;
    
    if (!Redecl->isInlineSpecified() || Redecl->getStorageClass() == Extern) 
      return true; // Not an inline definition
  }
  
  // C99 6.7.4p6:
  //   An inline definition does not provide an external definition for the 
  //   function, and does not forbid an external definition in another 
  //   translation unit.
  return false;
}

/// getOverloadedOperator - Which C++ overloaded operator this
/// function represents, if any.
OverloadedOperatorKind FunctionDecl::getOverloadedOperator() const {
  if (getDeclName().getNameKind() == DeclarationName::CXXOperatorName)
    return getDeclName().getCXXOverloadedOperator();
  else
    return OO_None;
}

/// getLiteralIdentifier - The literal suffix identifier this function
/// represents, if any.
const IdentifierInfo *FunctionDecl::getLiteralIdentifier() const {
  if (getDeclName().getNameKind() == DeclarationName::CXXLiteralOperatorName)
    return getDeclName().getCXXLiteralIdentifier();
  else
    return 0;
}

FunctionDecl::TemplatedKind FunctionDecl::getTemplatedKind() const {
  if (TemplateOrSpecialization.isNull())
    return TK_NonTemplate;
  if (TemplateOrSpecialization.is<FunctionTemplateDecl *>())
    return TK_FunctionTemplate;
  if (TemplateOrSpecialization.is<MemberSpecializationInfo *>())
    return TK_MemberSpecialization;
  if (TemplateOrSpecialization.is<FunctionTemplateSpecializationInfo *>())
    return TK_FunctionTemplateSpecialization;
  if (TemplateOrSpecialization.is
                               <DependentFunctionTemplateSpecializationInfo*>())
    return TK_DependentFunctionTemplateSpecialization;

  assert(false && "Did we miss a TemplateOrSpecialization type?");
  return TK_NonTemplate;
}

FunctionDecl *FunctionDecl::getInstantiatedFromMemberFunction() const {
  if (MemberSpecializationInfo *Info = getMemberSpecializationInfo())
    return cast<FunctionDecl>(Info->getInstantiatedFrom());
  
  return 0;
}

MemberSpecializationInfo *FunctionDecl::getMemberSpecializationInfo() const {
  return TemplateOrSpecialization.dyn_cast<MemberSpecializationInfo*>();
}

void 
FunctionDecl::setInstantiationOfMemberFunction(FunctionDecl *FD,
                                               TemplateSpecializationKind TSK) {
  assert(TemplateOrSpecialization.isNull() && 
         "Member function is already a specialization");
  MemberSpecializationInfo *Info 
    = new (getASTContext()) MemberSpecializationInfo(FD, TSK);
  TemplateOrSpecialization = Info;
}

bool FunctionDecl::isImplicitlyInstantiable() const {
  // If the function is invalid, it can't be implicitly instantiated.
  if (isInvalidDecl())
    return false;
  
  switch (getTemplateSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
  case TSK_ExplicitInstantiationDefinition:
    return false;
      
  case TSK_ImplicitInstantiation:
    return true;

  case TSK_ExplicitInstantiationDeclaration:
    // Handled below.
    break;
  }

  // Find the actual template from which we will instantiate.
  const FunctionDecl *PatternDecl = getTemplateInstantiationPattern();
  bool HasPattern = false;
  if (PatternDecl)
    HasPattern = PatternDecl->hasBody(PatternDecl);
  
  // C++0x [temp.explicit]p9:
  //   Except for inline functions, other explicit instantiation declarations
  //   have the effect of suppressing the implicit instantiation of the entity
  //   to which they refer. 
  if (!HasPattern || !PatternDecl) 
    return true;

  return PatternDecl->isInlined();
}                      
   
FunctionDecl *FunctionDecl::getTemplateInstantiationPattern() const {
  if (FunctionTemplateDecl *Primary = getPrimaryTemplate()) {
    while (Primary->getInstantiatedFromMemberTemplate()) {
      // If we have hit a point where the user provided a specialization of
      // this template, we're done looking.
      if (Primary->isMemberSpecialization())
        break;
      
      Primary = Primary->getInstantiatedFromMemberTemplate();
    }
    
    return Primary->getTemplatedDecl();
  } 
    
  return getInstantiatedFromMemberFunction();
}

FunctionTemplateDecl *FunctionDecl::getPrimaryTemplate() const {
  if (FunctionTemplateSpecializationInfo *Info
        = TemplateOrSpecialization
            .dyn_cast<FunctionTemplateSpecializationInfo*>()) {
    return Info->Template.getPointer();
  }
  return 0;
}

const TemplateArgumentList *
FunctionDecl::getTemplateSpecializationArgs() const {
  if (FunctionTemplateSpecializationInfo *Info
        = TemplateOrSpecialization
            .dyn_cast<FunctionTemplateSpecializationInfo*>()) {
    return Info->TemplateArguments;
  }
  return 0;
}

const TemplateArgumentListInfo *
FunctionDecl::getTemplateSpecializationArgsAsWritten() const {
  if (FunctionTemplateSpecializationInfo *Info
        = TemplateOrSpecialization
            .dyn_cast<FunctionTemplateSpecializationInfo*>()) {
    return Info->TemplateArgumentsAsWritten;
  }
  return 0;
}

void
FunctionDecl::setFunctionTemplateSpecialization(FunctionTemplateDecl *Template,
                                     const TemplateArgumentList *TemplateArgs,
                                                void *InsertPos,
                                                TemplateSpecializationKind TSK,
                        const TemplateArgumentListInfo *TemplateArgsAsWritten,
                                          SourceLocation PointOfInstantiation) {
  assert(TSK != TSK_Undeclared && 
         "Must specify the type of function template specialization");
  FunctionTemplateSpecializationInfo *Info
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (!Info)
    Info = new (getASTContext()) FunctionTemplateSpecializationInfo;

  Info->Function = this;
  Info->Template.setPointer(Template);
  Info->Template.setInt(TSK - 1);
  Info->TemplateArguments = TemplateArgs;
  Info->TemplateArgumentsAsWritten = TemplateArgsAsWritten;
  Info->PointOfInstantiation = PointOfInstantiation;
  TemplateOrSpecialization = Info;

  // Insert this function template specialization into the set of known
  // function template specializations.
  if (InsertPos)
    Template->getSpecializations().InsertNode(Info, InsertPos);
  else {
    // Try to insert the new node. If there is an existing node, leave it, the
    // set will contain the canonical decls while
    // FunctionTemplateDecl::findSpecialization will return
    // the most recent redeclarations.
    FunctionTemplateSpecializationInfo *Existing
      = Template->getSpecializations().GetOrInsertNode(Info);
    (void)Existing;
    assert((!Existing || Existing->Function->isCanonicalDecl()) &&
           "Set is supposed to only contain canonical decls");
  }
}

void
FunctionDecl::setFunctionTemplateSpecialization(FunctionTemplateDecl *Template,
                                                unsigned NumTemplateArgs,
                                           const TemplateArgument *TemplateArgs,
                                                 TemplateSpecializationKind TSK,
                                              unsigned NumTemplateArgsAsWritten,
                                   TemplateArgumentLoc *TemplateArgsAsWritten,
                                                SourceLocation LAngleLoc,
                                                SourceLocation RAngleLoc,
                                          SourceLocation PointOfInstantiation) {
  ASTContext &Ctx = getASTContext();
  TemplateArgumentList *TemplArgs
    = new (Ctx) TemplateArgumentList(Ctx, TemplateArgs, NumTemplateArgs);
  TemplateArgumentListInfo *TemplArgsInfo
    = new (Ctx) TemplateArgumentListInfo(LAngleLoc, RAngleLoc);
  for (unsigned i=0; i != NumTemplateArgsAsWritten; ++i)
    TemplArgsInfo->addArgument(TemplateArgsAsWritten[i]);

  setFunctionTemplateSpecialization(Template, TemplArgs, /*InsertPos=*/0, TSK,
                                    TemplArgsInfo, PointOfInstantiation);
}

void
FunctionDecl::setDependentTemplateSpecialization(ASTContext &Context,
                                    const UnresolvedSetImpl &Templates,
                             const TemplateArgumentListInfo &TemplateArgs) {
  assert(TemplateOrSpecialization.isNull());
  size_t Size = sizeof(DependentFunctionTemplateSpecializationInfo);
  Size += Templates.size() * sizeof(FunctionTemplateDecl*);
  Size += TemplateArgs.size() * sizeof(TemplateArgumentLoc);
  void *Buffer = Context.Allocate(Size);
  DependentFunctionTemplateSpecializationInfo *Info =
    new (Buffer) DependentFunctionTemplateSpecializationInfo(Templates,
                                                             TemplateArgs);
  TemplateOrSpecialization = Info;
}

DependentFunctionTemplateSpecializationInfo::
DependentFunctionTemplateSpecializationInfo(const UnresolvedSetImpl &Ts,
                                      const TemplateArgumentListInfo &TArgs)
  : AngleLocs(TArgs.getLAngleLoc(), TArgs.getRAngleLoc()) {

  d.NumTemplates = Ts.size();
  d.NumArgs = TArgs.size();

  FunctionTemplateDecl **TsArray =
    const_cast<FunctionTemplateDecl**>(getTemplates());
  for (unsigned I = 0, E = Ts.size(); I != E; ++I)
    TsArray[I] = cast<FunctionTemplateDecl>(Ts[I]->getUnderlyingDecl());

  TemplateArgumentLoc *ArgsArray =
    const_cast<TemplateArgumentLoc*>(getTemplateArgs());
  for (unsigned I = 0, E = TArgs.size(); I != E; ++I)
    new (&ArgsArray[I]) TemplateArgumentLoc(TArgs[I]);
}

TemplateSpecializationKind FunctionDecl::getTemplateSpecializationKind() const {
  // For a function template specialization, query the specialization
  // information object.
  FunctionTemplateSpecializationInfo *FTSInfo
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (FTSInfo)
    return FTSInfo->getTemplateSpecializationKind();

  MemberSpecializationInfo *MSInfo
    = TemplateOrSpecialization.dyn_cast<MemberSpecializationInfo*>();
  if (MSInfo)
    return MSInfo->getTemplateSpecializationKind();
  
  return TSK_Undeclared;
}

void
FunctionDecl::setTemplateSpecializationKind(TemplateSpecializationKind TSK,
                                          SourceLocation PointOfInstantiation) {
  if (FunctionTemplateSpecializationInfo *FTSInfo
        = TemplateOrSpecialization.dyn_cast<
                                    FunctionTemplateSpecializationInfo*>()) {
    FTSInfo->setTemplateSpecializationKind(TSK);
    if (TSK != TSK_ExplicitSpecialization &&
        PointOfInstantiation.isValid() &&
        FTSInfo->getPointOfInstantiation().isInvalid())
      FTSInfo->setPointOfInstantiation(PointOfInstantiation);
  } else if (MemberSpecializationInfo *MSInfo
             = TemplateOrSpecialization.dyn_cast<MemberSpecializationInfo*>()) {
    MSInfo->setTemplateSpecializationKind(TSK);
    if (TSK != TSK_ExplicitSpecialization &&
        PointOfInstantiation.isValid() &&
        MSInfo->getPointOfInstantiation().isInvalid())
      MSInfo->setPointOfInstantiation(PointOfInstantiation);
  } else
    assert(false && "Function cannot have a template specialization kind");
}

SourceLocation FunctionDecl::getPointOfInstantiation() const {
  if (FunctionTemplateSpecializationInfo *FTSInfo
        = TemplateOrSpecialization.dyn_cast<
                                        FunctionTemplateSpecializationInfo*>())
    return FTSInfo->getPointOfInstantiation();
  else if (MemberSpecializationInfo *MSInfo
             = TemplateOrSpecialization.dyn_cast<MemberSpecializationInfo*>())
    return MSInfo->getPointOfInstantiation();
  
  return SourceLocation();
}

bool FunctionDecl::isOutOfLine() const {
  if (Decl::isOutOfLine())
    return true;
  
  // If this function was instantiated from a member function of a 
  // class template, check whether that member function was defined out-of-line.
  if (FunctionDecl *FD = getInstantiatedFromMemberFunction()) {
    const FunctionDecl *Definition;
    if (FD->hasBody(Definition))
      return Definition->isOutOfLine();
  }
  
  // If this function was instantiated from a function template,
  // check whether that function template was defined out-of-line.
  if (FunctionTemplateDecl *FunTmpl = getPrimaryTemplate()) {
    const FunctionDecl *Definition;
    if (FunTmpl->getTemplatedDecl()->hasBody(Definition))
      return Definition->isOutOfLine();
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// FieldDecl Implementation
//===----------------------------------------------------------------------===//

FieldDecl *FieldDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             IdentifierInfo *Id, QualType T,
                             TypeSourceInfo *TInfo, Expr *BW, bool Mutable) {
  return new (C) FieldDecl(Decl::Field, DC, L, Id, T, TInfo, BW, Mutable);
}

bool FieldDecl::isAnonymousStructOrUnion() const {
  if (!isImplicit() || getDeclName())
    return false;

  if (const RecordType *Record = getType()->getAs<RecordType>())
    return Record->getDecl()->isAnonymousStructOrUnion();

  return false;
}

//===----------------------------------------------------------------------===//
// TagDecl Implementation
//===----------------------------------------------------------------------===//

SourceLocation TagDecl::getOuterLocStart() const {
  return getTemplateOrInnerLocStart(this);
}

SourceRange TagDecl::getSourceRange() const {
  SourceLocation E = RBraceLoc.isValid() ? RBraceLoc : getLocation();
  return SourceRange(getOuterLocStart(), E);
}

TagDecl* TagDecl::getCanonicalDecl() {
  return getFirstDeclaration();
}

void TagDecl::setTypedefForAnonDecl(TypedefDecl *TDD) { 
  TypedefDeclOrQualifier = TDD; 
  if (TypeForDecl)
    TypeForDecl->ClearLinkageCache();
}

void TagDecl::startDefinition() {
  IsBeingDefined = true;

  if (isa<CXXRecordDecl>(this)) {
    CXXRecordDecl *D = cast<CXXRecordDecl>(this);
    struct CXXRecordDecl::DefinitionData *Data = 
      new (getASTContext()) struct CXXRecordDecl::DefinitionData(D);
    for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I)
      cast<CXXRecordDecl>(*I)->DefinitionData = Data;
  }
}

void TagDecl::completeDefinition() {
  assert((!isa<CXXRecordDecl>(this) ||
          cast<CXXRecordDecl>(this)->hasDefinition()) &&
         "definition completed but not started");

  IsDefinition = true;
  IsBeingDefined = false;
}

TagDecl* TagDecl::getDefinition() const {
  if (isDefinition())
    return const_cast<TagDecl *>(this);

  for (redecl_iterator R = redecls_begin(), REnd = redecls_end();
       R != REnd; ++R)
    if (R->isDefinition())
      return *R;

  return 0;
}

void TagDecl::setQualifierInfo(NestedNameSpecifier *Qualifier,
                               SourceRange QualifierRange) {
  if (Qualifier) {
    // Make sure the extended qualifier info is allocated.
    if (!hasExtInfo())
      TypedefDeclOrQualifier = new (getASTContext()) ExtInfo;
    // Set qualifier info.
    getExtInfo()->NNS = Qualifier;
    getExtInfo()->NNSRange = QualifierRange;
  }
  else {
    // Here Qualifier == 0, i.e., we are removing the qualifier (if any).
    assert(QualifierRange.isInvalid());
    if (hasExtInfo()) {
      getASTContext().Deallocate(getExtInfo());
      TypedefDeclOrQualifier = (TypedefDecl*) 0;
    }
  }
}

//===----------------------------------------------------------------------===//
// EnumDecl Implementation
//===----------------------------------------------------------------------===//

EnumDecl *EnumDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                           IdentifierInfo *Id, SourceLocation TKL,
                           EnumDecl *PrevDecl) {
  EnumDecl *Enum = new (C) EnumDecl(DC, L, Id, PrevDecl, TKL);
  C.getTypeDeclType(Enum, PrevDecl);
  return Enum;
}

EnumDecl *EnumDecl::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) EnumDecl(0, SourceLocation(), 0, 0, SourceLocation());
}

void EnumDecl::completeDefinition(QualType NewType,
                                  QualType NewPromotionType,
                                  unsigned NumPositiveBits,
                                  unsigned NumNegativeBits) {
  assert(!isDefinition() && "Cannot redefine enums!");
  IntegerType = NewType;
  PromotionType = NewPromotionType;
  setNumPositiveBits(NumPositiveBits);
  setNumNegativeBits(NumNegativeBits);
  TagDecl::completeDefinition();
}

//===----------------------------------------------------------------------===//
// RecordDecl Implementation
//===----------------------------------------------------------------------===//

RecordDecl::RecordDecl(Kind DK, TagKind TK, DeclContext *DC, SourceLocation L,
                       IdentifierInfo *Id, RecordDecl *PrevDecl,
                       SourceLocation TKL)
  : TagDecl(DK, TK, DC, L, Id, PrevDecl, TKL) {
  HasFlexibleArrayMember = false;
  AnonymousStructOrUnion = false;
  HasObjectMember = false;
  assert(classof(static_cast<Decl*>(this)) && "Invalid Kind!");
}

RecordDecl *RecordDecl::Create(ASTContext &C, TagKind TK, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id,
                               SourceLocation TKL, RecordDecl* PrevDecl) {

  RecordDecl* R = new (C) RecordDecl(Record, TK, DC, L, Id, PrevDecl, TKL);
  C.getTypeDeclType(R, PrevDecl);
  return R;
}

RecordDecl *RecordDecl::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) RecordDecl(Record, TTK_Struct, 0, SourceLocation(), 0, 0,
                            SourceLocation());
}

bool RecordDecl::isInjectedClassName() const {
  return isImplicit() && getDeclName() && getDeclContext()->isRecord() &&
    cast<RecordDecl>(getDeclContext())->getDeclName() == getDeclName();
}

/// completeDefinition - Notes that the definition of this type is now
/// complete.
void RecordDecl::completeDefinition() {
  assert(!isDefinition() && "Cannot redefine record!");
  TagDecl::completeDefinition();
}

ValueDecl *RecordDecl::getAnonymousStructOrUnionObject() {
  // Force the decl chain to come into existence properly.
  if (!getNextDeclInContext()) getParent()->decls_begin();

  assert(isAnonymousStructOrUnion());
  ValueDecl *D = cast<ValueDecl>(getNextDeclInContext());
  assert(D->getType()->isRecordType());
  assert(D->getType()->getAs<RecordType>()->getDecl() == this);
  return D;
}

//===----------------------------------------------------------------------===//
// BlockDecl Implementation
//===----------------------------------------------------------------------===//

void BlockDecl::setParams(ParmVarDecl **NewParamInfo,
                          unsigned NParms) {
  assert(ParamInfo == 0 && "Already has param info!");

  // Zero params -> null pointer.
  if (NParms) {
    NumParams = NParms;
    void *Mem = getASTContext().Allocate(sizeof(ParmVarDecl*)*NumParams);
    ParamInfo = new (Mem) ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);
  }
}

unsigned BlockDecl::getNumParams() const {
  return NumParams;
}


//===----------------------------------------------------------------------===//
// Other Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TranslationUnitDecl *TranslationUnitDecl::Create(ASTContext &C) {
  return new (C) TranslationUnitDecl(C);
}

NamespaceDecl *NamespaceDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L, IdentifierInfo *Id) {
  return new (C) NamespaceDecl(DC, L, Id);
}

ImplicitParamDecl *ImplicitParamDecl::Create(ASTContext &C, DeclContext *DC,
    SourceLocation L, IdentifierInfo *Id, QualType T) {
  return new (C) ImplicitParamDecl(ImplicitParam, DC, L, Id, T);
}

FunctionDecl *FunctionDecl::Create(ASTContext &C, DeclContext *DC,
                                   const DeclarationNameInfo &NameInfo,
                                   QualType T, TypeSourceInfo *TInfo,
                                   StorageClass S, StorageClass SCAsWritten,
                                   bool isInline, bool hasWrittenPrototype) {
  FunctionDecl *New = new (C) FunctionDecl(Function, DC, NameInfo, T, TInfo,
                                           S, SCAsWritten, isInline);
  New->HasWrittenPrototype = hasWrittenPrototype;
  return New;
}

BlockDecl *BlockDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L) {
  return new (C) BlockDecl(DC, L);
}

EnumConstantDecl *EnumConstantDecl::Create(ASTContext &C, EnumDecl *CD,
                                           SourceLocation L,
                                           IdentifierInfo *Id, QualType T,
                                           Expr *E, const llvm::APSInt &V) {
  return new (C) EnumConstantDecl(CD, L, Id, T, E, V);
}

TypedefDecl *TypedefDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 TypeSourceInfo *TInfo) {
  return new (C) TypedefDecl(DC, L, Id, TInfo);
}

FileScopeAsmDecl *FileScopeAsmDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           StringLiteral *Str) {
  return new (C) FileScopeAsmDecl(DC, L, Str);
}
