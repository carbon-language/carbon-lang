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
#include "clang/Parse/DeclSpec.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>

using namespace clang;

void Attr::Destroy(ASTContext &C) {
  if (Next) {
    Next->Destroy(C);
    Next = 0;
  }
  this->~Attr();
  C.Deallocate((void*)this);
}

/// \brief Return the TypeLoc wrapper for the type source info.
TypeLoc TypeSourceInfo::getTypeLoc() const {
  return TypeLoc(Ty, (void*)(this + 1));
}

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//


TranslationUnitDecl *TranslationUnitDecl::Create(ASTContext &C) {
  return new (C) TranslationUnitDecl(C);
}

NamespaceDecl *NamespaceDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L, IdentifierInfo *Id) {
  return new (C) NamespaceDecl(DC, L, Id);
}

void NamespaceDecl::Destroy(ASTContext& C) {
  // NamespaceDecl uses "NextDeclarator" to chain namespace declarations
  // together. They are all top-level Decls.

  this->~NamespaceDecl();
  C.Deallocate((void *)this);
}


ImplicitParamDecl *ImplicitParamDecl::Create(ASTContext &C, DeclContext *DC,
    SourceLocation L, IdentifierInfo *Id, QualType T) {
  return new (C) ImplicitParamDecl(ImplicitParam, DC, L, Id, T);
}

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

ParmVarDecl *ParmVarDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 QualType T, TypeSourceInfo *TInfo,
                                 StorageClass S, Expr *DefArg) {
  return new (C) ParmVarDecl(ParmVar, DC, L, Id, T, TInfo, S, DefArg);
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

void VarDecl::setInit(ASTContext &C, Expr *I) {
  if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>()) {
    Eval->~EvaluatedStmt();
    C.Deallocate(Eval);
  }

  Init = I;
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

FunctionDecl *FunctionDecl::Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L,
                                   DeclarationName N, QualType T,
                                   TypeSourceInfo *TInfo,
                                   StorageClass S, bool isInline,
                                   bool hasWrittenPrototype) {
  FunctionDecl *New
    = new (C) FunctionDecl(Function, DC, L, N, T, TInfo, S, isInline);
  New->HasWrittenPrototype = hasWrittenPrototype;
  return New;
}

BlockDecl *BlockDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L) {
  return new (C) BlockDecl(DC, L);
}

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

EnumConstantDecl *EnumConstantDecl::Create(ASTContext &C, EnumDecl *CD,
                                           SourceLocation L,
                                           IdentifierInfo *Id, QualType T,
                                           Expr *E, const llvm::APSInt &V) {
  return new (C) EnumConstantDecl(CD, L, Id, T, E, V);
}

void EnumConstantDecl::Destroy(ASTContext& C) {
  if (Init) Init->Destroy(C);
  Decl::Destroy(C);
}

TypedefDecl *TypedefDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 TypeSourceInfo *TInfo) {
  return new (C) TypedefDecl(DC, L, Id, TInfo);
}

// Anchor TypedefDecl's vtable here.
TypedefDecl::~TypedefDecl() {}

EnumDecl *EnumDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                           IdentifierInfo *Id, SourceLocation TKL,
                           EnumDecl *PrevDecl) {
  EnumDecl *Enum = new (C) EnumDecl(DC, L, Id, PrevDecl, TKL);
  C.getTypeDeclType(Enum, PrevDecl);
  return Enum;
}

void EnumDecl::Destroy(ASTContext& C) {
  Decl::Destroy(C);
}

void EnumDecl::completeDefinition(ASTContext &C,
                                  QualType NewType,
                                  QualType NewPromotionType) {
  assert(!isDefinition() && "Cannot redefine enums!");
  IntegerType = NewType;
  PromotionType = NewPromotionType;
  TagDecl::completeDefinition();
}

FileScopeAsmDecl *FileScopeAsmDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           StringLiteral *Str) {
  return new (C) FileScopeAsmDecl(DC, L, Str);
}

//===----------------------------------------------------------------------===//
// NamedDecl Implementation
//===----------------------------------------------------------------------===//

static NamedDecl::Linkage getLinkageForNamespaceScopeDecl(const NamedDecl *D) {
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
      return NamedDecl::InternalLinkage;

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
        if (PrevVar->getLinkage() == NamedDecl::ExternalLinkage)
          FoundExtern = true;
      
      if (!FoundExtern)
        return NamedDecl::InternalLinkage;
    }
  } else if (isa<FunctionDecl>(D) || isa<FunctionTemplateDecl>(D)) {
    const FunctionDecl *Function = 0;
    if (const FunctionTemplateDecl *FunTmpl
                                        = dyn_cast<FunctionTemplateDecl>(D))
      Function = FunTmpl->getTemplatedDecl();
    else
      Function = cast<FunctionDecl>(D);

    // Explicitly declared static.
    if (Function->getStorageClass() == FunctionDecl::Static)
      return NamedDecl::InternalLinkage;
  } else if (const FieldDecl *Field = dyn_cast<FieldDecl>(D)) {
    //   - a data member of an anonymous union.
    if (cast<RecordDecl>(Field->getDeclContext())->isAnonymousStructOrUnion())
      return NamedDecl::InternalLinkage;
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
        if (NamedDecl::Linkage L = PrevVar->getLinkage())
          return L;
      }
    }

    // C99 6.2.2p5:
    //   If the declaration of an identifier for an object has file
    //   scope and no storage-class specifier, its linkage is
    //   external.
    return NamedDecl::ExternalLinkage;
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
        if (NamedDecl::Linkage L = PrevFunc->getLinkage())
          return L;
      }
    }

    return NamedDecl::ExternalLinkage;
  }

  //     - a named class (Clause 9), or an unnamed class defined in a
  //       typedef declaration in which the class has the typedef name
  //       for linkage purposes (7.1.3); or
  //     - a named enumeration (7.2), or an unnamed enumeration
  //       defined in a typedef declaration in which the enumeration
  //       has the typedef name for linkage purposes (7.1.3); or
  if (const TagDecl *Tag = dyn_cast<TagDecl>(D))
    if (Tag->getDeclName() || Tag->getTypedefForAnonDecl())
      return NamedDecl::ExternalLinkage;

  //     - an enumerator belonging to an enumeration with external linkage;
  if (isa<EnumConstantDecl>(D))
    if (cast<NamedDecl>(D->getDeclContext())->getLinkage() 
                                                 == NamedDecl::ExternalLinkage)
      return NamedDecl::ExternalLinkage;

  //     - a template, unless it is a function template that has
  //       internal linkage (Clause 14);
  if (isa<TemplateDecl>(D))
    return NamedDecl::ExternalLinkage;

  //     - a namespace (7.3), unless it is declared within an unnamed
  //       namespace.
  if (isa<NamespaceDecl>(D) && !D->isInAnonymousNamespace())
    return NamedDecl::ExternalLinkage;

  return NamedDecl::NoLinkage;
}

NamedDecl::Linkage NamedDecl::getLinkage() const {
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
        (getDeclName() || cast<TagDecl>(this)->getTypedefForAnonDecl()))) &&
      cast<RecordDecl>(getDeclContext())->getLinkage() == ExternalLinkage)
    return ExternalLinkage;

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

      return ExternalLinkage;
    }

    if (const VarDecl *Var = dyn_cast<VarDecl>(this))
      if (Var->getStorageClass() == VarDecl::Extern ||
          Var->getStorageClass() == VarDecl::PrivateExtern) {
        if (Var->getPreviousDeclaration())
          if (Linkage L = Var->getPreviousDeclaration()->getLinkage())
            return L;

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
  // FIXME: Collect contexts, then accumulate names to avoid unnecessary
  // std::string thrashing.
  std::vector<std::string> Names;
  std::string QualName;
  const DeclContext *Ctx = getDeclContext();

  if (Ctx->isFunctionOrMethod())
    return getNameAsString();

  while (Ctx) {
    if (const ClassTemplateSpecializationDecl *Spec
          = dyn_cast<ClassTemplateSpecializationDecl>(Ctx)) {
      const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                           TemplateArgs.getFlatArgumentList(),
                                           TemplateArgs.flat_size(),
                                           P);
      Names.push_back(Spec->getIdentifier()->getNameStart() + TemplateArgsStr);
    } else if (const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(Ctx)) {
      if (ND->isAnonymousNamespace())
        Names.push_back("<anonymous namespace>");
      else
        Names.push_back(ND->getNameAsString());
    } else if (const RecordDecl *RD = dyn_cast<RecordDecl>(Ctx)) {
      if (!RD->getIdentifier()) {
        std::string RecordString = "<anonymous ";
        RecordString += RD->getKindName();
        RecordString += ">";
        Names.push_back(RecordString);
      } else {
        Names.push_back(RD->getNameAsString());
      }
    } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Ctx)) {
      std::string Proto = FD->getNameAsString();

      const FunctionProtoType *FT = 0;
      if (FD->hasWrittenPrototype())
        FT = dyn_cast<FunctionProtoType>(FD->getType()->getAs<FunctionType>());

      Proto += "(";
      if (FT) {
        llvm::raw_string_ostream POut(Proto);
        unsigned NumParams = FD->getNumParams();
        for (unsigned i = 0; i < NumParams; ++i) {
          if (i)
            POut << ", ";
          std::string Param;
          FD->getParamDecl(i)->getType().getAsStringInternal(Param, P);
          POut << Param;
        }

        if (FT->isVariadic()) {
          if (NumParams > 0)
            POut << ", ";
          POut << "...";
        }
      }
      Proto += ")";

      Names.push_back(Proto);
    } else if (const NamedDecl *ND = dyn_cast<NamedDecl>(Ctx))
      Names.push_back(ND->getNameAsString());
    else
      break;

    Ctx = Ctx->getParent();
  }

  std::vector<std::string>::reverse_iterator
    I = Names.rbegin(),
    End = Names.rend();

  for (; I!=End; ++I)
    QualName += *I + "::";

  QualName += getNameAsString();

  return QualName;
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

//===----------------------------------------------------------------------===//
// DeclaratorDecl Implementation
//===----------------------------------------------------------------------===//

SourceLocation DeclaratorDecl::getTypeSpecStartLoc() const {
  if (DeclInfo) {
    TypeLoc TL = DeclInfo->getTypeLoc();
    while (true) {
      TypeLoc NextTL = TL.getNextTypeLoc();
      if (!NextTL)
        return TL.getSourceRange().getBegin();
      TL = NextTL;
    }
  }
  return SourceLocation();
}

//===----------------------------------------------------------------------===//
// VarDecl Implementation
//===----------------------------------------------------------------------===//

VarDecl *VarDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                         IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
                         StorageClass S) {
  return new (C) VarDecl(Var, DC, L, Id, T, TInfo, S);
}

void VarDecl::Destroy(ASTContext& C) {
  Expr *Init = getInit();
  if (Init) {
    Init->Destroy(C);
    if (EvaluatedStmt *Eval = this->Init.dyn_cast<EvaluatedStmt *>()) {
      Eval->~EvaluatedStmt();
      C.Deallocate(Eval);
    }
  }
  this->~VarDecl();
  C.Deallocate((void *)this);
}

VarDecl::~VarDecl() {
}

SourceRange VarDecl::getSourceRange() const {
  if (getInit())
    return SourceRange(getLocation(), getInit()->getLocEnd());
  return SourceRange(getLocation(), getLocation());
}

bool VarDecl::isOutOfLine() const {
  if (!isStaticDataMember())
    return false;
  
  if (Decl::isOutOfLine())
    return true;
  
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

VarDecl *VarDecl::getInstantiatedFromStaticDataMember() const {
  if (MemberSpecializationInfo *MSI = getMemberSpecializationInfo())
    return cast<VarDecl>(MSI->getInstantiatedFrom());
  
  return 0;
}

TemplateSpecializationKind VarDecl::getTemplateSpecializationKind() const {
  if (MemberSpecializationInfo *MSI
        = getASTContext().getInstantiatedFromStaticDataMember(this))
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

bool VarDecl::isTentativeDefinition(ASTContext &Context) const {
  if (!isFileVarDecl() || Context.getLangOptions().CPlusPlus)
    return false;

  const VarDecl *Def = 0;
  return (!getDefinition(Def) &&
          (getStorageClass() == None || getStorageClass() == Static));
}

const Expr *VarDecl::getDefinition(const VarDecl *&Def) const {
  redecl_iterator I = redecls_begin(), E = redecls_end();
  while (I != E && !I->getInit())
    ++I;

  if (I != E) {
    Def = *I;
    return I->getInit();
  }
  return 0;
}

VarDecl *VarDecl::getCanonicalDecl() {
  return getFirstDeclaration();
}

//===----------------------------------------------------------------------===//
// FunctionDecl Implementation
//===----------------------------------------------------------------------===//

void FunctionDecl::Destroy(ASTContext& C) {
  if (Body && Body.isOffset())
    Body.get(C.getExternalSource())->Destroy(C);

  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    (*I)->Destroy(C);

  FunctionTemplateSpecializationInfo *FTSInfo
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (FTSInfo)
    C.Deallocate(FTSInfo);
  
  MemberSpecializationInfo *MSInfo
    = TemplateOrSpecialization.dyn_cast<MemberSpecializationInfo*>();
  if (MSInfo)
    C.Deallocate(MSInfo);
  
  C.Deallocate(ParamInfo);

  Decl::Destroy(C);
}

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

void FunctionDecl::setParams(ASTContext& C, ParmVarDecl **NewParamInfo,
                             unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");
  assert(NumParams == getNumParams() && "Parameter count mismatch!");

  // Zero params -> null pointer.
  if (NumParams) {
    void *Mem = C.Allocate(sizeof(ParmVarDecl*)*NumParams);
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
  Stmt *Pattern = 0;
  if (PatternDecl)
    Pattern = PatternDecl->getBody(PatternDecl);
  
  if (Pattern && PatternDecl)
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

/// getOverloadedOperator - Which C++ overloaded operator this
/// function represents, if any.
OverloadedOperatorKind FunctionDecl::getOverloadedOperator() const {
  if (getDeclName().getNameKind() == DeclarationName::CXXOperatorName)
    return getDeclName().getCXXOverloadedOperator();
  else
    return OO_None;
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
  // If this function already has a definition or is invalid, it can't be
  // implicitly instantiated.
  if (isInvalidDecl() || getBody())
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
  Stmt *Pattern = 0;
  if (PatternDecl)
    Pattern = PatternDecl->getBody(PatternDecl);
  
  // C++0x [temp.explicit]p9:
  //   Except for inline functions, other explicit instantiation declarations
  //   have the effect of suppressing the implicit instantiation of the entity
  //   to which they refer. 
  if (!Pattern || !PatternDecl)
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

void
FunctionDecl::setFunctionTemplateSpecialization(ASTContext &Context,
                                                FunctionTemplateDecl *Template,
                                     const TemplateArgumentList *TemplateArgs,
                                                void *InsertPos,
                                              TemplateSpecializationKind TSK) {
  assert(TSK != TSK_Undeclared && 
         "Must specify the type of function template specialization");
  FunctionTemplateSpecializationInfo *Info
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (!Info)
    Info = new (Context) FunctionTemplateSpecializationInfo;

  Info->Function = this;
  Info->Template.setPointer(Template);
  Info->Template.setInt(TSK - 1);
  Info->TemplateArguments = TemplateArgs;
  TemplateOrSpecialization = Info;

  // Insert this function template specialization into the set of known
  // function template specializations.
  if (InsertPos)
    Template->getSpecializations().InsertNode(Info, InsertPos);
  else {
    // Try to insert the new node. If there is an existing node, remove it 
    // first.
    FunctionTemplateSpecializationInfo *Existing
      = Template->getSpecializations().GetOrInsertNode(Info);
    if (Existing) {
      Template->getSpecializations().RemoveNode(Existing);
      Template->getSpecializations().GetOrInsertNode(Info);
    }
  }
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
    if (FD->getBody(Definition))
      return Definition->isOutOfLine();
  }
  
  // If this function was instantiated from a function template,
  // check whether that function template was defined out-of-line.
  if (FunctionTemplateDecl *FunTmpl = getPrimaryTemplate()) {
    const FunctionDecl *Definition;
    if (FunTmpl->getTemplatedDecl()->getBody(Definition))
      return Definition->isOutOfLine();
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
// TagDecl Implementation
//===----------------------------------------------------------------------===//

SourceRange TagDecl::getSourceRange() const {
  SourceLocation E = RBraceLoc.isValid() ? RBraceLoc : getLocation();
  return SourceRange(TagKeywordLoc, E);
}

TagDecl* TagDecl::getCanonicalDecl() {
  return getFirstDeclaration();
}

void TagDecl::startDefinition() {
  if (TagType *TagT = const_cast<TagType *>(TypeForDecl->getAs<TagType>())) {
    TagT->decl.setPointer(this);
    TagT->decl.setInt(1);
  }
}

void TagDecl::completeDefinition() {
  IsDefinition = true;
  if (TagType *TagT = const_cast<TagType *>(TypeForDecl->getAs<TagType>())) {
    assert(TagT->decl.getPointer() == this &&
           "Attempt to redefine a tag definition?");
    TagT->decl.setInt(0);
  }
}

TagDecl* TagDecl::getDefinition(ASTContext& C) const {
  if (isDefinition())
    return const_cast<TagDecl *>(this);

  for (redecl_iterator R = redecls_begin(), REnd = redecls_end();
       R != REnd; ++R)
    if (R->isDefinition())
      return *R;

  return 0;
}

TagDecl::TagKind TagDecl::getTagKindForTypeSpec(unsigned TypeSpec) {
  switch (TypeSpec) {
  default: llvm_unreachable("unexpected type specifier");
  case DeclSpec::TST_struct: return TK_struct;
  case DeclSpec::TST_class: return TK_class;
  case DeclSpec::TST_union: return TK_union;
  case DeclSpec::TST_enum: return TK_enum;
  }
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

RecordDecl::~RecordDecl() {
}

void RecordDecl::Destroy(ASTContext& C) {
  TagDecl::Destroy(C);
}

bool RecordDecl::isInjectedClassName() const {
  return isImplicit() && getDeclName() && getDeclContext()->isRecord() &&
    cast<RecordDecl>(getDeclContext())->getDeclName() == getDeclName();
}

/// completeDefinition - Notes that the definition of this type is now
/// complete.
void RecordDecl::completeDefinition(ASTContext& C) {
  assert(!isDefinition() && "Cannot redefine record!");
  TagDecl::completeDefinition();
}

//===----------------------------------------------------------------------===//
// BlockDecl Implementation
//===----------------------------------------------------------------------===//

BlockDecl::~BlockDecl() {
}

void BlockDecl::Destroy(ASTContext& C) {
  if (Body)
    Body->Destroy(C);

  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    (*I)->Destroy(C);

  C.Deallocate(ParamInfo);
  Decl::Destroy(C);
}

void BlockDecl::setParams(ASTContext& C, ParmVarDecl **NewParamInfo,
                          unsigned NParms) {
  assert(ParamInfo == 0 && "Already has param info!");

  // Zero params -> null pointer.
  if (NParms) {
    NumParams = NParms;
    void *Mem = C.Allocate(sizeof(ParmVarDecl*)*NumParams);
    ParamInfo = new (Mem) ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);
  }
}

unsigned BlockDecl::getNumParams() const {
  return NumParams;
}
