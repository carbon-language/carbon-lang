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
TypeLoc DeclaratorInfo::getTypeLoc() const {
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
                                 QualType T, DeclaratorInfo *DInfo,
                                 StorageClass S, Expr *DefArg) {
  return new (C) ParmVarDecl(ParmVar, DC, L, Id, T, DInfo, S, DefArg);
}

QualType ParmVarDecl::getOriginalType() const {
  if (const OriginalParmVarDecl *PVD =
      dyn_cast<OriginalParmVarDecl>(this))
    return PVD->OriginalType;
  return getType();
}

SourceRange ParmVarDecl::getDefaultArgRange() const {
  if (const Expr *E = getInit())
    return E->getSourceRange();
  
  if (const Expr *E = getUninstantiatedDefaultArg())
    return E->getSourceRange();
    
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

OriginalParmVarDecl *OriginalParmVarDecl::Create(
                                 ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 QualType T, DeclaratorInfo *DInfo,
                                 QualType OT, StorageClass S, Expr *DefArg) {
  return new (C) OriginalParmVarDecl(DC, L, Id, T, DInfo, OT, S, DefArg);
}

FunctionDecl *FunctionDecl::Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L,
                                   DeclarationName N, QualType T,
                                   DeclaratorInfo *DInfo,
                                   StorageClass S, bool isInline,
                                   bool hasWrittenPrototype) {
  FunctionDecl *New
    = new (C) FunctionDecl(Function, DC, L, N, T, DInfo, S, isInline);
  New->HasWrittenPrototype = hasWrittenPrototype;
  return New;
}

BlockDecl *BlockDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L) {
  return new (C) BlockDecl(DC, L);
}

FieldDecl *FieldDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             IdentifierInfo *Id, QualType T,
                             DeclaratorInfo *DInfo, Expr *BW, bool Mutable) {
  return new (C) FieldDecl(Decl::Field, DC, L, Id, T, DInfo, BW, Mutable);
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
                                 SourceLocation L,
                                 IdentifierInfo *Id, QualType T) {
  return new (C) TypedefDecl(DC, L, Id, T);
}

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

void EnumDecl::completeDefinition(ASTContext &C, QualType NewType) {
  assert(!isDefinition() && "Cannot redefine enums!");
  IntegerType = NewType;
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

std::string NamedDecl::getQualifiedNameAsString() const {
  return getQualifiedNameAsString(getASTContext().getLangOptions());
}

std::string NamedDecl::getQualifiedNameAsString(const PrintingPolicy &P) const {
  std::vector<std::string> Names;
  std::string QualName;
  const DeclContext *Ctx = getDeclContext();

  if (Ctx->isFunctionOrMethod())
    return getNameAsString();

  while (Ctx) {
    if (Ctx->isFunctionOrMethod())
      // FIXME: That probably will happen, when D was member of local
      // scope class/struct/union. How do we handle this case?
      break;

    if (const ClassTemplateSpecializationDecl *Spec
          = dyn_cast<ClassTemplateSpecializationDecl>(Ctx)) {
      const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                           TemplateArgs.getFlatArgumentList(),
                                           TemplateArgs.flat_size(),
                                           P);
      Names.push_back(Spec->getIdentifier()->getName() + TemplateArgsStr);
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

  // For non-function declarations, if the declarations are of the
  // same kind then this must be a redeclaration, or semantic analysis
  // would not have given us the new declaration.
  return this->getKind() == OldD->getKind();
}

bool NamedDecl::hasLinkage() const {
  if (const VarDecl *VD = dyn_cast<VarDecl>(this))
    return VD->hasExternalStorage() || VD->isFileVarDecl();

  if (isa<FunctionDecl>(this) && !isa<CXXMethodDecl>(this))
    return true;

  return false;
}

NamedDecl *NamedDecl::getUnderlyingDecl() {
  NamedDecl *ND = this;
  while (true) {
    if (UsingDecl *UD = dyn_cast<UsingDecl>(ND))
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
  if (DeclInfo)
    return DeclInfo->getTypeLoc().getTypeSpecRange().getBegin();
  return SourceLocation();
}

//===----------------------------------------------------------------------===//
// VarDecl Implementation
//===----------------------------------------------------------------------===//

VarDecl *VarDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                         IdentifierInfo *Id, QualType T, DeclaratorInfo *DInfo,
                         StorageClass S) {
  return new (C) VarDecl(Var, DC, L, Id, T, DInfo, S);
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

void VarDecl::setTemplateSpecializationKind(TemplateSpecializationKind TSK) {
  MemberSpecializationInfo *MSI = getMemberSpecializationInfo();
  assert(MSI && "Not an instantiated static data member?");
  MSI->setTemplateSpecializationKind(TSK);
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

/// \brief For an inline function definition in C, determine whether the 
/// definition will be externally visible.
///
/// Inline function definitions are always available for inlining optimizations.
/// However, depending on the language dialect, declaration specifiers, and
/// attributes, the definition of an inline function may or may not be
/// "externally" visible to other translation units in the program.
///
/// In C99, inline definitions are not externally visible by default. However,
/// if even one of the globa-scope declarations is marked "extern inline", the
/// inline definition becomes externally visible (C99 6.7.4p6).
///
/// In GNU89 mode, or if the gnu_inline attribute is attached to the function
/// definition, we use the GNU semantics for inline, which are nearly the 
/// opposite of C99 semantics. In particular, "inline" by itself will create 
/// an externally visible symbol, but "extern inline" will not create an 
/// externally visible symbol.
bool FunctionDecl::isInlineDefinitionExternallyVisible() const {
  assert(isThisDeclarationADefinition() && "Must have the function definition");
  assert(isInline() && "Function must be inline");
  
  if (!getASTContext().getLangOptions().C99 || hasAttr<GNUInlineAttr>()) {
    // GNU inline semantics. Based on a number of examples, we came up with the
    // following heuristic: if the "inline" keyword is present on a
    // declaration of the function but "extern" is not present on that
    // declaration, then the symbol is externally visible. Otherwise, the GNU
    // "extern inline" semantics applies and the symbol is not externally
    // visible.
    for (redecl_iterator Redecl = redecls_begin(), RedeclEnd = redecls_end();
         Redecl != RedeclEnd;
         ++Redecl) {
      if (Redecl->isInline() && Redecl->getStorageClass() != Extern)
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
    
    if (!Redecl->isInline() || Redecl->getStorageClass() == Extern) 
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
FunctionDecl::setTemplateSpecializationKind(TemplateSpecializationKind TSK) {
  if (FunctionTemplateSpecializationInfo *FTSInfo
        = TemplateOrSpecialization.dyn_cast<
                                        FunctionTemplateSpecializationInfo*>())
    FTSInfo->setTemplateSpecializationKind(TSK);
  else if (MemberSpecializationInfo *MSInfo
             = TemplateOrSpecialization.dyn_cast<MemberSpecializationInfo*>())
    MSInfo->setTemplateSpecializationKind(TSK);
  else
    assert(false && "Function cannot have a template specialization kind");
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
  default: llvm::llvm_unreachable("unexpected type specifier");
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
