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
  return TypeLoc::Create(Ty, (void*)(this + 1));
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

void VarDecl::setInit(ASTContext &C, Expr *I) { 
    if (EvaluatedStmt *Eval = Init.dyn_cast<EvaluatedStmt *>()) {
      Eval->~EvaluatedStmt();
      C.Deallocate(Eval);
    }

    Init = I;
  }

bool VarDecl::isExternC(ASTContext &Context) const {
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
      PrintingPolicy Policy(getASTContext().getLangOptions());
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                           TemplateArgs.getFlatArgumentList(),
                                           TemplateArgs.flat_size(),
                                           Policy);
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

VarDecl *VarDecl::getInstantiatedFromStaticDataMember() {
  return getASTContext().getInstantiatedFromStaticDataMember(this);
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

  C.Deallocate(ParamInfo);

  Decl::Destroy(C);
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

Stmt *FunctionDecl::getBodyIfAvailable() const {
  for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I) {
    if (I->Body && !I->Body.isOffset()) {
      return I->Body.get(0);
    }
  }

  return 0;
}

void FunctionDecl::setBody(Stmt *B) {
  Body = B;
  if (B)
    EndRangeLoc = B->getLocEnd();
}

bool FunctionDecl::isMain(ASTContext &Context) const {
  return !Context.getLangOptions().Freestanding &&
    getDeclContext()->getLookupContext()->isTranslationUnit() &&
    getIdentifier() && getIdentifier()->isStr("main");
}

bool FunctionDecl::isExternC(ASTContext &Context) const {
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
unsigned FunctionDecl::getBuiltinID(ASTContext &Context) const {
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
  const FunctionType *FT = getType()->getAsFunctionType();
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

bool FunctionDecl::hasActiveGNUInlineAttribute(ASTContext &Context) const {
  if (!isInline() || !hasAttr<GNUInlineAttr>())
    return false;

  for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I)
    if (I->isInline() && !I->hasAttr<GNUInlineAttr>())
      return false;

  return true;
}

bool FunctionDecl::isExternGNUInline(ASTContext &Context) const {
  if (!hasActiveGNUInlineAttribute(Context))
    return false;

  for (redecl_iterator I = redecls_begin(), E = redecls_end(); I != E; ++I)
    if (I->getStorageClass() == Extern && I->hasAttr<GNUInlineAttr>())
      return true;

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
                                                void *InsertPos) {
  FunctionTemplateSpecializationInfo *Info 
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (!Info)
    Info = new (Context) FunctionTemplateSpecializationInfo;
  
  Info->Function = this;
  Info->Template.setPointer(Template);
  Info->Template.setInt(0); // Implicit instantiation, unless told otherwise
  Info->TemplateArguments = TemplateArgs;
  TemplateOrSpecialization = Info;
  
  // Insert this function template specialization into the set of known
  // function template specialiations.
  Template->getSpecializations().InsertNode(Info, InsertPos);
}

bool FunctionDecl::isExplicitSpecialization() const {
  // FIXME: check this property for explicit specializations of member
  // functions of class templates.
  FunctionTemplateSpecializationInfo *Info 
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (!Info)
    return false;
  
  return Info->isExplicitSpecialization();
}

void FunctionDecl::setExplicitSpecialization(bool ES) {
  // FIXME: set this property for explicit specializations of member functions
  // of class templates.
  FunctionTemplateSpecializationInfo *Info 
    = TemplateOrSpecialization.dyn_cast<FunctionTemplateSpecializationInfo*>();
  if (Info)
    Info->setExplicitSpecialization(ES);
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
