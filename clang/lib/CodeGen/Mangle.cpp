//===--- Mangle.cpp - Mangle C++ Names --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements C++ name mangling according to the Itanium C++ ABI,
// which is used in GCC 3.2 and newer (and many compilers that are
// ABI-compatible with GCC):
//
//   http://www.codesourcery.com/public/cxx-abi/abi.html
//
//===----------------------------------------------------------------------===//
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "CGVTables.h"

#define MANGLE_CHECKER 0

#if MANGLE_CHECKER
#include <cxxabi.h>
#endif

using namespace clang;
using namespace CodeGen;

MiscNameMangler::MiscNameMangler(MangleContext &C,
                                 llvm::SmallVectorImpl<char> &Res)
  : Context(C), Out(Res) { }

void MiscNameMangler::mangleBlock(GlobalDecl GD, const BlockDecl *BD) {
  // Mangle the context of the block.
  // FIXME: We currently mimic GCC's mangling scheme, which leaves much to be
  // desired. Come up with a better mangling scheme.
  const DeclContext *DC = BD->getDeclContext();
  while (isa<BlockDecl>(DC) || isa<EnumDecl>(DC))
    DC = DC->getParent();
  if (DC->isFunctionOrMethod()) {
    Out << "__";
    if (const ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(DC))
      mangleObjCMethodName(Method);
    else {
      const NamedDecl *ND = cast<NamedDecl>(DC);
      if (IdentifierInfo *II = ND->getIdentifier())
        Out << II->getName();
      else if (const CXXDestructorDecl *D = dyn_cast<CXXDestructorDecl>(ND)) {
        llvm::SmallString<64> Buffer;
        Context.mangleCXXDtor(D, GD.getDtorType(), Buffer);
        Out << Buffer;
      }
      else if (const CXXConstructorDecl *D = dyn_cast<CXXConstructorDecl>(ND)) {
        llvm::SmallString<64> Buffer;
        Context.mangleCXXCtor(D, GD.getCtorType(), Buffer);
        Out << Buffer;
      }
      else {
        // FIXME: We were doing a mangleUnqualifiedName() before, but that's
        // a private member of a class that will soon itself be private to the
        // Itanium C++ ABI object. What should we do now? Right now, I'm just
        // calling the mangleName() method on the MangleContext; is there a
        // better way?
        llvm::SmallString<64> Buffer;
        Context.mangleName(ND, Buffer);
        Out << Buffer;
      }
    }
    Out << "_block_invoke_" << Context.getBlockId(BD, true);
  } else {
    Out << "__block_global_" << Context.getBlockId(BD, false);
  }
}

void MiscNameMangler::mangleObjCMethodName(const ObjCMethodDecl *MD) {
  llvm::SmallString<64> Name;
  llvm::raw_svector_ostream OS(Name);
  
  const ObjCContainerDecl *CD =
  dyn_cast<ObjCContainerDecl>(MD->getDeclContext());
  assert (CD && "Missing container decl in GetNameForMethod");
  OS << (MD->isInstanceMethod() ? '-' : '+') << '[' << CD->getName();
  if (const ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(CD))
    OS << '(' << CID << ')';
  OS << ' ' << MD->getSelector().getAsString() << ']';
  
  Out << OS.str().size() << OS.str();
}

namespace {

static const CXXRecordDecl *GetLocalClassDecl(const NamedDecl *ND) {
  const DeclContext *DC = dyn_cast<DeclContext>(ND);
  if (!DC)
    DC = ND->getDeclContext();
  while (!DC->isNamespace() && !DC->isTranslationUnit()) {
    if (isa<FunctionDecl>(DC->getParent()))
      return dyn_cast<CXXRecordDecl>(DC);
    DC = DC->getParent();
  }
  return 0;
}

static const CXXMethodDecl *getStructor(const CXXMethodDecl *MD) {
  assert((isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD)) &&
         "Passed in decl is not a ctor or dtor!");

  if (const TemplateDecl *TD = MD->getPrimaryTemplate()) {
    MD = cast<CXXMethodDecl>(TD->getTemplatedDecl());

    assert((isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD)) &&
           "Templated decl is not a ctor or dtor!");
  }

  return MD;
}

static const unsigned UnknownArity = ~0U;

/// CXXNameMangler - Manage the mangling of a single name.
class CXXNameMangler {
  MangleContext &Context;
  llvm::raw_svector_ostream Out;

  const CXXMethodDecl *Structor;
  unsigned StructorType;

  /// SeqID - The next subsitution sequence number.
  unsigned SeqID;

  llvm::DenseMap<uintptr_t, unsigned> Substitutions;

  ASTContext &getASTContext() const { return Context.getASTContext(); }

public:
  CXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res)
    : Context(C), Out(Res), Structor(0), StructorType(0), SeqID(0) { }
  CXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res,
                 const CXXConstructorDecl *D, CXXCtorType Type)
    : Context(C), Out(Res), Structor(getStructor(D)), StructorType(Type),
    SeqID(0) { }
  CXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res,
                 const CXXDestructorDecl *D, CXXDtorType Type)
    : Context(C), Out(Res), Structor(getStructor(D)), StructorType(Type),
    SeqID(0) { }

#if MANGLE_CHECKER
  ~CXXNameMangler() {
    if (Out.str()[0] == '\01')
      return;

    int status = 0;
    char *result = abi::__cxa_demangle(Out.str().str().c_str(), 0, 0, &status);
    assert(status == 0 && "Could not demangle mangled name!");
    free(result);
  }
#endif
  llvm::raw_svector_ostream &getStream() { return Out; }

  void mangle(const NamedDecl *D, llvm::StringRef Prefix = "_Z");
  void mangleCallOffset(int64_t NonVirtual, int64_t Virtual);
  void mangleNumber(const llvm::APSInt &I);
  void mangleNumber(int64_t Number);
  void mangleFloat(const llvm::APFloat &F);
  void mangleFunctionEncoding(const FunctionDecl *FD);
  void mangleName(const NamedDecl *ND);
  void mangleType(QualType T);
  void mangleNameOrStandardSubstitution(const NamedDecl *ND);
  
private:
  bool mangleSubstitution(const NamedDecl *ND);
  bool mangleSubstitution(QualType T);
  bool mangleSubstitution(TemplateName Template);
  bool mangleSubstitution(uintptr_t Ptr);

  bool mangleStandardSubstitution(const NamedDecl *ND);

  void addSubstitution(const NamedDecl *ND) {
    ND = cast<NamedDecl>(ND->getCanonicalDecl());

    addSubstitution(reinterpret_cast<uintptr_t>(ND));
  }
  void addSubstitution(QualType T);
  void addSubstitution(TemplateName Template);
  void addSubstitution(uintptr_t Ptr);

  void mangleUnresolvedScope(NestedNameSpecifier *Qualifier);
  void mangleUnresolvedName(NestedNameSpecifier *Qualifier,
                            DeclarationName Name,
                            unsigned KnownArity = UnknownArity);

  void mangleName(const TemplateDecl *TD,
                  const TemplateArgument *TemplateArgs,
                  unsigned NumTemplateArgs);
  void mangleUnqualifiedName(const NamedDecl *ND) {
    mangleUnqualifiedName(ND, ND->getDeclName(), UnknownArity);
  }
  void mangleUnqualifiedName(const NamedDecl *ND, DeclarationName Name,
                             unsigned KnownArity);
  void mangleUnscopedName(const NamedDecl *ND);
  void mangleUnscopedTemplateName(const TemplateDecl *ND);
  void mangleUnscopedTemplateName(TemplateName);
  void mangleSourceName(const IdentifierInfo *II);
  void mangleLocalName(const NamedDecl *ND);
  void mangleNestedName(const NamedDecl *ND, const DeclContext *DC,
                        bool NoFunction=false);
  void mangleNestedName(const TemplateDecl *TD,
                        const TemplateArgument *TemplateArgs,
                        unsigned NumTemplateArgs);
  void manglePrefix(const DeclContext *DC, bool NoFunction=false);
  void mangleTemplatePrefix(const TemplateDecl *ND);
  void mangleTemplatePrefix(TemplateName Template);
  void mangleOperatorName(OverloadedOperatorKind OO, unsigned Arity);
  void mangleQualifiers(Qualifiers Quals);

  void mangleObjCMethodName(const ObjCMethodDecl *MD);

  // Declare manglers for every type class.
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) void mangleType(const CLASS##Type *T);
#include "clang/AST/TypeNodes.def"

  void mangleType(const TagType*);
  void mangleType(TemplateName);
  void mangleBareFunctionType(const FunctionType *T,
                              bool MangleReturnType);
  bool mangleNeonVectorType(const VectorType *T);

  void mangleIntegerLiteral(QualType T, const llvm::APSInt &Value);
  void mangleMemberExpr(const Expr *Base, bool IsArrow,
                        NestedNameSpecifier *Qualifier,
                        DeclarationName Name,
                        unsigned KnownArity);
  void mangleExpression(const Expr *E, unsigned Arity = UnknownArity);
  void mangleCXXCtorType(CXXCtorType T);
  void mangleCXXDtorType(CXXDtorType T);

  void mangleTemplateArgs(const ExplicitTemplateArgumentList &TemplateArgs);
  void mangleTemplateArgs(TemplateName Template,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs);  
  void mangleTemplateArgs(const TemplateParameterList &PL,
                          const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs);
  void mangleTemplateArgs(const TemplateParameterList &PL,
                          const TemplateArgumentList &AL);
  void mangleTemplateArg(const NamedDecl *P, const TemplateArgument &A);

  void mangleTemplateParameter(unsigned Index);
};
}

static bool isInCLinkageSpecification(const Decl *D) {
  D = D->getCanonicalDecl();
  for (const DeclContext *DC = D->getDeclContext();
       !DC->isTranslationUnit(); DC = DC->getParent()) {
    if (const LinkageSpecDecl *Linkage = dyn_cast<LinkageSpecDecl>(DC))
      return Linkage->getLanguage() == LinkageSpecDecl::lang_c;
  }

  return false;
}

bool MangleContext::shouldMangleDeclName(const NamedDecl *D) {
  // In C, functions with no attributes never need to be mangled. Fastpath them.
  if (!getASTContext().getLangOptions().CPlusPlus && !D->hasAttrs())
    return false;

  // Any decl can be declared with __asm("foo") on it, and this takes precedence
  // over all other naming in the .o file.
  if (D->hasAttr<AsmLabelAttr>())
    return true;

  // Clang's "overloadable" attribute extension to C/C++ implies name mangling
  // (always) as does passing a C++ member function and a function
  // whose name is not a simple identifier.
  const FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (FD && (FD->hasAttr<OverloadableAttr>() || isa<CXXMethodDecl>(FD) ||
             !FD->getDeclName().isIdentifier()))
    return true;

  // Otherwise, no mangling is done outside C++ mode.
  if (!getASTContext().getLangOptions().CPlusPlus)
    return false;

  // Variables at global scope with non-internal linkage are not mangled
  if (!FD) {
    const DeclContext *DC = D->getDeclContext();
    // Check for extern variable declared locally.
    if (DC->isFunctionOrMethod() && D->hasLinkage())
      while (!DC->isNamespace() && !DC->isTranslationUnit())
        DC = DC->getParent();
    if (DC->isTranslationUnit() && D->getLinkage() != InternalLinkage)
      return false;
  }

  // Class members are always mangled.
  if (D->getDeclContext()->isRecord())
    return true;

  // C functions and "main" are not mangled.
  if ((FD && FD->isMain()) || isInCLinkageSpecification(D))
    return false;

  return true;
}

void CXXNameMangler::mangle(const NamedDecl *D, llvm::StringRef Prefix) {
  // Any decl can be declared with __asm("foo") on it, and this takes precedence
  // over all other naming in the .o file.
  if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>()) {
    // If we have an asm name, then we use it as the mangling.
    Out << '\01';  // LLVM IR Marker for __asm("foo")
    Out << ALA->getLabel();
    return;
  }

  // <mangled-name> ::= _Z <encoding>
  //            ::= <data name>
  //            ::= <special-name>
  Out << Prefix;
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    mangleFunctionEncoding(FD);
  else if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    mangleName(VD);
  else
    mangleName(cast<FieldDecl>(D));
}

void CXXNameMangler::mangleFunctionEncoding(const FunctionDecl *FD) {
  // <encoding> ::= <function name> <bare-function-type>
  mangleName(FD);

  // Don't mangle in the type if this isn't a decl we should typically mangle.
  if (!Context.shouldMangleDeclName(FD))
    return;

  // Whether the mangling of a function type includes the return type depends on
  // the context and the nature of the function. The rules for deciding whether
  // the return type is included are:
  //
  //   1. Template functions (names or types) have return types encoded, with
  //   the exceptions listed below.
  //   2. Function types not appearing as part of a function name mangling,
  //   e.g. parameters, pointer types, etc., have return type encoded, with the
  //   exceptions listed below.
  //   3. Non-template function names do not have return types encoded.
  //
  // The exceptions mentioned in (1) and (2) above, for which the return type is
  // never included, are
  //   1. Constructors.
  //   2. Destructors.
  //   3. Conversion operator functions, e.g. operator int.
  bool MangleReturnType = false;
  if (FunctionTemplateDecl *PrimaryTemplate = FD->getPrimaryTemplate()) {
    if (!(isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD) ||
          isa<CXXConversionDecl>(FD)))
      MangleReturnType = true;

    // Mangle the type of the primary template.
    FD = PrimaryTemplate->getTemplatedDecl();
  }

  // Do the canonicalization out here because parameter types can
  // undergo additional canonicalization (e.g. array decay).
  FunctionType *FT = cast<FunctionType>(Context.getASTContext()
                                          .getCanonicalType(FD->getType()));

  mangleBareFunctionType(FT, MangleReturnType);
}

static const DeclContext *IgnoreLinkageSpecDecls(const DeclContext *DC) {
  while (isa<LinkageSpecDecl>(DC)) {
    DC = DC->getParent();
  }

  return DC;
}

/// isStd - Return whether a given namespace is the 'std' namespace.
static bool isStd(const NamespaceDecl *NS) {
  if (!IgnoreLinkageSpecDecls(NS->getParent())->isTranslationUnit())
    return false;
  
  const IdentifierInfo *II = NS->getOriginalNamespace()->getIdentifier();
  return II && II->isStr("std");
}

// isStdNamespace - Return whether a given decl context is a toplevel 'std'
// namespace.
static bool isStdNamespace(const DeclContext *DC) {
  if (!DC->isNamespace())
    return false;

  return isStd(cast<NamespaceDecl>(DC));
}

static const TemplateDecl *
isTemplate(const NamedDecl *ND, const TemplateArgumentList *&TemplateArgs) {
  // Check if we have a function template.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)){
    if (const TemplateDecl *TD = FD->getPrimaryTemplate()) {
      TemplateArgs = FD->getTemplateSpecializationArgs();
      return TD;
    }
  }

  // Check if we have a class template.
  if (const ClassTemplateSpecializationDecl *Spec =
        dyn_cast<ClassTemplateSpecializationDecl>(ND)) {
    TemplateArgs = &Spec->getTemplateArgs();
    return Spec->getSpecializedTemplate();
  }

  return 0;
}

void CXXNameMangler::mangleName(const NamedDecl *ND) {
  //  <name> ::= <nested-name>
  //         ::= <unscoped-name>
  //         ::= <unscoped-template-name> <template-args>
  //         ::= <local-name>
  //
  const DeclContext *DC = ND->getDeclContext();

  // If this is an extern variable declared locally, the relevant DeclContext
  // is that of the containing namespace, or the translation unit.
  if (isa<FunctionDecl>(DC) && ND->hasLinkage())
    while (!DC->isNamespace() && !DC->isTranslationUnit())
      DC = DC->getParent();
  else if (GetLocalClassDecl(ND)) {
    mangleLocalName(ND);
    return;
  }

  while (isa<LinkageSpecDecl>(DC))
    DC = DC->getParent();

  if (DC->isTranslationUnit() || isStdNamespace(DC)) {
    // Check if we have a template.
    const TemplateArgumentList *TemplateArgs = 0;
    if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
      mangleUnscopedTemplateName(TD);
      TemplateParameterList *TemplateParameters = TD->getTemplateParameters();
      mangleTemplateArgs(*TemplateParameters, *TemplateArgs);
      return;
    }

    mangleUnscopedName(ND);
    return;
  }

  if (isa<FunctionDecl>(DC) || isa<ObjCMethodDecl>(DC)) {
    mangleLocalName(ND);
    return;
  }

  mangleNestedName(ND, DC);
}
void CXXNameMangler::mangleName(const TemplateDecl *TD,
                                const TemplateArgument *TemplateArgs,
                                unsigned NumTemplateArgs) {
  const DeclContext *DC = IgnoreLinkageSpecDecls(TD->getDeclContext());

  if (DC->isTranslationUnit() || isStdNamespace(DC)) {
    mangleUnscopedTemplateName(TD);
    TemplateParameterList *TemplateParameters = TD->getTemplateParameters();
    mangleTemplateArgs(*TemplateParameters, TemplateArgs, NumTemplateArgs);
  } else {
    mangleNestedName(TD, TemplateArgs, NumTemplateArgs);
  }
}

void CXXNameMangler::mangleUnscopedName(const NamedDecl *ND) {
  //  <unscoped-name> ::= <unqualified-name>
  //                  ::= St <unqualified-name>   # ::std::
  if (isStdNamespace(ND->getDeclContext()))
    Out << "St";

  mangleUnqualifiedName(ND);
}

void CXXNameMangler::mangleUnscopedTemplateName(const TemplateDecl *ND) {
  //     <unscoped-template-name> ::= <unscoped-name>
  //                              ::= <substitution>
  if (mangleSubstitution(ND))
    return;

  // <template-template-param> ::= <template-param>
  if (const TemplateTemplateParmDecl *TTP
                                     = dyn_cast<TemplateTemplateParmDecl>(ND)) {
    mangleTemplateParameter(TTP->getIndex());
    return;
  }

  mangleUnscopedName(ND->getTemplatedDecl());
  addSubstitution(ND);
}

void CXXNameMangler::mangleUnscopedTemplateName(TemplateName Template) {
  //     <unscoped-template-name> ::= <unscoped-name>
  //                              ::= <substitution>
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleUnscopedTemplateName(TD);
  
  if (mangleSubstitution(Template))
    return;

  // FIXME: How to cope with operators here?
  DependentTemplateName *Dependent = Template.getAsDependentTemplateName();
  assert(Dependent && "Not a dependent template name?");
  if (!Dependent->isIdentifier()) {
    // FIXME: We can't possibly know the arity of the operator here!
    Diagnostic &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(Diagnostic::Error,
                                      "cannot mangle dependent operator name");
    Diags.Report(FullSourceLoc(), DiagID);
    return;
  }
  
  mangleSourceName(Dependent->getIdentifier());
  addSubstitution(Template);
}

void CXXNameMangler::mangleFloat(const llvm::APFloat &F) {
  // TODO: avoid this copy with careful stream management.
  llvm::SmallString<20> Buffer;
  F.bitcastToAPInt().toString(Buffer, 16, false);
  Out.write(Buffer.data(), Buffer.size());
}

void CXXNameMangler::mangleNumber(const llvm::APSInt &Value) {
  if (Value.isSigned() && Value.isNegative()) {
    Out << 'n';
    Value.abs().print(Out, true);
  } else
    Value.print(Out, Value.isSigned());
}

void CXXNameMangler::mangleNumber(int64_t Number) {
  //  <number> ::= [n] <non-negative decimal integer>
  if (Number < 0) {
    Out << 'n';
    Number = -Number;
  }

  Out << Number;
}

void CXXNameMangler::mangleCallOffset(int64_t NonVirtual, int64_t Virtual) {
  //  <call-offset>  ::= h <nv-offset> _
  //                 ::= v <v-offset> _
  //  <nv-offset>    ::= <offset number>        # non-virtual base override
  //  <v-offset>     ::= <offset number> _ <virtual offset number>
  //                      # virtual base override, with vcall offset
  if (!Virtual) {
    Out << 'h';
    mangleNumber(NonVirtual);
    Out << '_';
    return;
  }

  Out << 'v';
  mangleNumber(NonVirtual);
  Out << '_';
  mangleNumber(Virtual);
  Out << '_';
}

void CXXNameMangler::mangleUnresolvedScope(NestedNameSpecifier *Qualifier) {
  Qualifier = getASTContext().getCanonicalNestedNameSpecifier(Qualifier);
  switch (Qualifier->getKind()) {
  case NestedNameSpecifier::Global:
    // nothing
    break;
  case NestedNameSpecifier::Namespace:
    mangleName(Qualifier->getAsNamespace());
    break;
  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    const Type *QTy = Qualifier->getAsType();

    if (const TemplateSpecializationType *TST =
        dyn_cast<TemplateSpecializationType>(QTy)) {
      if (!mangleSubstitution(QualType(TST, 0))) {
        mangleTemplatePrefix(TST->getTemplateName());
        
        // FIXME: GCC does not appear to mangle the template arguments when
        // the template in question is a dependent template name. Should we
        // emulate that badness?
        mangleTemplateArgs(TST->getTemplateName(), TST->getArgs(),
                           TST->getNumArgs());
        addSubstitution(QualType(TST, 0));
      }
    } else {
      // We use the QualType mangle type variant here because it handles
      // substitutions.
      mangleType(QualType(QTy, 0));
    }
  }
    break;
  case NestedNameSpecifier::Identifier:
    // Member expressions can have these without prefixes.
    if (Qualifier->getPrefix())
      mangleUnresolvedScope(Qualifier->getPrefix());
    mangleSourceName(Qualifier->getAsIdentifier());
    break;
  }
}

/// Mangles a name which was not resolved to a specific entity.
void CXXNameMangler::mangleUnresolvedName(NestedNameSpecifier *Qualifier,
                                          DeclarationName Name,
                                          unsigned KnownArity) {
  if (Qualifier)
    mangleUnresolvedScope(Qualifier);
  // FIXME: ambiguity of unqualified lookup with ::

  mangleUnqualifiedName(0, Name, KnownArity);
}

static const FieldDecl *FindFirstNamedDataMember(const RecordDecl *RD) {
  assert(RD->isAnonymousStructOrUnion() &&
         "Expected anonymous struct or union!");
  
  for (RecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I) {
    const FieldDecl *FD = *I;
    
    if (FD->getIdentifier())
      return FD;
    
    if (const RecordType *RT = FD->getType()->getAs<RecordType>()) {
      if (const FieldDecl *NamedDataMember = 
          FindFirstNamedDataMember(RT->getDecl()))
        return NamedDataMember;
    }
  }

  // We didn't find a named data member.
  return 0;
}

void CXXNameMangler::mangleUnqualifiedName(const NamedDecl *ND,
                                           DeclarationName Name,
                                           unsigned KnownArity) {
  //  <unqualified-name> ::= <operator-name>
  //                     ::= <ctor-dtor-name>
  //                     ::= <source-name>
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier: {
    if (const IdentifierInfo *II = Name.getAsIdentifierInfo()) {
      // We must avoid conflicts between internally- and externally-
      // linked variable declaration names in the same TU.
      // This naming convention is the same as that followed by GCC, though it
      // shouldn't actually matter.
      if (ND && isa<VarDecl>(ND) && ND->getLinkage() == InternalLinkage &&
          ND->getDeclContext()->isFileContext())
        Out << 'L';

      mangleSourceName(II);
      break;
    }

    // Otherwise, an anonymous entity.  We must have a declaration.
    assert(ND && "mangling empty name without declaration");

    if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
      if (NS->isAnonymousNamespace()) {
        // This is how gcc mangles these names.
        Out << "12_GLOBAL__N_1";
        break;
      }
    }

    if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
      // We must have an anonymous union or struct declaration.
      const RecordDecl *RD = 
        cast<RecordDecl>(VD->getType()->getAs<RecordType>()->getDecl());
      
      // Itanium C++ ABI 5.1.2:
      //
      //   For the purposes of mangling, the name of an anonymous union is
      //   considered to be the name of the first named data member found by a
      //   pre-order, depth-first, declaration-order walk of the data members of
      //   the anonymous union. If there is no such data member (i.e., if all of
      //   the data members in the union are unnamed), then there is no way for
      //   a program to refer to the anonymous union, and there is therefore no
      //   need to mangle its name.
      const FieldDecl *FD = FindFirstNamedDataMember(RD);

      // It's actually possible for various reasons for us to get here
      // with an empty anonymous struct / union.  Fortunately, it
      // doesn't really matter what name we generate.
      if (!FD) break;
      assert(FD->getIdentifier() && "Data member name isn't an identifier!");
      
      mangleSourceName(FD->getIdentifier());
      break;
    }
    
    // We must have an anonymous struct.
    const TagDecl *TD = cast<TagDecl>(ND);
    if (const TypedefDecl *D = TD->getTypedefForAnonDecl()) {
      assert(TD->getDeclContext() == D->getDeclContext() &&
             "Typedef should not be in another decl context!");
      assert(D->getDeclName().getAsIdentifierInfo() &&
             "Typedef was not named!");
      mangleSourceName(D->getDeclName().getAsIdentifierInfo());
      break;
    }

    // Get a unique id for the anonymous struct.
    uint64_t AnonStructId = Context.getAnonymousStructId(TD);

    // Mangle it as a source name in the form
    // [n] $_<id>
    // where n is the length of the string.
    llvm::SmallString<8> Str;
    Str += "$_";
    Str += llvm::utostr(AnonStructId);

    Out << Str.size();
    Out << Str.str();
    break;
  }

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    assert(false && "Can't mangle Objective-C selector names here!");
    break;

  case DeclarationName::CXXConstructorName:
    if (ND == Structor)
      // If the named decl is the C++ constructor we're mangling, use the type
      // we were given.
      mangleCXXCtorType(static_cast<CXXCtorType>(StructorType));
    else
      // Otherwise, use the complete constructor name. This is relevant if a
      // class with a constructor is declared within a constructor.
      mangleCXXCtorType(Ctor_Complete);
    break;

  case DeclarationName::CXXDestructorName:
    if (ND == Structor)
      // If the named decl is the C++ destructor we're mangling, use the type we
      // were given.
      mangleCXXDtorType(static_cast<CXXDtorType>(StructorType));
    else
      // Otherwise, use the complete destructor name. This is relevant if a
      // class with a destructor is declared within a destructor.
      mangleCXXDtorType(Dtor_Complete);
    break;

  case DeclarationName::CXXConversionFunctionName:
    // <operator-name> ::= cv <type>    # (cast)
    Out << "cv";
    mangleType(Context.getASTContext().getCanonicalType(Name.getCXXNameType()));
    break;

  case DeclarationName::CXXOperatorName: {
    unsigned Arity;
    if (ND) {
      Arity = cast<FunctionDecl>(ND)->getNumParams();

      // If we have a C++ member function, we need to include the 'this' pointer.
      // FIXME: This does not make sense for operators that are static, but their
      // names stay the same regardless of the arity (operator new for instance).
      if (isa<CXXMethodDecl>(ND))
        Arity++;
    } else
      Arity = KnownArity;

    mangleOperatorName(Name.getCXXOverloadedOperator(), Arity);
    break;
  }

  case DeclarationName::CXXLiteralOperatorName:
    // FIXME: This mangling is not yet official.
    Out << "li";
    mangleSourceName(Name.getCXXLiteralIdentifier());
    break;

  case DeclarationName::CXXUsingDirective:
    assert(false && "Can't mangle a using directive name!");
    break;
  }
}

void CXXNameMangler::mangleSourceName(const IdentifierInfo *II) {
  // <source-name> ::= <positive length number> <identifier>
  // <number> ::= [n] <non-negative decimal integer>
  // <identifier> ::= <unqualified source code identifier>
  Out << II->getLength() << II->getName();
}

void CXXNameMangler::mangleNestedName(const NamedDecl *ND,
                                      const DeclContext *DC,
                                      bool NoFunction) {
  // <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
  //               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E

  Out << 'N';
  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(ND))
    mangleQualifiers(Qualifiers::fromCVRMask(Method->getTypeQualifiers()));

  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = 0;
  if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
    mangleTemplatePrefix(TD);
    TemplateParameterList *TemplateParameters = TD->getTemplateParameters();
    mangleTemplateArgs(*TemplateParameters, *TemplateArgs);
  }
  else {
    manglePrefix(DC, NoFunction);
    mangleUnqualifiedName(ND);
  }

  Out << 'E';
}
void CXXNameMangler::mangleNestedName(const TemplateDecl *TD,
                                      const TemplateArgument *TemplateArgs,
                                      unsigned NumTemplateArgs) {
  // <nested-name> ::= N [<CV-qualifiers>] <template-prefix> <template-args> E

  Out << 'N';

  mangleTemplatePrefix(TD);
  TemplateParameterList *TemplateParameters = TD->getTemplateParameters();
  mangleTemplateArgs(*TemplateParameters, TemplateArgs, NumTemplateArgs);

  Out << 'E';
}

void CXXNameMangler::mangleLocalName(const NamedDecl *ND) {
  // <local-name> := Z <function encoding> E <entity name> [<discriminator>]
  //              := Z <function encoding> E s [<discriminator>]
  // <discriminator> := _ <non-negative number>
  const DeclContext *DC = ND->getDeclContext();
  Out << 'Z';

  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(DC)) {
   mangleObjCMethodName(MD);
  } else if (const CXXRecordDecl *RD = GetLocalClassDecl(ND)) {
    mangleFunctionEncoding(cast<FunctionDecl>(RD->getDeclContext()));
    Out << 'E';

    // Mangle the name relative to the closest enclosing function.
    if (ND == RD) // equality ok because RD derived from ND above
      mangleUnqualifiedName(ND);
    else
      mangleNestedName(ND, DC, true /*NoFunction*/);

    unsigned disc;
    if (Context.getNextDiscriminator(RD, disc)) {
      if (disc < 10)
        Out << '_' << disc;
      else
        Out << "__" << disc << '_';
    }

    return;
  }
  else
    mangleFunctionEncoding(cast<FunctionDecl>(DC));

  Out << 'E';
  mangleUnqualifiedName(ND);
}

void CXXNameMangler::manglePrefix(const DeclContext *DC, bool NoFunction) {
  //  <prefix> ::= <prefix> <unqualified-name>
  //           ::= <template-prefix> <template-args>
  //           ::= <template-param>
  //           ::= # empty
  //           ::= <substitution>

  while (isa<LinkageSpecDecl>(DC))
    DC = DC->getParent();

  if (DC->isTranslationUnit())
    return;

  if (const BlockDecl *Block = dyn_cast<BlockDecl>(DC)) {
    manglePrefix(DC->getParent(), NoFunction);    
    llvm::SmallString<64> Name;
    Context.mangleBlock(GlobalDecl(), Block, Name);
    Out << Name.size() << Name;
    return;
  }
  
  if (mangleSubstitution(cast<NamedDecl>(DC)))
    return;

  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = 0;
  if (const TemplateDecl *TD = isTemplate(cast<NamedDecl>(DC), TemplateArgs)) {
    mangleTemplatePrefix(TD);
    TemplateParameterList *TemplateParameters = TD->getTemplateParameters();
    mangleTemplateArgs(*TemplateParameters, *TemplateArgs);
  }
  else if(NoFunction && (isa<FunctionDecl>(DC) || isa<ObjCMethodDecl>(DC)))
    return;
  else if (const ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(DC))
    mangleObjCMethodName(Method);
  else {
    manglePrefix(DC->getParent(), NoFunction);
    mangleUnqualifiedName(cast<NamedDecl>(DC));
  }

  addSubstitution(cast<NamedDecl>(DC));
}

void CXXNameMangler::mangleTemplatePrefix(TemplateName Template) {
  // <template-prefix> ::= <prefix> <template unqualified-name>
  //                   ::= <template-param>
  //                   ::= <substitution>
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleTemplatePrefix(TD);

  if (QualifiedTemplateName *Qualified = Template.getAsQualifiedTemplateName())
    mangleUnresolvedScope(Qualified->getQualifier());
  
  if (OverloadedTemplateStorage *Overloaded
                                      = Template.getAsOverloadedTemplate()) {
    mangleUnqualifiedName(0, (*Overloaded->begin())->getDeclName(), 
                          UnknownArity);
    return;
  }
   
  DependentTemplateName *Dependent = Template.getAsDependentTemplateName();
  assert(Dependent && "Unknown template name kind?");
  mangleUnresolvedScope(Dependent->getQualifier());
  mangleUnscopedTemplateName(Template);
}

void CXXNameMangler::mangleTemplatePrefix(const TemplateDecl *ND) {
  // <template-prefix> ::= <prefix> <template unqualified-name>
  //                   ::= <template-param>
  //                   ::= <substitution>
  // <template-template-param> ::= <template-param>
  //                               <substitution>

  if (mangleSubstitution(ND))
    return;

  // <template-template-param> ::= <template-param>
  if (const TemplateTemplateParmDecl *TTP
                                     = dyn_cast<TemplateTemplateParmDecl>(ND)) {
    mangleTemplateParameter(TTP->getIndex());
    return;
  }

  manglePrefix(ND->getDeclContext());
  mangleUnqualifiedName(ND->getTemplatedDecl());
  addSubstitution(ND);
}

/// Mangles a template name under the production <type>.  Required for
/// template template arguments.
///   <type> ::= <class-enum-type>
///          ::= <template-param>
///          ::= <substitution>
void CXXNameMangler::mangleType(TemplateName TN) {
  if (mangleSubstitution(TN))
    return;
      
  TemplateDecl *TD = 0;

  switch (TN.getKind()) {
  case TemplateName::QualifiedTemplate:
    TD = TN.getAsQualifiedTemplateName()->getTemplateDecl();
    goto HaveDecl;

  case TemplateName::Template:
    TD = TN.getAsTemplateDecl();
    goto HaveDecl;

  HaveDecl:
    if (isa<TemplateTemplateParmDecl>(TD))
      mangleTemplateParameter(cast<TemplateTemplateParmDecl>(TD)->getIndex());
    else
      mangleName(TD);
    break;

  case TemplateName::OverloadedTemplate:
    llvm_unreachable("can't mangle an overloaded template name as a <type>");
    break;

  case TemplateName::DependentTemplate: {
    const DependentTemplateName *Dependent = TN.getAsDependentTemplateName();
    assert(Dependent->isIdentifier());

    // <class-enum-type> ::= <name>
    // <name> ::= <nested-name>
    mangleUnresolvedScope(Dependent->getQualifier());
    mangleSourceName(Dependent->getIdentifier());
    break;
  }

  }

  addSubstitution(TN);
}

void
CXXNameMangler::mangleOperatorName(OverloadedOperatorKind OO, unsigned Arity) {
  switch (OO) {
  // <operator-name> ::= nw     # new
  case OO_New: Out << "nw"; break;
  //              ::= na        # new[]
  case OO_Array_New: Out << "na"; break;
  //              ::= dl        # delete
  case OO_Delete: Out << "dl"; break;
  //              ::= da        # delete[]
  case OO_Array_Delete: Out << "da"; break;
  //              ::= ps        # + (unary)
  //              ::= pl        # + (binary or unknown)
  case OO_Plus:
    Out << (Arity == 1? "ps" : "pl"); break;
  //              ::= ng        # - (unary)
  //              ::= mi        # - (binary or unknown)
  case OO_Minus:
    Out << (Arity == 1? "ng" : "mi"); break;
  //              ::= ad        # & (unary)
  //              ::= an        # & (binary or unknown)
  case OO_Amp:
    Out << (Arity == 1? "ad" : "an"); break;
  //              ::= de        # * (unary)
  //              ::= ml        # * (binary or unknown)
  case OO_Star:
    // Use binary when unknown.
    Out << (Arity == 1? "de" : "ml"); break;
  //              ::= co        # ~
  case OO_Tilde: Out << "co"; break;
  //              ::= dv        # /
  case OO_Slash: Out << "dv"; break;
  //              ::= rm        # %
  case OO_Percent: Out << "rm"; break;
  //              ::= or        # |
  case OO_Pipe: Out << "or"; break;
  //              ::= eo        # ^
  case OO_Caret: Out << "eo"; break;
  //              ::= aS        # =
  case OO_Equal: Out << "aS"; break;
  //              ::= pL        # +=
  case OO_PlusEqual: Out << "pL"; break;
  //              ::= mI        # -=
  case OO_MinusEqual: Out << "mI"; break;
  //              ::= mL        # *=
  case OO_StarEqual: Out << "mL"; break;
  //              ::= dV        # /=
  case OO_SlashEqual: Out << "dV"; break;
  //              ::= rM        # %=
  case OO_PercentEqual: Out << "rM"; break;
  //              ::= aN        # &=
  case OO_AmpEqual: Out << "aN"; break;
  //              ::= oR        # |=
  case OO_PipeEqual: Out << "oR"; break;
  //              ::= eO        # ^=
  case OO_CaretEqual: Out << "eO"; break;
  //              ::= ls        # <<
  case OO_LessLess: Out << "ls"; break;
  //              ::= rs        # >>
  case OO_GreaterGreater: Out << "rs"; break;
  //              ::= lS        # <<=
  case OO_LessLessEqual: Out << "lS"; break;
  //              ::= rS        # >>=
  case OO_GreaterGreaterEqual: Out << "rS"; break;
  //              ::= eq        # ==
  case OO_EqualEqual: Out << "eq"; break;
  //              ::= ne        # !=
  case OO_ExclaimEqual: Out << "ne"; break;
  //              ::= lt        # <
  case OO_Less: Out << "lt"; break;
  //              ::= gt        # >
  case OO_Greater: Out << "gt"; break;
  //              ::= le        # <=
  case OO_LessEqual: Out << "le"; break;
  //              ::= ge        # >=
  case OO_GreaterEqual: Out << "ge"; break;
  //              ::= nt        # !
  case OO_Exclaim: Out << "nt"; break;
  //              ::= aa        # &&
  case OO_AmpAmp: Out << "aa"; break;
  //              ::= oo        # ||
  case OO_PipePipe: Out << "oo"; break;
  //              ::= pp        # ++
  case OO_PlusPlus: Out << "pp"; break;
  //              ::= mm        # --
  case OO_MinusMinus: Out << "mm"; break;
  //              ::= cm        # ,
  case OO_Comma: Out << "cm"; break;
  //              ::= pm        # ->*
  case OO_ArrowStar: Out << "pm"; break;
  //              ::= pt        # ->
  case OO_Arrow: Out << "pt"; break;
  //              ::= cl        # ()
  case OO_Call: Out << "cl"; break;
  //              ::= ix        # []
  case OO_Subscript: Out << "ix"; break;

  //              ::= qu        # ?
  // The conditional operator can't be overloaded, but we still handle it when
  // mangling expressions.
  case OO_Conditional: Out << "qu"; break;

  case OO_None:
  case NUM_OVERLOADED_OPERATORS:
    assert(false && "Not an overloaded operator");
    break;
  }
}

void CXXNameMangler::mangleQualifiers(Qualifiers Quals) {
  // <CV-qualifiers> ::= [r] [V] [K]    # restrict (C99), volatile, const
  if (Quals.hasRestrict())
    Out << 'r';
  if (Quals.hasVolatile())
    Out << 'V';
  if (Quals.hasConst())
    Out << 'K';

  if (Quals.hasAddressSpace()) {
    // Extension:
    //
    //   <type> ::= U <address-space-number>
    // 
    // where <address-space-number> is a source name consisting of 'AS' 
    // followed by the address space <number>.
    llvm::SmallString<64> ASString;
    ASString = "AS" + llvm::utostr_32(Quals.getAddressSpace());
    Out << 'U' << ASString.size() << ASString;
  }
  
  // FIXME: For now, just drop all extension qualifiers on the floor.
}

void CXXNameMangler::mangleObjCMethodName(const ObjCMethodDecl *MD) {
  llvm::SmallString<64> Buffer;
  MiscNameMangler(Context, Buffer).mangleObjCMethodName(MD);
  Out << Buffer;
}

void CXXNameMangler::mangleType(QualType T) {
  // Only operate on the canonical type!
  T = Context.getASTContext().getCanonicalType(T);

  bool IsSubstitutable = T.hasLocalQualifiers() || !isa<BuiltinType>(T);
  if (IsSubstitutable && mangleSubstitution(T))
    return;

  if (Qualifiers Quals = T.getLocalQualifiers()) {
    mangleQualifiers(Quals);
    // Recurse:  even if the qualified type isn't yet substitutable,
    // the unqualified type might be.
    mangleType(T.getLocalUnqualifiedType());
  } else {
    switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT) \
    case Type::CLASS: \
      llvm_unreachable("can't mangle non-canonical type " #CLASS "Type"); \
      return;
#define TYPE(CLASS, PARENT) \
    case Type::CLASS: \
      mangleType(static_cast<const CLASS##Type*>(T.getTypePtr())); \
      break;
#include "clang/AST/TypeNodes.def"
    }
  }

  // Add the substitution.
  if (IsSubstitutable)
    addSubstitution(T);
}

void CXXNameMangler::mangleNameOrStandardSubstitution(const NamedDecl *ND) {
  if (!mangleStandardSubstitution(ND))
    mangleName(ND);
}

void CXXNameMangler::mangleType(const BuiltinType *T) {
  //  <type>         ::= <builtin-type>
  //  <builtin-type> ::= v  # void
  //                 ::= w  # wchar_t
  //                 ::= b  # bool
  //                 ::= c  # char
  //                 ::= a  # signed char
  //                 ::= h  # unsigned char
  //                 ::= s  # short
  //                 ::= t  # unsigned short
  //                 ::= i  # int
  //                 ::= j  # unsigned int
  //                 ::= l  # long
  //                 ::= m  # unsigned long
  //                 ::= x  # long long, __int64
  //                 ::= y  # unsigned long long, __int64
  //                 ::= n  # __int128
  // UNSUPPORTED:    ::= o  # unsigned __int128
  //                 ::= f  # float
  //                 ::= d  # double
  //                 ::= e  # long double, __float80
  // UNSUPPORTED:    ::= g  # __float128
  // UNSUPPORTED:    ::= Dd # IEEE 754r decimal floating point (64 bits)
  // UNSUPPORTED:    ::= De # IEEE 754r decimal floating point (128 bits)
  // UNSUPPORTED:    ::= Df # IEEE 754r decimal floating point (32 bits)
  // UNSUPPORTED:    ::= Dh # IEEE 754r half-precision floating point (16 bits)
  //                 ::= Di # char32_t
  //                 ::= Ds # char16_t
  //                 ::= Dn # std::nullptr_t (i.e., decltype(nullptr))
  //                 ::= u <source-name>    # vendor extended type
  switch (T->getKind()) {
  case BuiltinType::Void: Out << 'v'; break;
  case BuiltinType::Bool: Out << 'b'; break;
  case BuiltinType::Char_U: case BuiltinType::Char_S: Out << 'c'; break;
  case BuiltinType::UChar: Out << 'h'; break;
  case BuiltinType::UShort: Out << 't'; break;
  case BuiltinType::UInt: Out << 'j'; break;
  case BuiltinType::ULong: Out << 'm'; break;
  case BuiltinType::ULongLong: Out << 'y'; break;
  case BuiltinType::UInt128: Out << 'o'; break;
  case BuiltinType::SChar: Out << 'a'; break;
  case BuiltinType::WChar: Out << 'w'; break;
  case BuiltinType::Char16: Out << "Ds"; break;
  case BuiltinType::Char32: Out << "Di"; break;
  case BuiltinType::Short: Out << 's'; break;
  case BuiltinType::Int: Out << 'i'; break;
  case BuiltinType::Long: Out << 'l'; break;
  case BuiltinType::LongLong: Out << 'x'; break;
  case BuiltinType::Int128: Out << 'n'; break;
  case BuiltinType::Float: Out << 'f'; break;
  case BuiltinType::Double: Out << 'd'; break;
  case BuiltinType::LongDouble: Out << 'e'; break;
  case BuiltinType::NullPtr: Out << "Dn"; break;

  case BuiltinType::Overload:
  case BuiltinType::Dependent:
    assert(false &&
           "Overloaded and dependent types shouldn't get to name mangling");
    break;
  case BuiltinType::UndeducedAuto:
    assert(0 && "Should not see undeduced auto here");
    break;
  case BuiltinType::ObjCId: Out << "11objc_object"; break;
  case BuiltinType::ObjCClass: Out << "10objc_class"; break;
  case BuiltinType::ObjCSel: Out << "13objc_selector"; break;
  }
}

// <type>          ::= <function-type>
// <function-type> ::= F [Y] <bare-function-type> E
void CXXNameMangler::mangleType(const FunctionProtoType *T) {
  Out << 'F';
  // FIXME: We don't have enough information in the AST to produce the 'Y'
  // encoding for extern "C" function types.
  mangleBareFunctionType(T, /*MangleReturnType=*/true);
  Out << 'E';
}
void CXXNameMangler::mangleType(const FunctionNoProtoType *T) {
  llvm_unreachable("Can't mangle K&R function prototypes");
}
void CXXNameMangler::mangleBareFunctionType(const FunctionType *T,
                                            bool MangleReturnType) {
  // We should never be mangling something without a prototype.
  const FunctionProtoType *Proto = cast<FunctionProtoType>(T);

  // <bare-function-type> ::= <signature type>+
  if (MangleReturnType)
    mangleType(Proto->getResultType());

  if (Proto->getNumArgs() == 0 && !Proto->isVariadic()) {
    //   <builtin-type> ::= v   # void
    Out << 'v';
    return;
  }

  for (FunctionProtoType::arg_type_iterator Arg = Proto->arg_type_begin(),
                                         ArgEnd = Proto->arg_type_end();
       Arg != ArgEnd; ++Arg)
    mangleType(*Arg);

  // <builtin-type>      ::= z  # ellipsis
  if (Proto->isVariadic())
    Out << 'z';
}

// <type>            ::= <class-enum-type>
// <class-enum-type> ::= <name>
void CXXNameMangler::mangleType(const UnresolvedUsingType *T) {
  mangleName(T->getDecl());
}

// <type>            ::= <class-enum-type>
// <class-enum-type> ::= <name>
void CXXNameMangler::mangleType(const EnumType *T) {
  mangleType(static_cast<const TagType*>(T));
}
void CXXNameMangler::mangleType(const RecordType *T) {
  mangleType(static_cast<const TagType*>(T));
}
void CXXNameMangler::mangleType(const TagType *T) {
  mangleName(T->getDecl());
}

// <type>       ::= <array-type>
// <array-type> ::= A <positive dimension number> _ <element type>
//              ::= A [<dimension expression>] _ <element type>
void CXXNameMangler::mangleType(const ConstantArrayType *T) {
  Out << 'A' << T->getSize() << '_';
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const VariableArrayType *T) {
  Out << 'A';
  // decayed vla types (size 0) will just be skipped.
  if (T->getSizeExpr())
    mangleExpression(T->getSizeExpr());
  Out << '_';
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const DependentSizedArrayType *T) {
  Out << 'A';
  mangleExpression(T->getSizeExpr());
  Out << '_';
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const IncompleteArrayType *T) {
  Out << "A_";
  mangleType(T->getElementType());
}

// <type>                   ::= <pointer-to-member-type>
// <pointer-to-member-type> ::= M <class type> <member type>
void CXXNameMangler::mangleType(const MemberPointerType *T) {
  Out << 'M';
  mangleType(QualType(T->getClass(), 0));
  QualType PointeeType = T->getPointeeType();
  if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(PointeeType)) {
    mangleQualifiers(Qualifiers::fromCVRMask(FPT->getTypeQuals()));
    mangleType(FPT);
    
    // Itanium C++ ABI 5.1.8:
    //
    //   The type of a non-static member function is considered to be different,
    //   for the purposes of substitution, from the type of a namespace-scope or
    //   static member function whose type appears similar. The types of two
    //   non-static member functions are considered to be different, for the
    //   purposes of substitution, if the functions are members of different
    //   classes. In other words, for the purposes of substitution, the class of 
    //   which the function is a member is considered part of the type of 
    //   function.

    // We increment the SeqID here to emulate adding an entry to the
    // substitution table. We can't actually add it because we don't want this
    // particular function type to be substituted.
    ++SeqID;
  } else
    mangleType(PointeeType);
}

// <type>           ::= <template-param>
void CXXNameMangler::mangleType(const TemplateTypeParmType *T) {
  mangleTemplateParameter(T->getIndex());
}

// <type> ::= P <type>   # pointer-to
void CXXNameMangler::mangleType(const PointerType *T) {
  Out << 'P';
  mangleType(T->getPointeeType());
}
void CXXNameMangler::mangleType(const ObjCObjectPointerType *T) {
  Out << 'P';
  mangleType(T->getPointeeType());
}

// <type> ::= R <type>   # reference-to
void CXXNameMangler::mangleType(const LValueReferenceType *T) {
  Out << 'R';
  mangleType(T->getPointeeType());
}

// <type> ::= O <type>   # rvalue reference-to (C++0x)
void CXXNameMangler::mangleType(const RValueReferenceType *T) {
  Out << 'O';
  mangleType(T->getPointeeType());
}

// <type> ::= C <type>   # complex pair (C 2000)
void CXXNameMangler::mangleType(const ComplexType *T) {
  Out << 'C';
  mangleType(T->getElementType());
}

// ARM's ABI for Neon vector types specifies that they should be mangled as
// if they are structs (to match ARM's initial implementation).  If the
// vector type is not one of the special types predefined by ARM, return false
// so it will be mangled as an ordinary vector type.
bool CXXNameMangler::mangleNeonVectorType(const VectorType *T) {
  QualType EltType = T->getElementType();
  if (!EltType->isBuiltinType())
    return false;
  unsigned EltBits = 0;
  const char *EltName = 0;
  switch (cast<BuiltinType>(EltType)->getKind()) {
  case BuiltinType::SChar:     EltBits =  8; EltName = "int8_t"; break;
  case BuiltinType::UChar:     EltBits =  8; EltName = "uint8_t"; break;
  case BuiltinType::Short:     EltBits = 16; EltName = "int16_t"; break;
  case BuiltinType::UShort:    EltBits = 16; EltName = "uint16_t"; break;
  case BuiltinType::Int:       EltBits = 32; EltName = "int32_t"; break;
  case BuiltinType::UInt:      EltBits = 32; EltName = "uint32_t"; break;
  case BuiltinType::LongLong:  EltBits = 64; EltName = "int64_t"; break;
  case BuiltinType::ULongLong: EltBits = 64; EltName = "uint64_t"; break;
  case BuiltinType::Float:     EltBits = 32; EltName = "float32_t"; break;
  default: return false;
  }
  const char *BaseName = 0;
  unsigned BitSize = T->getNumElements() * EltBits;
  if (BitSize == 64)
    BaseName = "__simd64_";
  else if (BitSize == 128)
    BaseName = "__simd128_";
  else
    return false;
  Out << strlen(BaseName) + strlen(EltName);
  Out << BaseName << EltName;
  return true;
}

// GNU extension: vector types
// <type>                  ::= <vector-type>
// <vector-type>           ::= Dv <positive dimension number> _
//                                    <extended element type>
//                         ::= Dv [<dimension expression>] _ <element type>
// <extended element type> ::= <element type>
//                         ::= p # AltiVec vector pixel
void CXXNameMangler::mangleType(const VectorType *T) {
  if (T->getVectorKind() == VectorType::NeonVector && mangleNeonVectorType(T))
    return;
  Out << "Dv" << T->getNumElements() << '_';
  if (T->getVectorKind() == VectorType::AltiVecPixel)
    Out << 'p';
  else if (T->getVectorKind() == VectorType::AltiVecBool)
    Out << 'b';
  else
    mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const ExtVectorType *T) {
  mangleType(static_cast<const VectorType*>(T));
}
void CXXNameMangler::mangleType(const DependentSizedExtVectorType *T) {
  Out << "Dv";
  mangleExpression(T->getSizeExpr());
  Out << '_';
  mangleType(T->getElementType());
}

void CXXNameMangler::mangleType(const ObjCInterfaceType *T) {
  mangleSourceName(T->getDecl()->getIdentifier());
}

void CXXNameMangler::mangleType(const ObjCObjectType *T) {
  // We don't allow overloading by different protocol qualification,
  // so mangling them isn't necessary.
  mangleType(T->getBaseType());
}

void CXXNameMangler::mangleType(const BlockPointerType *T) {
  Out << "U13block_pointer";
  mangleType(T->getPointeeType());
}

void CXXNameMangler::mangleType(const InjectedClassNameType *T) {
  // Mangle injected class name types as if the user had written the
  // specialization out fully.  It may not actually be possible to see
  // this mangling, though.
  mangleType(T->getInjectedSpecializationType());
}

void CXXNameMangler::mangleType(const TemplateSpecializationType *T) {
  if (TemplateDecl *TD = T->getTemplateName().getAsTemplateDecl()) {
    mangleName(TD, T->getArgs(), T->getNumArgs());
  } else {
    if (mangleSubstitution(QualType(T, 0)))
      return;
    
    mangleTemplatePrefix(T->getTemplateName());
    
    // FIXME: GCC does not appear to mangle the template arguments when
    // the template in question is a dependent template name. Should we
    // emulate that badness?
    mangleTemplateArgs(T->getTemplateName(), T->getArgs(), T->getNumArgs());
    addSubstitution(QualType(T, 0));
  }
}

void CXXNameMangler::mangleType(const DependentNameType *T) {
  // Typename types are always nested
  Out << 'N';
  mangleUnresolvedScope(T->getQualifier());
  mangleSourceName(T->getIdentifier());    
  Out << 'E';
}

void CXXNameMangler::mangleType(const DependentTemplateSpecializationType *T) {
  // Dependently-scoped template types are always nested
  Out << 'N';

  // TODO: avoid making this TemplateName.
  TemplateName Prefix =
    getASTContext().getDependentTemplateName(T->getQualifier(),
                                             T->getIdentifier());
  mangleTemplatePrefix(Prefix);

  // FIXME: GCC does not appear to mangle the template arguments when
  // the template in question is a dependent template name. Should we
  // emulate that badness?
  mangleTemplateArgs(Prefix, T->getArgs(), T->getNumArgs());    
  Out << 'E';
}

void CXXNameMangler::mangleType(const TypeOfType *T) {
  // FIXME: this is pretty unsatisfactory, but there isn't an obvious
  // "extension with parameters" mangling.
  Out << "u6typeof";
}

void CXXNameMangler::mangleType(const TypeOfExprType *T) {
  // FIXME: this is pretty unsatisfactory, but there isn't an obvious
  // "extension with parameters" mangling.
  Out << "u6typeof";
}

void CXXNameMangler::mangleType(const DecltypeType *T) {
  Expr *E = T->getUnderlyingExpr();

  // type ::= Dt <expression> E  # decltype of an id-expression
  //                             #   or class member access
  //      ::= DT <expression> E  # decltype of an expression

  // This purports to be an exhaustive list of id-expressions and
  // class member accesses.  Note that we do not ignore parentheses;
  // parentheses change the semantics of decltype for these
  // expressions (and cause the mangler to use the other form).
  if (isa<DeclRefExpr>(E) ||
      isa<MemberExpr>(E) ||
      isa<UnresolvedLookupExpr>(E) ||
      isa<DependentScopeDeclRefExpr>(E) ||
      isa<CXXDependentScopeMemberExpr>(E) ||
      isa<UnresolvedMemberExpr>(E))
    Out << "Dt";
  else
    Out << "DT";
  mangleExpression(E);
  Out << 'E';
}

void CXXNameMangler::mangleIntegerLiteral(QualType T,
                                          const llvm::APSInt &Value) {
  //  <expr-primary> ::= L <type> <value number> E # integer literal
  Out << 'L';

  mangleType(T);
  if (T->isBooleanType()) {
    // Boolean values are encoded as 0/1.
    Out << (Value.getBoolValue() ? '1' : '0');
  } else {
    mangleNumber(Value);
  }
  Out << 'E';

}

/// Mangles a member expression.  Implicit accesses are not handled,
/// but that should be okay, because you shouldn't be able to
/// make an implicit access in a function template declaration.
void CXXNameMangler::mangleMemberExpr(const Expr *Base,
                                      bool IsArrow,
                                      NestedNameSpecifier *Qualifier,
                                      DeclarationName Member,
                                      unsigned Arity) {
  // gcc-4.4 uses 'dt' for dot expressions, which is reasonable.
  // OTOH, gcc also mangles the name as an expression.
  Out << (IsArrow ? "pt" : "dt");
  mangleExpression(Base);
  mangleUnresolvedName(Qualifier, Member, Arity);
}

void CXXNameMangler::mangleExpression(const Expr *E, unsigned Arity) {
  // <expression> ::= <unary operator-name> <expression>
  //              ::= <binary operator-name> <expression> <expression>
  //              ::= <trinary operator-name> <expression> <expression> <expression>
  //              ::= cl <expression>* E             # call
  //              ::= cv <type> expression           # conversion with one argument
  //              ::= cv <type> _ <expression>* E # conversion with a different number of arguments
  //              ::= st <type>                      # sizeof (a type)
  //              ::= at <type>                      # alignof (a type)
  //              ::= <template-param>
  //              ::= <function-param>
  //              ::= sr <type> <unqualified-name>                   # dependent name
  //              ::= sr <type> <unqualified-name> <template-args>   # dependent template-id
  //              ::= sZ <template-param>                            # size of a parameter pack
  //              ::= <expr-primary>
  // <expr-primary> ::= L <type> <value number> E    # integer literal
  //                ::= L <type <value float> E      # floating literal
  //                ::= L <mangled-name> E           # external name
  switch (E->getStmtClass()) {
  case Expr::NoStmtClass:
#define EXPR(Type, Base)
#define STMT(Type, Base) \
  case Expr::Type##Class:
#include "clang/AST/StmtNodes.inc"
    // fallthrough

  // These all can only appear in local or variable-initialization
  // contexts and so should never appear in a mangling.
  case Expr::AddrLabelExprClass:
  case Expr::BlockDeclRefExprClass:
  case Expr::CXXThisExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::InitListExprClass:
  case Expr::ParenListExprClass:
  case Expr::CXXScalarValueInitExprClass:
    llvm_unreachable("unexpected statement kind");
    break;

  // FIXME: invent manglings for all these.
  case Expr::BlockExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::ChooseExprClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCImplicitSetterGetterRefExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::OffsetOfExprClass:
  case Expr::PredefinedExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::StmtExprClass:
  case Expr::TypesCompatibleExprClass:
  case Expr::UnaryTypeTraitExprClass:
  case Expr::VAArgExprClass:
  case Expr::CXXUuidofExprClass:
  case Expr::CXXNoexceptExprClass: {
    // As bad as this diagnostic is, it's better than crashing.
    Diagnostic &Diags = Context.getDiags();
    unsigned DiagID = Diags.getCustomDiagID(Diagnostic::Error,
                                     "cannot yet mangle expression type %0");
    Diags.Report(FullSourceLoc(E->getExprLoc(),
                               getASTContext().getSourceManager()),
                 DiagID)
      << E->getStmtClassName() << E->getSourceRange();
    break;
  }

  case Expr::CXXDefaultArgExprClass:
    mangleExpression(cast<CXXDefaultArgExpr>(E)->getExpr(), Arity);
    break;

  case Expr::CXXMemberCallExprClass: // fallthrough
  case Expr::CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    Out << "cl";
    mangleExpression(CE->getCallee(), CE->getNumArgs());
    for (unsigned I = 0, N = CE->getNumArgs(); I != N; ++I)
      mangleExpression(CE->getArg(I));
    Out << 'E';
    break;
  }

  case Expr::CXXNewExprClass: {
    // Proposal from David Vandervoorde, 2010.06.30
    const CXXNewExpr *New = cast<CXXNewExpr>(E);
    if (New->isGlobalNew()) Out << "gs";
    Out << (New->isArray() ? "na" : "nw");
    for (CXXNewExpr::const_arg_iterator I = New->placement_arg_begin(),
           E = New->placement_arg_end(); I != E; ++I)
      mangleExpression(*I);
    Out << '_';
    mangleType(New->getAllocatedType());
    if (New->hasInitializer()) {
      Out << "pi";
      for (CXXNewExpr::const_arg_iterator I = New->constructor_arg_begin(),
             E = New->constructor_arg_end(); I != E; ++I)
        mangleExpression(*I);
    }
    Out << 'E';
    break;
  }

  case Expr::MemberExprClass: {
    const MemberExpr *ME = cast<MemberExpr>(E);
    mangleMemberExpr(ME->getBase(), ME->isArrow(),
                     ME->getQualifier(), ME->getMemberDecl()->getDeclName(),
                     Arity);
    break;
  }

  case Expr::UnresolvedMemberExprClass: {
    const UnresolvedMemberExpr *ME = cast<UnresolvedMemberExpr>(E);
    mangleMemberExpr(ME->getBase(), ME->isArrow(),
                     ME->getQualifier(), ME->getMemberName(),
                     Arity);
    if (ME->hasExplicitTemplateArgs())
      mangleTemplateArgs(ME->getExplicitTemplateArgs());
    break;
  }

  case Expr::CXXDependentScopeMemberExprClass: {
    const CXXDependentScopeMemberExpr *ME
      = cast<CXXDependentScopeMemberExpr>(E);
    mangleMemberExpr(ME->getBase(), ME->isArrow(),
                     ME->getQualifier(), ME->getMember(),
                     Arity);
    if (ME->hasExplicitTemplateArgs())
      mangleTemplateArgs(ME->getExplicitTemplateArgs());
    break;
  }

  case Expr::UnresolvedLookupExprClass: {
    // The ABI doesn't cover how to mangle overload sets, so we mangle
    // using something as close as possible to the original lookup
    // expression.
    const UnresolvedLookupExpr *ULE = cast<UnresolvedLookupExpr>(E);
    mangleUnresolvedName(ULE->getQualifier(), ULE->getName(), Arity);
    if (ULE->hasExplicitTemplateArgs())
      mangleTemplateArgs(ULE->getExplicitTemplateArgs());
    break;
  }

  case Expr::CXXUnresolvedConstructExprClass: {
    const CXXUnresolvedConstructExpr *CE = cast<CXXUnresolvedConstructExpr>(E);
    unsigned N = CE->arg_size();

    Out << "cv";
    mangleType(CE->getType());
    if (N != 1) Out << '_';
    for (unsigned I = 0; I != N; ++I) mangleExpression(CE->getArg(I));
    if (N != 1) Out << 'E';
    break;
  }

  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXConstructExprClass: {
    const CXXConstructExpr *CE = cast<CXXConstructExpr>(E);
    unsigned N = CE->getNumArgs();

    Out << "cv";
    mangleType(CE->getType());
    if (N != 1) Out << '_';
    for (unsigned I = 0; I != N; ++I) mangleExpression(CE->getArg(I));
    if (N != 1) Out << 'E';
    break;
  }

  case Expr::SizeOfAlignOfExprClass: {
    const SizeOfAlignOfExpr *SAE = cast<SizeOfAlignOfExpr>(E);
    if (SAE->isSizeOf()) Out << 's';
    else Out << 'a';
    if (SAE->isArgumentType()) {
      Out << 't';
      mangleType(SAE->getArgumentType());
    } else {
      Out << 'z';
      mangleExpression(SAE->getArgumentExpr());
    }
    break;
  }

  case Expr::CXXThrowExprClass: {
    const CXXThrowExpr *TE = cast<CXXThrowExpr>(E);

    // Proposal from David Vandervoorde, 2010.06.30
    if (TE->getSubExpr()) {
      Out << "tw";
      mangleExpression(TE->getSubExpr());
    } else {
      Out << "tr";
    }
    break;
  }

  case Expr::CXXTypeidExprClass: {
    const CXXTypeidExpr *TIE = cast<CXXTypeidExpr>(E);

    // Proposal from David Vandervoorde, 2010.06.30
    if (TIE->isTypeOperand()) {
      Out << "ti";
      mangleType(TIE->getTypeOperand());
    } else {
      Out << "te";
      mangleExpression(TIE->getExprOperand());
    }
    break;
  }

  case Expr::CXXDeleteExprClass: {
    const CXXDeleteExpr *DE = cast<CXXDeleteExpr>(E);

    // Proposal from David Vandervoorde, 2010.06.30
    if (DE->isGlobalDelete()) Out << "gs";
    Out << (DE->isArrayForm() ? "da" : "dl");
    mangleExpression(DE->getArgument());
    break;
  }

  case Expr::UnaryOperatorClass: {
    const UnaryOperator *UO = cast<UnaryOperator>(E);
    mangleOperatorName(UnaryOperator::getOverloadedOperator(UO->getOpcode()),
                       /*Arity=*/1);
    mangleExpression(UO->getSubExpr());
    break;
  }

  case Expr::ArraySubscriptExprClass: {
    const ArraySubscriptExpr *AE = cast<ArraySubscriptExpr>(E);

    // Array subscript is treated as a syntactically wierd form of
    // binary operator.
    Out << "ix";
    mangleExpression(AE->getLHS());
    mangleExpression(AE->getRHS());
    break;
  }

  case Expr::CompoundAssignOperatorClass: // fallthrough
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *BO = cast<BinaryOperator>(E);
    mangleOperatorName(BinaryOperator::getOverloadedOperator(BO->getOpcode()),
                       /*Arity=*/2);
    mangleExpression(BO->getLHS());
    mangleExpression(BO->getRHS());
    break;
  }

  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *CO = cast<ConditionalOperator>(E);
    mangleOperatorName(OO_Conditional, /*Arity=*/3);
    mangleExpression(CO->getCond());
    mangleExpression(CO->getLHS(), Arity);
    mangleExpression(CO->getRHS(), Arity);
    break;
  }

  case Expr::ImplicitCastExprClass: {
    mangleExpression(cast<ImplicitCastExpr>(E)->getSubExpr(), Arity);
    break;
  }

  case Expr::CStyleCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::CXXFunctionalCastExprClass: {
    const ExplicitCastExpr *ECE = cast<ExplicitCastExpr>(E);
    Out << "cv";
    mangleType(ECE->getType());
    mangleExpression(ECE->getSubExpr());
    break;
  }

  case Expr::CXXOperatorCallExprClass: {
    const CXXOperatorCallExpr *CE = cast<CXXOperatorCallExpr>(E);
    unsigned NumArgs = CE->getNumArgs();
    mangleOperatorName(CE->getOperator(), /*Arity=*/NumArgs);
    // Mangle the arguments.
    for (unsigned i = 0; i != NumArgs; ++i)
      mangleExpression(CE->getArg(i));
    break;
  }

  case Expr::ParenExprClass:
    mangleExpression(cast<ParenExpr>(E)->getSubExpr(), Arity);
    break;

  case Expr::DeclRefExprClass: {
    const NamedDecl *D = cast<DeclRefExpr>(E)->getDecl();

    switch (D->getKind()) {
    default:
      //  <expr-primary> ::= L <mangled-name> E # external name
      Out << 'L';
      mangle(D, "_Z");
      Out << 'E';
      break;

    case Decl::EnumConstant: {
      const EnumConstantDecl *ED = cast<EnumConstantDecl>(D);
      mangleIntegerLiteral(ED->getType(), ED->getInitVal());
      break;
    }

    case Decl::NonTypeTemplateParm: {
      const NonTypeTemplateParmDecl *PD = cast<NonTypeTemplateParmDecl>(D);
      mangleTemplateParameter(PD->getIndex());
      break;
    }

    }

    break;
  }

  case Expr::DependentScopeDeclRefExprClass: {
    const DependentScopeDeclRefExpr *DRE = cast<DependentScopeDeclRefExpr>(E);
    NestedNameSpecifier *NNS = DRE->getQualifier();
    const Type *QTy = NNS->getAsType();

    // When we're dealing with a nested-name-specifier that has just a
    // dependent identifier in it, mangle that as a typename.  FIXME:
    // It isn't clear that we ever actually want to have such a
    // nested-name-specifier; why not just represent it as a typename type?
    if (!QTy && NNS->getAsIdentifier() && NNS->getPrefix()) {
      QTy = getASTContext().getDependentNameType(ETK_Typename,
                                                 NNS->getPrefix(),
                                                 NNS->getAsIdentifier())
              .getTypePtr();
    }
    assert(QTy && "Qualifier was not type!");

    // ::= sr <type> <unqualified-name>                  # dependent name
    // ::= sr <type> <unqualified-name> <template-args>  # dependent template-id
    Out << "sr";
    mangleType(QualType(QTy, 0));
    mangleUnqualifiedName(0, DRE->getDeclName(), Arity);
    if (DRE->hasExplicitTemplateArgs())
      mangleTemplateArgs(DRE->getExplicitTemplateArgs());

    break;
  }

  case Expr::CXXBindTemporaryExprClass:
    mangleExpression(cast<CXXBindTemporaryExpr>(E)->getSubExpr());
    break;

  case Expr::CXXExprWithTemporariesClass:
    mangleExpression(cast<CXXExprWithTemporaries>(E)->getSubExpr(), Arity);
    break;

  case Expr::FloatingLiteralClass: {
    const FloatingLiteral *FL = cast<FloatingLiteral>(E);
    Out << 'L';
    mangleType(FL->getType());
    mangleFloat(FL->getValue());
    Out << 'E';
    break;
  }

  case Expr::CharacterLiteralClass:
    Out << 'L';
    mangleType(E->getType());
    Out << cast<CharacterLiteral>(E)->getValue();
    Out << 'E';
    break;

  case Expr::CXXBoolLiteralExprClass:
    Out << "Lb";
    Out << (cast<CXXBoolLiteralExpr>(E)->getValue() ? '1' : '0');
    Out << 'E';
    break;

  case Expr::IntegerLiteralClass: {
    llvm::APSInt Value(cast<IntegerLiteral>(E)->getValue());
    if (E->getType()->isSignedIntegerType())
      Value.setIsSigned(true);
    mangleIntegerLiteral(E->getType(), Value);
    break;
  }

  case Expr::ImaginaryLiteralClass: {
    const ImaginaryLiteral *IE = cast<ImaginaryLiteral>(E);
    // Mangle as if a complex literal.
    // Proposal from David Vandevoorde, 2010.06.30.
    Out << 'L';
    mangleType(E->getType());
    if (const FloatingLiteral *Imag =
          dyn_cast<FloatingLiteral>(IE->getSubExpr())) {
      // Mangle a floating-point zero of the appropriate type.
      mangleFloat(llvm::APFloat(Imag->getValue().getSemantics()));
      Out << '_';
      mangleFloat(Imag->getValue());
    } else {
      Out << "0_";
      llvm::APSInt Value(cast<IntegerLiteral>(IE->getSubExpr())->getValue());
      if (IE->getSubExpr()->getType()->isSignedIntegerType())
        Value.setIsSigned(true);
      mangleNumber(Value);
    }
    Out << 'E';
    break;
  }

  case Expr::StringLiteralClass: {
    // Revised proposal from David Vandervoorde, 2010.07.15.
    Out << 'L';
    assert(isa<ConstantArrayType>(E->getType()));
    mangleType(E->getType());
    Out << 'E';
    break;
  }

  case Expr::GNUNullExprClass:
    // FIXME: should this really be mangled the same as nullptr?
    // fallthrough

  case Expr::CXXNullPtrLiteralExprClass: {
    // Proposal from David Vandervoorde, 2010.06.30, as
    // modified by ABI list discussion.
    Out << "LDnE";
    break;
  }

  }
}

void CXXNameMangler::mangleCXXCtorType(CXXCtorType T) {
  // <ctor-dtor-name> ::= C1  # complete object constructor
  //                  ::= C2  # base object constructor
  //                  ::= C3  # complete object allocating constructor
  //
  switch (T) {
  case Ctor_Complete:
    Out << "C1";
    break;
  case Ctor_Base:
    Out << "C2";
    break;
  case Ctor_CompleteAllocating:
    Out << "C3";
    break;
  }
}

void CXXNameMangler::mangleCXXDtorType(CXXDtorType T) {
  // <ctor-dtor-name> ::= D0  # deleting destructor
  //                  ::= D1  # complete object destructor
  //                  ::= D2  # base object destructor
  //
  switch (T) {
  case Dtor_Deleting:
    Out << "D0";
    break;
  case Dtor_Complete:
    Out << "D1";
    break;
  case Dtor_Base:
    Out << "D2";
    break;
  }
}

void CXXNameMangler::mangleTemplateArgs(
                          const ExplicitTemplateArgumentList &TemplateArgs) {
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned I = 0, E = TemplateArgs.NumTemplateArgs; I != E; ++I)
    mangleTemplateArg(0, TemplateArgs.getTemplateArgs()[I].getArgument());
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArgs(TemplateName Template,
                                        const TemplateArgument *TemplateArgs,
                                        unsigned NumTemplateArgs) {
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleTemplateArgs(*TD->getTemplateParameters(), TemplateArgs,
                              NumTemplateArgs);
  
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    mangleTemplateArg(0, TemplateArgs[i]);
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArgs(const TemplateParameterList &PL,
                                        const TemplateArgumentList &AL) {
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned i = 0, e = AL.size(); i != e; ++i)
    mangleTemplateArg(PL.getParam(i), AL[i]);
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArgs(const TemplateParameterList &PL,
                                        const TemplateArgument *TemplateArgs,
                                        unsigned NumTemplateArgs) {
  // <template-args> ::= I <template-arg>+ E
  Out << 'I';
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    mangleTemplateArg(PL.getParam(i), TemplateArgs[i]);
  Out << 'E';
}

void CXXNameMangler::mangleTemplateArg(const NamedDecl *P,
                                       const TemplateArgument &A) {
  // <template-arg> ::= <type>              # type or template
  //                ::= X <expression> E    # expression
  //                ::= <expr-primary>      # simple expressions
  //                ::= I <template-arg>* E # argument pack
  //                ::= sp <expression>     # pack expansion of (C++0x)
  switch (A.getKind()) {
  default:
    assert(0 && "Unknown template argument kind!");
  case TemplateArgument::Type:
    mangleType(A.getAsType());
    break;
  case TemplateArgument::Template:
    // This is mangled as <type>.
    mangleType(A.getAsTemplate());
    break;
  case TemplateArgument::Expression:
    Out << 'X';
    mangleExpression(A.getAsExpr());
    Out << 'E';
    break;
  case TemplateArgument::Integral:
    mangleIntegerLiteral(A.getIntegralType(), *A.getAsIntegral());
    break;
  case TemplateArgument::Declaration: {
    assert(P && "Missing template parameter for declaration argument");
    //  <expr-primary> ::= L <mangled-name> E # external name

    // Clang produces AST's where pointer-to-member-function expressions
    // and pointer-to-function expressions are represented as a declaration not
    // an expression. We compensate for it here to produce the correct mangling.
    NamedDecl *D = cast<NamedDecl>(A.getAsDecl());
    const NonTypeTemplateParmDecl *Parameter = cast<NonTypeTemplateParmDecl>(P);
    bool compensateMangling = D->isCXXClassMember() &&
      !Parameter->getType()->isReferenceType();
    if (compensateMangling) {
      Out << 'X';
      mangleOperatorName(OO_Amp, 1);
    }

    Out << 'L';
    // References to external entities use the mangled name; if the name would
    // not normally be manged then mangle it as unqualified.
    //
    // FIXME: The ABI specifies that external names here should have _Z, but
    // gcc leaves this off.
    if (compensateMangling)
      mangle(D, "_Z");
    else
      mangle(D, "Z");
    Out << 'E';

    if (compensateMangling)
      Out << 'E';

    break;
  }
  }
}

void CXXNameMangler::mangleTemplateParameter(unsigned Index) {
  // <template-param> ::= T_    # first template parameter
  //                  ::= T <parameter-2 non-negative number> _
  if (Index == 0)
    Out << "T_";
  else
    Out << 'T' << (Index - 1) << '_';
}

// <substitution> ::= S <seq-id> _
//                ::= S_
bool CXXNameMangler::mangleSubstitution(const NamedDecl *ND) {
  // Try one of the standard substitutions first.
  if (mangleStandardSubstitution(ND))
    return true;

  ND = cast<NamedDecl>(ND->getCanonicalDecl());
  return mangleSubstitution(reinterpret_cast<uintptr_t>(ND));
}

bool CXXNameMangler::mangleSubstitution(QualType T) {
  if (!T.getCVRQualifiers()) {
    if (const RecordType *RT = T->getAs<RecordType>())
      return mangleSubstitution(RT->getDecl());
  }

  uintptr_t TypePtr = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());

  return mangleSubstitution(TypePtr);
}

bool CXXNameMangler::mangleSubstitution(TemplateName Template) {
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return mangleSubstitution(TD);
  
  Template = Context.getASTContext().getCanonicalTemplateName(Template);
  return mangleSubstitution(
                      reinterpret_cast<uintptr_t>(Template.getAsVoidPointer()));
}

bool CXXNameMangler::mangleSubstitution(uintptr_t Ptr) {
  llvm::DenseMap<uintptr_t, unsigned>::iterator I = Substitutions.find(Ptr);
  if (I == Substitutions.end())
    return false;

  unsigned SeqID = I->second;
  if (SeqID == 0)
    Out << "S_";
  else {
    SeqID--;

    // <seq-id> is encoded in base-36, using digits and upper case letters.
    char Buffer[10];
    char *BufferPtr = llvm::array_endof(Buffer);

    if (SeqID == 0) *--BufferPtr = '0';

    while (SeqID) {
      assert(BufferPtr > Buffer && "Buffer overflow!");

      char c = static_cast<char>(SeqID % 36);

      *--BufferPtr =  (c < 10 ? '0' + c : 'A' + c - 10);
      SeqID /= 36;
    }

    Out << 'S'
        << llvm::StringRef(BufferPtr, llvm::array_endof(Buffer)-BufferPtr)
        << '_';
  }

  return true;
}

static bool isCharType(QualType T) {
  if (T.isNull())
    return false;

  return T->isSpecificBuiltinType(BuiltinType::Char_S) ||
    T->isSpecificBuiltinType(BuiltinType::Char_U);
}

/// isCharSpecialization - Returns whether a given type is a template
/// specialization of a given name with a single argument of type char.
static bool isCharSpecialization(QualType T, const char *Name) {
  if (T.isNull())
    return false;

  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  const ClassTemplateSpecializationDecl *SD =
    dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl());
  if (!SD)
    return false;

  if (!isStdNamespace(SD->getDeclContext()))
    return false;

  const TemplateArgumentList &TemplateArgs = SD->getTemplateArgs();
  if (TemplateArgs.size() != 1)
    return false;

  if (!isCharType(TemplateArgs[0].getAsType()))
    return false;

  return SD->getIdentifier()->getName() == Name;
}

template <std::size_t StrLen>
bool isStreamCharSpecialization(const ClassTemplateSpecializationDecl *SD,
                                const char (&Str)[StrLen]) {
  if (!SD->getIdentifier()->isStr(Str))
    return false;

  const TemplateArgumentList &TemplateArgs = SD->getTemplateArgs();
  if (TemplateArgs.size() != 2)
    return false;

  if (!isCharType(TemplateArgs[0].getAsType()))
    return false;

  if (!isCharSpecialization(TemplateArgs[1].getAsType(), "char_traits"))
    return false;

  return true;
}

bool CXXNameMangler::mangleStandardSubstitution(const NamedDecl *ND) {
  // <substitution> ::= St # ::std::
  if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
    if (isStd(NS)) {
      Out << "St";
      return true;
    }
  }

  if (const ClassTemplateDecl *TD = dyn_cast<ClassTemplateDecl>(ND)) {
    if (!isStdNamespace(TD->getDeclContext()))
      return false;

    // <substitution> ::= Sa # ::std::allocator
    if (TD->getIdentifier()->isStr("allocator")) {
      Out << "Sa";
      return true;
    }

    // <<substitution> ::= Sb # ::std::basic_string
    if (TD->getIdentifier()->isStr("basic_string")) {
      Out << "Sb";
      return true;
    }
  }

  if (const ClassTemplateSpecializationDecl *SD =
        dyn_cast<ClassTemplateSpecializationDecl>(ND)) {
    if (!isStdNamespace(SD->getDeclContext()))
      return false;

    //    <substitution> ::= Ss # ::std::basic_string<char,
    //                            ::std::char_traits<char>,
    //                            ::std::allocator<char> >
    if (SD->getIdentifier()->isStr("basic_string")) {
      const TemplateArgumentList &TemplateArgs = SD->getTemplateArgs();

      if (TemplateArgs.size() != 3)
        return false;

      if (!isCharType(TemplateArgs[0].getAsType()))
        return false;

      if (!isCharSpecialization(TemplateArgs[1].getAsType(), "char_traits"))
        return false;

      if (!isCharSpecialization(TemplateArgs[2].getAsType(), "allocator"))
        return false;

      Out << "Ss";
      return true;
    }

    //    <substitution> ::= Si # ::std::basic_istream<char,
    //                            ::std::char_traits<char> >
    if (isStreamCharSpecialization(SD, "basic_istream")) {
      Out << "Si";
      return true;
    }

    //    <substitution> ::= So # ::std::basic_ostream<char,
    //                            ::std::char_traits<char> >
    if (isStreamCharSpecialization(SD, "basic_ostream")) {
      Out << "So";
      return true;
    }

    //    <substitution> ::= Sd # ::std::basic_iostream<char,
    //                            ::std::char_traits<char> >
    if (isStreamCharSpecialization(SD, "basic_iostream")) {
      Out << "Sd";
      return true;
    }
  }
  return false;
}

void CXXNameMangler::addSubstitution(QualType T) {
  if (!T.getCVRQualifiers()) {
    if (const RecordType *RT = T->getAs<RecordType>()) {
      addSubstitution(RT->getDecl());
      return;
    }
  }

  uintptr_t TypePtr = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
  addSubstitution(TypePtr);
}

void CXXNameMangler::addSubstitution(TemplateName Template) {
  if (TemplateDecl *TD = Template.getAsTemplateDecl())
    return addSubstitution(TD);
  
  Template = Context.getASTContext().getCanonicalTemplateName(Template);
  addSubstitution(reinterpret_cast<uintptr_t>(Template.getAsVoidPointer()));
}

void CXXNameMangler::addSubstitution(uintptr_t Ptr) {
  assert(!Substitutions.count(Ptr) && "Substitution already exists!");
  Substitutions[Ptr] = SeqID++;
}

//

/// \brief Mangles the name of the declaration D and emits that name to the
/// given output stream.
///
/// If the declaration D requires a mangled name, this routine will emit that
/// mangled name to \p os and return true. Otherwise, \p os will be unchanged
/// and this routine will return false. In this case, the caller should just
/// emit the identifier of the declaration (\c D->getIdentifier()) as its
/// name.
void MangleContext::mangleName(const NamedDecl *D,
                               llvm::SmallVectorImpl<char> &Res) {
  assert((isa<FunctionDecl>(D) || isa<VarDecl>(D)) &&
          "Invalid mangleName() call, argument is not a variable or function!");
  assert(!isa<CXXConstructorDecl>(D) && !isa<CXXDestructorDecl>(D) &&
         "Invalid mangleName() call on 'structor decl!");

  PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                 getASTContext().getSourceManager(),
                                 "Mangling declaration");

  CXXNameMangler Mangler(*this, Res);
  return Mangler.mangle(D);
}

void MangleContext::mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                                  llvm::SmallVectorImpl<char> &Res) {
  CXXNameMangler Mangler(*this, Res, D, Type);
  Mangler.mangle(D);
}

void MangleContext::mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                                  llvm::SmallVectorImpl<char> &Res) {
  CXXNameMangler Mangler(*this, Res, D, Type);
  Mangler.mangle(D);
}

void MangleContext::mangleBlock(GlobalDecl GD, const BlockDecl *BD,
                                llvm::SmallVectorImpl<char> &Res) {
  MiscNameMangler Mangler(*this, Res);
  Mangler.mangleBlock(GD, BD);
}

void MangleContext::mangleThunk(const CXXMethodDecl *MD,
                                const ThunkInfo &Thunk,
                                llvm::SmallVectorImpl<char> &Res) {
  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  //  <special-name> ::= Tc <call-offset> <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  //                      # first call-offset is 'this' adjustment
  //                      # second call-offset is result adjustment
  
  assert(!isa<CXXDestructorDecl>(MD) &&
         "Use mangleCXXDtor for destructor decls!");
  
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZT";
  if (!Thunk.Return.isEmpty())
    Mangler.getStream() << 'c';
  
  // Mangle the 'this' pointer adjustment.
  Mangler.mangleCallOffset(Thunk.This.NonVirtual, Thunk.This.VCallOffsetOffset);
  
  // Mangle the return pointer adjustment if there is one.
  if (!Thunk.Return.isEmpty())
    Mangler.mangleCallOffset(Thunk.Return.NonVirtual,
                             Thunk.Return.VBaseOffsetOffset);
  
  Mangler.mangleFunctionEncoding(MD);
}

void 
MangleContext::mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                                  const ThisAdjustment &ThisAdjustment,
                                  llvm::SmallVectorImpl<char> &Res) {
  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  
  CXXNameMangler Mangler(*this, Res, DD, Type);
  Mangler.getStream() << "_ZT";

  // Mangle the 'this' pointer adjustment.
  Mangler.mangleCallOffset(ThisAdjustment.NonVirtual, 
                           ThisAdjustment.VCallOffsetOffset);

  Mangler.mangleFunctionEncoding(DD);
}

/// mangleGuardVariable - Returns the mangled name for a guard variable
/// for the passed in VarDecl.
void MangleContext::mangleItaniumGuardVariable(const VarDecl *D,
                                         llvm::SmallVectorImpl<char> &Res) {
  //  <special-name> ::= GV <object name>       # Guard variable for one-time
  //                                            # initialization
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZGV";
  Mangler.mangleName(D);
}

void MangleContext::mangleReferenceTemporary(const VarDecl *D,
                                             llvm::SmallVectorImpl<char> &Res) {
  // We match the GCC mangling here.
  //  <special-name> ::= GR <object name>
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZGR";
  Mangler.mangleName(D);
}

void MangleContext::mangleCXXVTable(const CXXRecordDecl *RD,
                                    llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TV <type>  # virtual table
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTV";
  Mangler.mangleNameOrStandardSubstitution(RD);
}

void MangleContext::mangleCXXVTT(const CXXRecordDecl *RD,
                                 llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TT <type>  # VTT structure
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTT";
  Mangler.mangleNameOrStandardSubstitution(RD);
}

void MangleContext::mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                                        const CXXRecordDecl *Type,
                                        llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TC <type> <offset number> _ <base type>
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTC";
  Mangler.mangleNameOrStandardSubstitution(RD);
  Mangler.getStream() << Offset;
  Mangler.getStream() << '_';
  Mangler.mangleNameOrStandardSubstitution(Type);
}

void MangleContext::mangleCXXRTTI(QualType Ty,
                                  llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TI <type>  # typeinfo structure
  assert(!Ty.hasQualifiers() && "RTTI info cannot have top-level qualifiers");
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTI";
  Mangler.mangleType(Ty);
}

void MangleContext::mangleCXXRTTIName(QualType Ty,
                                      llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TS <type>  # typeinfo name (null terminated byte string)
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTS";
  Mangler.mangleType(Ty);
}
