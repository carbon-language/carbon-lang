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
#include "CGVtable.h"
using namespace clang;
using namespace CodeGen;

namespace {
  
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
  
/// CXXNameMangler - Manage the mangling of a single name.
class CXXNameMangler {
  MangleContext &Context;
  llvm::raw_svector_ostream Out;

  const CXXMethodDecl *Structor;
  unsigned StructorType;

  llvm::DenseMap<uintptr_t, unsigned> Substitutions;

public:
  CXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res)
    : Context(C), Out(Res), Structor(0), StructorType(0) { }
  CXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res,
                 const CXXConstructorDecl *D, CXXCtorType Type)
    : Context(C), Out(Res), Structor(getStructor(D)), StructorType(Type) { }
  CXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res,
                 const CXXDestructorDecl *D, CXXDtorType Type)
    : Context(C), Out(Res), Structor(getStructor(D)), StructorType(Type) { }

  llvm::raw_svector_ostream &getStream() { return Out; }

  void mangle(const NamedDecl *D, llvm::StringRef Prefix = "_Z");
  void mangleCallOffset(const ThunkAdjustment &Adjustment);
  void mangleNumber(int64_t Number);
  void mangleFunctionEncoding(const FunctionDecl *FD);
  void mangleName(const NamedDecl *ND);
  void mangleType(QualType T);

private:
  bool mangleSubstitution(const NamedDecl *ND);
  bool mangleSubstitution(QualType T);
  bool mangleSubstitution(uintptr_t Ptr);

  bool mangleStandardSubstitution(const NamedDecl *ND);

  void addSubstitution(const NamedDecl *ND) {
    ND = cast<NamedDecl>(ND->getCanonicalDecl());

    addSubstitution(reinterpret_cast<uintptr_t>(ND));
  }
  void addSubstitution(QualType T);
  void addSubstitution(uintptr_t Ptr);

  void mangleName(const TemplateDecl *TD,
                  const TemplateArgument *TemplateArgs,
                  unsigned NumTemplateArgs);
  void mangleUnqualifiedName(const NamedDecl *ND);
  void mangleUnscopedName(const NamedDecl *ND);
  void mangleUnscopedTemplateName(const TemplateDecl *ND);
  void mangleSourceName(const IdentifierInfo *II);
  void mangleLocalName(const NamedDecl *ND);
  void mangleNestedName(const NamedDecl *ND, const DeclContext *DC);
  void mangleNestedName(const TemplateDecl *TD,
                        const TemplateArgument *TemplateArgs,
                        unsigned NumTemplateArgs);
  void manglePrefix(const DeclContext *DC);
  void mangleTemplatePrefix(const TemplateDecl *ND);
  void mangleOperatorName(OverloadedOperatorKind OO, unsigned Arity);
  void mangleQualifiers(Qualifiers Quals);

  void mangleObjCMethodName(const ObjCMethodDecl *MD);
  
  // Declare manglers for every type class.
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) void mangleType(const CLASS##Type *T);
#include "clang/AST/TypeNodes.def"

  void mangleType(const TagType*);
  void mangleBareFunctionType(const FunctionType *T,
                              bool MangleReturnType);

  void mangleIntegerLiteral(QualType T, const llvm::APSInt &Value);
  void mangleExpression(const Expr *E);
  void mangleCXXCtorType(CXXCtorType T);
  void mangleCXXDtorType(CXXDtorType T);

  void mangleTemplateArgs(const TemplateArgument *TemplateArgs,
                          unsigned NumTemplateArgs);
  void mangleTemplateArgumentList(const TemplateArgumentList &L);
  void mangleTemplateArgument(const TemplateArgument &A);

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

  // No mangling in an "implicit extern C" header.
  if (D->getLocation().isValid() &&
      getASTContext().getSourceManager().
      isInExternCSystemHeader(D->getLocation()))
    return false;

  // Variables at global scope are not mangled.
  if (!FD) {
    const DeclContext *DC = D->getDeclContext();
    // Check for extern variable declared locally.
    if (isa<FunctionDecl>(DC) && D->hasLinkage())
      while (!DC->isNamespace() && !DC->isTranslationUnit())
        DC = DC->getParent();
    if (DC->isTranslationUnit())
      return false;
  }

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
  else
    mangleName(cast<VarDecl>(D));
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

/// isStd - Return whether a given namespace is the 'std' namespace.
static bool isStd(const NamespaceDecl *NS) {
  const IdentifierInfo *II = NS->getOriginalNamespace()->getIdentifier();
  return II && II->isStr("std");
}

static const DeclContext *IgnoreLinkageSpecDecls(const DeclContext *DC) {
  while (isa<LinkageSpecDecl>(DC)) {
    assert(cast<LinkageSpecDecl>(DC)->getLanguage() ==
           LinkageSpecDecl::lang_cxx && "Unexpected linkage decl!");
    DC = DC->getParent();
  }
  
  return DC;
}

// isStdNamespace - Return whether a given decl context is a toplevel 'std'
// namespace.
static bool isStdNamespace(const DeclContext *DC) {
  if (!DC->isNamespace())
    return false;
  
  if (!IgnoreLinkageSpecDecls(DC->getParent())->isTranslationUnit())
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

  while (isa<LinkageSpecDecl>(DC))
    DC = DC->getParent();

  if (DC->isTranslationUnit() || isStdNamespace(DC)) {
    // Check if we have a template.
    const TemplateArgumentList *TemplateArgs = 0;
    if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
      mangleUnscopedTemplateName(TD);
      mangleTemplateArgumentList(*TemplateArgs);
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
    mangleTemplateArgs(TemplateArgs, NumTemplateArgs);
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

  mangleUnscopedName(ND->getTemplatedDecl());
  addSubstitution(ND);
}

void CXXNameMangler::mangleNumber(int64_t Number) {
  //  <number> ::= [n] <non-negative decimal integer>
  if (Number < 0) {
    Out << 'n';
    Number = -Number;
  }
  
  Out << Number;
}

void CXXNameMangler::mangleCallOffset(const ThunkAdjustment &Adjustment) {
  //  <call-offset>  ::= h <nv-offset> _
  //                 ::= v <v-offset> _
  //  <nv-offset>    ::= <offset number>        # non-virtual base override
  //  <v-offset>     ::= <offset number> _ <virtual offset number>
  //                      # virtual base override, with vcall offset
  if (!Adjustment.Virtual) {
    Out << 'h';
    mangleNumber(Adjustment.NonVirtual);
    Out << '_';
    return;
  }
  
  Out << 'v';
  mangleNumber(Adjustment.NonVirtual);
  Out << '_';
  mangleNumber(Adjustment.Virtual);
  Out << '_';
}

void CXXNameMangler::mangleUnqualifiedName(const NamedDecl *ND) {
  //  <unqualified-name> ::= <operator-name>
  //                     ::= <ctor-dtor-name>
  //                     ::= <source-name>
  DeclarationName Name = ND->getDeclName();
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier: {
    if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
      if (NS->isAnonymousNamespace()) {
        // This is how gcc mangles these names.  It's apparently
        // always '1', no matter how many different anonymous
        // namespaces appear in a context.
        Out << "12_GLOBAL__N_1";
        break;
      }
    }

    if (const IdentifierInfo *II = Name.getAsIdentifierInfo()) {
      mangleSourceName(II);
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

  case DeclarationName::CXXOperatorName:
    mangleOperatorName(Name.getCXXOverloadedOperator(),
                       cast<FunctionDecl>(ND)->getNumParams());
    break;

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
                                      const DeclContext *DC) {
  // <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
  //               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E

  Out << 'N';
  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(ND))
    mangleQualifiers(Qualifiers::fromCVRMask(Method->getTypeQualifiers()));

  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = 0;
  if (const TemplateDecl *TD = isTemplate(ND, TemplateArgs)) {
    mangleTemplatePrefix(TD);
    mangleTemplateArgumentList(*TemplateArgs);
  } else {
    manglePrefix(DC);
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
  mangleTemplateArgs(TemplateArgs, NumTemplateArgs);

  Out << 'E';
}

void CXXNameMangler::mangleLocalName(const NamedDecl *ND) {
  // <local-name> := Z <function encoding> E <entity name> [<discriminator>]
  //              := Z <function encoding> E s [<discriminator>]
  // <discriminator> := _ <non-negative number>
  Out << 'Z';
  
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(ND->getDeclContext()))
    mangleObjCMethodName(MD);
  else  
    mangleFunctionEncoding(cast<FunctionDecl>(ND->getDeclContext()));

  Out << 'E';
  mangleUnqualifiedName(ND);
}

void CXXNameMangler::manglePrefix(const DeclContext *DC) {
  //  <prefix> ::= <prefix> <unqualified-name>
  //           ::= <template-prefix> <template-args>
  //           ::= <template-param>
  //           ::= # empty
  //           ::= <substitution>

  while (isa<LinkageSpecDecl>(DC))
    DC = DC->getParent();

  if (DC->isTranslationUnit())
    return;

  if (mangleSubstitution(cast<NamedDecl>(DC)))
    return;

  // Check if we have a template.
  const TemplateArgumentList *TemplateArgs = 0;
  if (const TemplateDecl *TD = isTemplate(cast<NamedDecl>(DC), TemplateArgs)) {
    mangleTemplatePrefix(TD);
    mangleTemplateArgumentList(*TemplateArgs);
  } else {
    manglePrefix(DC->getParent());
    mangleUnqualifiedName(cast<NamedDecl>(DC));
  }

  addSubstitution(cast<NamedDecl>(DC));
}

void CXXNameMangler::mangleTemplatePrefix(const TemplateDecl *ND) {
  // <template-prefix> ::= <prefix> <template unqualified-name>
  //                   ::= <template-param>
  //                   ::= <substitution>

  if (mangleSubstitution(ND))
    return;

  // FIXME: <template-param>

  manglePrefix(ND->getDeclContext());
  mangleUnqualifiedName(ND->getTemplatedDecl());

  addSubstitution(ND);
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
  //              ::= pl        # +
  case OO_Plus: Out << (Arity == 1? "ps" : "pl"); break;
  //              ::= ng        # - (unary)
  //              ::= mi        # -
  case OO_Minus: Out << (Arity == 1? "ng" : "mi"); break;
  //              ::= ad        # & (unary)
  //              ::= an        # &
  case OO_Amp: Out << (Arity == 1? "ad" : "an"); break;
  //              ::= de        # * (unary)
  //              ::= ml        # *
  case OO_Star: Out << (Arity == 1? "de" : "ml"); break;
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

  // FIXME: For now, just drop all extension qualifiers on the floor.
}

void CXXNameMangler::mangleObjCMethodName(const ObjCMethodDecl *MD) {
  llvm::SmallString<64> Name;
  llvm::raw_svector_ostream OS(Name);
  
  const ObjCContainerDecl *CD = 
    dyn_cast<ObjCContainerDecl>(MD->getDeclContext());
  assert (CD && "Missing container decl in GetNameForMethod");
  OS << (MD->isInstanceMethod() ? '-' : '+') << '[' << CD->getName();
  if (const ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(CD))
    OS << '(' << CID->getNameAsString() << ')';
  OS << ' ' << MD->getSelector().getAsString() << ']';
  
  Out << OS.str().size() << OS.str();
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
  //                 ::= u <source-name>    # vendor extended type
  // From our point of view, std::nullptr_t is a builtin, but as far as mangling
  // is concerned, it's a type called std::nullptr_t.
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
  case BuiltinType::NullPtr: Out << "St9nullptr_t"; break;

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

  if (Proto->getNumArgs() == 0) {
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
  Out << 'A' << '_';
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
  } else
    mangleType(PointeeType);
}

// <type>           ::= <template-param>
void CXXNameMangler::mangleType(const TemplateTypeParmType *T) {
  mangleTemplateParameter(T->getIndex());
}

// FIXME: <type> ::= <template-template-param> <template-args>

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

// GNU extension: vector types
void CXXNameMangler::mangleType(const VectorType *T) {
  Out << "U8__vector";
  mangleType(T->getElementType());
}
void CXXNameMangler::mangleType(const ExtVectorType *T) {
  mangleType(static_cast<const VectorType*>(T));
}
void CXXNameMangler::mangleType(const DependentSizedExtVectorType *T) {
  Out << "U8__vector";
  mangleType(T->getElementType());
}

void CXXNameMangler::mangleType(const ObjCInterfaceType *T) {
  mangleSourceName(T->getDecl()->getIdentifier());
}

void CXXNameMangler::mangleType(const BlockPointerType *T) {
  assert(false && "can't mangle block pointer types yet");
}

void CXXNameMangler::mangleType(const FixedWidthIntType *T) {
  assert(false && "can't mangle arbitary-precision integer type yet");
}

void CXXNameMangler::mangleType(const TemplateSpecializationType *T) {
  TemplateDecl *TD = T->getTemplateName().getAsTemplateDecl();
  assert(TD && "FIXME: Support dependent template names!");

  mangleName(TD, T->getArgs(), T->getNumArgs());
}

void CXXNameMangler::mangleType(const TypenameType *T) {
  // Typename types are always nested
  Out << 'N';

  const Type *QTy = T->getQualifier()->getAsType();
  if (const TemplateSpecializationType *TST =
        dyn_cast<TemplateSpecializationType>(QTy)) {
    if (!mangleSubstitution(QualType(TST, 0))) {
      TemplateDecl *TD = TST->getTemplateName().getAsTemplateDecl();

      mangleTemplatePrefix(TD);
      mangleTemplateArgs(TST->getArgs(), TST->getNumArgs());
      addSubstitution(QualType(TST, 0));
    }
  } else if (const TemplateTypeParmType *TTPT =
              dyn_cast<TemplateTypeParmType>(QTy)) {
    // We use the QualType mangle type variant here because it handles
    // substitutions.
    mangleType(QualType(TTPT, 0));
  } else
    assert(false && "Unhandled type!");

  mangleSourceName(T->getIdentifier());

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
    if (Value.isNegative())
      Out << 'n';
    Value.abs().print(Out, false);
  }
  Out << 'E';
  
}

void CXXNameMangler::mangleExpression(const Expr *E) {
  // <expression> ::= <unary operator-name> <expression>
	//              ::= <binary operator-name> <expression> <expression>
	//              ::= <trinary operator-name> <expression> <expression> <expression>
  //              ::= cl <expression>* E	        # call
  //              ::= cv <type> expression           # conversion with one argument
  //              ::= cv <type> _ <expression>* E # conversion with a different number of arguments
  //              ::= st <type>		        # sizeof (a type)
  //              ::= at <type>                      # alignof (a type)
  //              ::= <template-param>
  //              ::= <function-param>
  //              ::= sr <type> <unqualified-name>                   # dependent name
  //              ::= sr <type> <unqualified-name> <template-args>   # dependent template-id
  //              ::= sZ <template-param>                            # size of a parameter pack
	//              ::= <expr-primary>
  switch (E->getStmtClass()) {
  default: assert(false && "Unhandled expression kind!");

  case Expr::UnaryOperatorClass: {
    const UnaryOperator *UO = cast<UnaryOperator>(E);
    mangleOperatorName(UnaryOperator::getOverloadedOperator(UO->getOpcode()), 
                       /*Arity=*/1);
    mangleExpression(UO->getSubExpr());
    break;
  }
      
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
    mangleExpression(CO->getLHS());
    mangleExpression(CO->getRHS());
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
    mangleExpression(cast<ParenExpr>(E)->getSubExpr());
    break;

  case Expr::DeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(E)->getDecl();

    switch (D->getKind()) {
    default: assert(false && "Unhandled decl kind!");
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
    const Type *QTy = DRE->getQualifier()->getAsType();
    assert(QTy && "Qualifier was not type!");

    // ::= sr <type> <unqualified-name>                   # dependent name
    Out << "sr";
    mangleType(QualType(QTy, 0));

    assert(DRE->getDeclName().getNameKind() == DeclarationName::Identifier &&
           "Unhandled decl name kind!");
    mangleSourceName(DRE->getDeclName().getAsIdentifierInfo());

    break;
  }

  case Expr::IntegerLiteralClass:
    mangleIntegerLiteral(E->getType(), 
                         llvm::APSInt(cast<IntegerLiteral>(E)->getValue()));
    break;

  }
}

// FIXME: <type> ::= G <type>   # imaginary (C 2000)
// FIXME: <type> ::= U <source-name> <type>     # vendor extended type qualifier

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

void CXXNameMangler::mangleTemplateArgumentList(const TemplateArgumentList &L) {
  // <template-args> ::= I <template-arg>+ E
  Out << "I";
  for (unsigned i = 0, e = L.size(); i != e; ++i)
    mangleTemplateArgument(L[i]);
  Out << "E";
}

void CXXNameMangler::mangleTemplateArgs(const TemplateArgument *TemplateArgs,
                                        unsigned NumTemplateArgs) {
  // <template-args> ::= I <template-arg>+ E
  Out << "I";
  for (unsigned i = 0; i != NumTemplateArgs; ++i)
    mangleTemplateArgument(TemplateArgs[i]);
  Out << "E";
}

void CXXNameMangler::mangleTemplateArgument(const TemplateArgument &A) {
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
  case TemplateArgument::Expression:
    Out << 'X';
    mangleExpression(A.getAsExpr());
    Out << 'E';
    break;
  case TemplateArgument::Integral:
    mangleIntegerLiteral(A.getIntegralType(), *A.getAsIntegral());
    break;
  case TemplateArgument::Declaration: {
    //  <expr-primary> ::= L <mangled-name> E # external name

    // FIXME: Clang produces AST's where pointer-to-member-function expressions
    // and pointer-to-function expressions are represented as a declaration not
    // an expression; this is not how gcc represents them and this changes the
    // mangling.
    Out << 'L';
    // References to external entities use the mangled name; if the name would
    // not normally be manged then mangle it as unqualified.
    //
    // FIXME: The ABI specifies that external names here should have _Z, but
    // gcc leaves this off.
    mangle(cast<NamedDecl>(A.getAsDecl()), "Z");
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

bool CXXNameMangler::mangleSubstitution(uintptr_t Ptr) {
  llvm::DenseMap<uintptr_t, unsigned>::iterator I =
    Substitutions.find(Ptr);
  if (I == Substitutions.end())
    return false;

  unsigned SeqID = I->second;
  if (SeqID == 0)
    Out << "S_";
  else {
    SeqID--;

    // <seq-id> is encoded in base-36, using digits and upper case letters.
    char Buffer[10];
    char *BufferPtr = Buffer + 9;

    *BufferPtr = 0;
    if (SeqID == 0) *--BufferPtr = '0';

    while (SeqID) {
      assert(BufferPtr > Buffer && "Buffer overflow!");

      unsigned char c = static_cast<unsigned char>(SeqID) % 36;

      *--BufferPtr =  (c < 10 ? '0' + c : 'A' + c - 10);
      SeqID /= 36;
    }

    Out << 'S' << BufferPtr << '_';
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

void CXXNameMangler::addSubstitution(uintptr_t Ptr) {
  unsigned SeqID = Substitutions.size();

  assert(!Substitutions.count(Ptr) && "Substitution already exists!");
  Substitutions[Ptr] = SeqID;
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

/// \brief Mangles the a thunk with the offset n for the declaration D and
/// emits that name to the given output stream.
void MangleContext::mangleThunk(const FunctionDecl *FD, 
                                const ThunkAdjustment &ThisAdjustment,
                                llvm::SmallVectorImpl<char> &Res) {
  assert(!isa<CXXDestructorDecl>(FD) &&
         "Use mangleCXXDtor for destructor decls!");

  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZT";
  Mangler.mangleCallOffset(ThisAdjustment);
  Mangler.mangleFunctionEncoding(FD);
}

void MangleContext::mangleCXXDtorThunk(const CXXDestructorDecl *D,
                                       CXXDtorType Type,
                                       const ThunkAdjustment &ThisAdjustment,
                                       llvm::SmallVectorImpl<char> &Res) {
  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  CXXNameMangler Mangler(*this, Res, D, Type);
  Mangler.getStream() << "_ZT";
  Mangler.mangleCallOffset(ThisAdjustment);
  Mangler.mangleFunctionEncoding(D);
}

/// \brief Mangles the a covariant thunk for the declaration D and emits that
/// name to the given output stream.
void 
MangleContext::mangleCovariantThunk(const FunctionDecl *FD,
                                    const CovariantThunkAdjustment& Adjustment,
                                    llvm::SmallVectorImpl<char> &Res) {
  assert(!isa<CXXDestructorDecl>(FD) &&
         "No such thing as a covariant thunk for a destructor!");

  //  <special-name> ::= Tc <call-offset> <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  //                      # first call-offset is 'this' adjustment
  //                      # second call-offset is result adjustment
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTc";
  Mangler.mangleCallOffset(Adjustment.ThisAdjustment);
  Mangler.mangleCallOffset(Adjustment.ReturnAdjustment);
  Mangler.mangleFunctionEncoding(FD);
}

/// mangleGuardVariable - Returns the mangled name for a guard variable
/// for the passed in VarDecl.
void MangleContext::mangleGuardVariable(const VarDecl *D,
                                        llvm::SmallVectorImpl<char> &Res) {
  //  <special-name> ::= GV <object name>       # Guard variable for one-time
  //                                            # initialization
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZGV";
  Mangler.mangleName(D);
}

void MangleContext::mangleCXXVtable(const CXXRecordDecl *RD,
                                    llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TV <type>  # virtual table
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTV";
  Mangler.mangleName(RD);
}

void MangleContext::mangleCXXVTT(const CXXRecordDecl *RD,
                                 llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TT <type>  # VTT structure
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTT";
  Mangler.mangleName(RD);
}

void MangleContext::mangleCXXCtorVtable(const CXXRecordDecl *RD, int64_t Offset,
                                        const CXXRecordDecl *Type,
                                        llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TC <type> <offset number> _ <base type>
  CXXNameMangler Mangler(*this, Res);
  Mangler.getStream() << "_ZTC";
  Mangler.mangleName(RD);
  Mangler.getStream() << Offset;
  Mangler.getStream() << "_";
  Mangler.mangleName(Type);
}

void MangleContext::mangleCXXRTTI(QualType Ty,
                                  llvm::SmallVectorImpl<char> &Res) {
  // <special-name> ::= TI <type>  # typeinfo structure
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
