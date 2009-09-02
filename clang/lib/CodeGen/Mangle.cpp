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
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN CXXNameMangler {
    ASTContext &Context;
    llvm::raw_ostream &Out;

    const CXXMethodDecl *Structor;
    unsigned StructorType;
    CXXCtorType CtorType;
    
  public:
    CXXNameMangler(ASTContext &C, llvm::raw_ostream &os)
      : Context(C), Out(os), Structor(0), StructorType(0) { }

    bool mangle(const NamedDecl *D);
    void mangleThunk(const NamedDecl *ND, bool Virtual, int64_t nv, int64_t v);
    void mangleGuardVariable(const VarDecl *D);
    
    void mangleCXXVtable(QualType Type);
    void mangleCXXRtti(QualType Type);
    void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type);
    void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type);

  private:
    bool mangleFunctionDecl(const FunctionDecl *FD);
    
    void mangleFunctionEncoding(const FunctionDecl *FD);
    void mangleName(const NamedDecl *ND);
    void mangleUnqualifiedName(const NamedDecl *ND);
    void mangleSourceName(const IdentifierInfo *II);
    void mangleLocalName(const NamedDecl *ND);
    void mangleNestedName(const NamedDecl *ND);
    void manglePrefix(const DeclContext *DC);
    void mangleOperatorName(OverloadedOperatorKind OO, unsigned Arity);
    void mangleCVQualifiers(unsigned Quals);
    void mangleType(QualType T);
    void mangleType(const BuiltinType *T);
    void mangleType(const FunctionType *T);
    void mangleBareFunctionType(const FunctionType *T, bool MangleReturnType);
    void mangleType(const TagType *T);
    void mangleType(const ArrayType *T);
    void mangleType(const MemberPointerType *T);
    void mangleType(const TemplateTypeParmType *T);
    void mangleType(const ObjCInterfaceType *T);
    void mangleExpression(Expr *E);
    void mangleCXXCtorType(CXXCtorType T);
    void mangleCXXDtorType(CXXDtorType T);
    
    void mangleTemplateArgumentList(const TemplateArgumentList &L);
    void mangleTemplateArgument(const TemplateArgument &A);
  };
}

static bool isInCLinkageSpecification(const Decl *D) {
  for (const DeclContext *DC = D->getDeclContext(); 
       !DC->isTranslationUnit(); DC = DC->getParent()) {
    if (const LinkageSpecDecl *Linkage = dyn_cast<LinkageSpecDecl>(DC)) 
      return Linkage->getLanguage() == LinkageSpecDecl::lang_c;
  }
  
  return false;
}

bool CXXNameMangler::mangleFunctionDecl(const FunctionDecl *FD) {
  // Clang's "overloadable" attribute extension to C/C++ implies name mangling
  // (always).
  if (!FD->hasAttr<OverloadableAttr>()) {
    // C functions are not mangled, and "main" is never mangled.
    if (!Context.getLangOptions().CPlusPlus || FD->isMain(Context))
      return false;
    
    // No mangling in an "implicit extern C" header. 
    if (FD->getLocation().isValid() &&
        Context.getSourceManager().isInExternCSystemHeader(FD->getLocation()))
      return false;
    
    // No name mangling in a C linkage specification.
    if (isInCLinkageSpecification(FD))
      return false;
  }

  // If we get here, mangle the decl name!
  Out << "_Z";
  mangleFunctionEncoding(FD);
  return true;
}

bool CXXNameMangler::mangle(const NamedDecl *D) {
  // Any decl can be declared with __asm("foo") on it, and this takes precedence
  // over all other naming in the .o file.
  if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>()) {
    // If we have an asm name, then we use it as the mangling.
    Out << '\01';  // LLVM IR Marker for __asm("foo")
    Out << ALA->getLabel();
    return true;
  }
  
  // <mangled-name> ::= _Z <encoding>
  //            ::= <data name>
  //            ::= <special-name>

  // FIXME: Actually use a visitor to decode these?
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return mangleFunctionDecl(FD);
  
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (!Context.getLangOptions().CPlusPlus ||
        isInCLinkageSpecification(D) ||
        D->getDeclContext()->isTranslationUnit())
      return false;
    
    Out << "_Z";
    mangleName(VD);
    return true;
  }
  
  return false;
}

void CXXNameMangler::mangleCXXCtor(const CXXConstructorDecl *D, 
                                   CXXCtorType Type) {
  assert(!Structor && "Structor already set!");
  Structor = D;
  StructorType = Type;
  
  mangle(D);
}

void CXXNameMangler::mangleCXXDtor(const CXXDestructorDecl *D, 
                                   CXXDtorType Type) {
  assert(!Structor && "Structor already set!");
  Structor = D;
  StructorType = Type;
  
  mangle(D);
}

void CXXNameMangler::mangleCXXVtable(QualType T) {
  // <special-name> ::= TV <type>  # virtual table
  Out << "_ZTV";
  mangleType(T);
}

void CXXNameMangler::mangleCXXRtti(QualType T) {
  // <special-name> ::= TI <type>  # typeinfo structure
  Out << "_ZTI";
  mangleType(T);
}

void CXXNameMangler::mangleGuardVariable(const VarDecl *D)
{
  //  <special-name> ::= GV <object name>	# Guard variable for one-time 
  //                                            # initialization

  Out << "_ZGV";
  mangleName(D);
}

void CXXNameMangler::mangleFunctionEncoding(const FunctionDecl *FD) {
  // <encoding> ::= <function name> <bare-function-type>
  mangleName(FD);
  
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
  if (FD->getPrimaryTemplate() &&
      !(isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD) ||
        isa<CXXConversionDecl>(FD)))
    MangleReturnType = true;
  mangleBareFunctionType(FD->getType()->getAsFunctionType(), MangleReturnType);
}

static bool isStdNamespace(const DeclContext *DC) {
  if (!DC->isNamespace() || !DC->getParent()->isTranslationUnit())
    return false;

  const NamespaceDecl *NS = cast<NamespaceDecl>(DC);
  return NS->getOriginalNamespace()->getIdentifier()->isStr("std");
}

void CXXNameMangler::mangleName(const NamedDecl *ND) {
  //  <name> ::= <nested-name>
  //         ::= <unscoped-name>
  //         ::= <unscoped-template-name> <template-args>
  //         ::= <local-name>     # See Scope Encoding below
  //
  //  <unscoped-name> ::= <unqualified-name>
  //                  ::= St <unqualified-name>   # ::std::
  if (ND->getDeclContext()->isTranslationUnit()) 
    mangleUnqualifiedName(ND);
  else if (isStdNamespace(ND->getDeclContext())) {
    Out << "St";
    mangleUnqualifiedName(ND);
  } else if (isa<FunctionDecl>(ND->getDeclContext()))
    mangleLocalName(ND);
  else
    mangleNestedName(ND);
}

void CXXNameMangler::mangleThunk(const NamedDecl *D, bool Virtual, int64_t nv,
                                 int64_t v) {
  //  <special-name> ::= T <call-offset> <base encoding>
  //                      # base is the nominal target function of thunk
  //  <call-offset>  ::= h <nv-offset> _
  //                 ::= v <v-offset> _
  //  <nv-offset>    ::= <offset number>        # non-virtual base override
  //  <v-offset>     ::= <offset nubmer> _ <virtual offset number>
  //                      # virtual base override, with vcall offset
  if (!Virtual) {
    Out << "_Th";
    if (nv < 0) {
      Out << "n";
      nv = -nv;
    }
    Out << nv;
  } else {
    Out << "_Tv";
    if (nv < 0) {
      Out << "n";
      nv = -nv;
    }
    Out << nv;
    Out << "_";
    if (v < 0) {
      Out << "n";
      v = -v;
    }
    Out << v;
  }
  Out << "_";
  mangleName(D);
}

void CXXNameMangler::mangleUnqualifiedName(const NamedDecl *ND) {
  //  <unqualified-name> ::= <operator-name>
  //                     ::= <ctor-dtor-name>  
  //                     ::= <source-name>   
  DeclarationName Name = ND->getDeclName();
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    mangleSourceName(Name.getAsIdentifierInfo());
    break;

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
    // <operator-name> ::= cv <type>	# (cast) 
    Out << "cv";
    mangleType(Context.getCanonicalType(Name.getCXXNameType()));
    break;

  case DeclarationName::CXXOperatorName:
    mangleOperatorName(Name.getCXXOverloadedOperator(),
                       cast<FunctionDecl>(ND)->getNumParams());
    break;

  case DeclarationName::CXXUsingDirective:
    assert(false && "Can't mangle a using directive name!");
    break;
  }
  
  if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(ND)) {
    if (const TemplateArgumentList *TemplateArgs 
          = Function->getTemplateSpecializationArgs())
      mangleTemplateArgumentList(*TemplateArgs);
  }
}

void CXXNameMangler::mangleSourceName(const IdentifierInfo *II) {
  // <source-name> ::= <positive length number> <identifier>
  // <number> ::= [n] <non-negative decimal integer>
  // <identifier> ::= <unqualified source code identifier>
  Out << II->getLength() << II->getName();
}

void CXXNameMangler::mangleNestedName(const NamedDecl *ND) {
  // <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
  //               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E
  // FIXME: no template support
  Out << 'N';
  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(ND))
    mangleCVQualifiers(Method->getTypeQualifiers());
  manglePrefix(ND->getDeclContext());
  mangleUnqualifiedName(ND);
  Out << 'E';
}

void CXXNameMangler::mangleLocalName(const NamedDecl *ND) {
  // <local-name> := Z <function encoding> E <entity name> [<discriminator>]
  //              := Z <function encoding> E s [<discriminator>]
  // <discriminator> := _ <non-negative number> 
  Out << 'Z';
  mangleFunctionEncoding(cast<FunctionDecl>(ND->getDeclContext()));
  Out << 'E';
  mangleSourceName(ND->getIdentifier());
}

void CXXNameMangler::manglePrefix(const DeclContext *DC) {
  //  <prefix> ::= <prefix> <unqualified-name>
  //           ::= <template-prefix> <template-args>
  //           ::= <template-param>
  //           ::= # empty
  //           ::= <substitution>
  // FIXME: We only handle mangling of namespaces and classes at the moment.
  if (!DC->getParent()->isTranslationUnit())
    manglePrefix(DC->getParent());

  if (const NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(DC))
    mangleSourceName(Namespace->getIdentifier());
  else if (const RecordDecl *Record = dyn_cast<RecordDecl>(DC)) {
    if (const ClassTemplateSpecializationDecl *D = 
        dyn_cast<ClassTemplateSpecializationDecl>(Record)) {
      mangleType(QualType(D->getTypeForDecl(), 0));
    } else
      mangleSourceName(Record->getIdentifier());
  }
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
  // UNSUPPORTED: ::= qu        # ?

  case OO_None:
  case OO_Conditional:
  case NUM_OVERLOADED_OPERATORS:
    assert(false && "Not an overloaded operator"); 
    break;
  }
}

void CXXNameMangler::mangleCVQualifiers(unsigned Quals) {
  // <CV-qualifiers> ::= [r] [V] [K] 	# restrict (C99), volatile, const
  if (Quals & QualType::Restrict)
    Out << 'r';
  if (Quals & QualType::Volatile)
    Out << 'V';
  if (Quals & QualType::Const)
    Out << 'K';
}

void CXXNameMangler::mangleType(QualType T) {
  // Only operate on the canonical type!
  T = Context.getCanonicalType(T);

  // FIXME: Should we have a TypeNodes.def to make this easier? (YES!)

  //  <type> ::= <CV-qualifiers> <type>
  mangleCVQualifiers(T.getCVRQualifiers());

  //         ::= <builtin-type>
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T.getTypePtr()))
    mangleType(BT);
  //         ::= <function-type>
  else if (const FunctionType *FT = dyn_cast<FunctionType>(T.getTypePtr()))
    mangleType(FT);
  //         ::= <class-enum-type>
  else if (const TagType *TT = dyn_cast<TagType>(T.getTypePtr()))
    mangleType(TT);
  //         ::= <array-type>
  else if (const ArrayType *AT = dyn_cast<ArrayType>(T.getTypePtr()))
    mangleType(AT);
  //         ::= <pointer-to-member-type>
  else if (const MemberPointerType *MPT 
             = dyn_cast<MemberPointerType>(T.getTypePtr()))
    mangleType(MPT);
  //         ::= <template-param>
  else if (const TemplateTypeParmType *TypeParm 
             = dyn_cast<TemplateTypeParmType>(T.getTypePtr()))
    mangleType(TypeParm);
  //  FIXME: ::= <template-template-param> <template-args>
  //  FIXME: ::= <substitution> # See Compression below
  //         ::= P <type>   # pointer-to
  else if (const PointerType *PT = dyn_cast<PointerType>(T.getTypePtr())) {
    Out << 'P';
    mangleType(PT->getPointeeType());
  }
  else if (const ObjCObjectPointerType *PT = 
           dyn_cast<ObjCObjectPointerType>(T.getTypePtr())) {
    Out << 'P';
    mangleType(PT->getPointeeType());
  }
  //         ::= R <type>   # reference-to
  else if (const LValueReferenceType *RT =
           dyn_cast<LValueReferenceType>(T.getTypePtr())) {
    Out << 'R';
    mangleType(RT->getPointeeType());
  }
  //         ::= O <type>   # rvalue reference-to (C++0x)
  else if (const RValueReferenceType *RT =
           dyn_cast<RValueReferenceType>(T.getTypePtr())) {
    Out << 'O';
    mangleType(RT->getPointeeType());
  }
  //         ::= C <type>   # complex pair (C 2000)
  else if (const ComplexType *CT = dyn_cast<ComplexType>(T.getTypePtr())) {
    Out << 'C';
    mangleType(CT->getElementType());
  } else if (const VectorType *VT = dyn_cast<VectorType>(T.getTypePtr())) {
    // GNU extension: vector types
    Out << "U8__vector";
    mangleType(VT->getElementType());
  } else if (const ObjCInterfaceType *IT = 
             dyn_cast<ObjCInterfaceType>(T.getTypePtr())) {
    mangleType(IT);
  }
  // FIXME:  ::= G <type>   # imaginary (C 2000)
  // FIXME:  ::= U <source-name> <type>     # vendor extended type qualifier
  else
    assert(false && "Cannot mangle unknown type");
}

void CXXNameMangler::mangleType(const BuiltinType *T) {
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
  }
}

void CXXNameMangler::mangleType(const FunctionType *T) {
  // <function-type> ::= F [Y] <bare-function-type> E
  Out << 'F';
  // FIXME: We don't have enough information in the AST to produce the 'Y'
  // encoding for extern "C" function types.
  mangleBareFunctionType(T, /*MangleReturnType=*/true);
  Out << 'E';
}

void CXXNameMangler::mangleBareFunctionType(const FunctionType *T,
                                            bool MangleReturnType) {
  // <bare-function-type> ::= <signature type>+
  if (MangleReturnType)
    mangleType(T->getResultType());

  const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(T);
  assert(Proto && "Can't mangle K&R function prototypes");

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

void CXXNameMangler::mangleType(const TagType *T) {
  //  <class-enum-type> ::= <name>
  
  if (!T->getDecl()->getIdentifier())
    mangleName(T->getDecl()->getTypedefForAnonDecl());
  else
    mangleName(T->getDecl());
  
  // If this is a class template specialization, mangle the template arguments.
  if (ClassTemplateSpecializationDecl *Spec 
      = dyn_cast<ClassTemplateSpecializationDecl>(T->getDecl()))
    mangleTemplateArgumentList(Spec->getTemplateArgs());
}

void CXXNameMangler::mangleType(const ArrayType *T) {
  // <array-type> ::= A <positive dimension number> _ <element type>
  //              ::= A [<dimension expression>] _ <element type>
  Out << 'A';
  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(T))
    Out << CAT->getSize();
  else if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(T))
    mangleExpression(VAT->getSizeExpr());
  else if (const DependentSizedArrayType *DSAT 
             = dyn_cast<DependentSizedArrayType>(T))
    mangleExpression(DSAT->getSizeExpr());

  Out << '_';
  mangleType(T->getElementType());
}

void CXXNameMangler::mangleType(const MemberPointerType *T) {
  //  <pointer-to-member-type> ::= M <class type> <member type>
  Out << 'M';
  mangleType(QualType(T->getClass(), 0));
  QualType PointeeType = T->getPointeeType();
  if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(PointeeType)) {
    mangleCVQualifiers(FPT->getTypeQuals());
    mangleType(FPT);
  } else 
    mangleType(PointeeType);
}

void CXXNameMangler::mangleType(const TemplateTypeParmType *T) {
  // <template-param> ::= T_    # first template parameter
  //                  ::= T <parameter-2 non-negative number> _
  if (T->getIndex() == 0)
    Out << "T_";
  else
    Out << 'T' << (T->getIndex() - 1) << '_';
}

void CXXNameMangler::mangleType(const ObjCInterfaceType *T) {
  mangleSourceName(T->getDecl()->getIdentifier());
}

void CXXNameMangler::mangleExpression(Expr *E) {
  assert(false && "Cannot mangle expressions yet");
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

void CXXNameMangler::mangleTemplateArgumentList(const TemplateArgumentList &L) {
  // <template-args> ::= I <template-arg>+ E
  Out << "I";
  
  for (unsigned i = 0, e = L.size(); i != e; ++i) {
    const TemplateArgument &A = L[i];
  
    mangleTemplateArgument(A);
  }
  
  Out << "E";
}

void CXXNameMangler::mangleTemplateArgument(const TemplateArgument &A) {
  // <template-arg> ::= <type>			        # type or template
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
  case TemplateArgument::Integral:
    //  <expr-primary> ::= L <type> <value number> E # integer literal

    Out << 'L';
    
    mangleType(A.getIntegralType());
    
    const llvm::APSInt *Integral = A.getAsIntegral();
    if (A.getIntegralType()->isBooleanType()) {
      // Boolean values are encoded as 0/1.
      Out << (Integral->getBoolValue() ? '1' : '0');
    } else {
      if (Integral->isNegative())
        Out << 'n';
      Integral->abs().print(Out, false);
    }
      
    Out << 'E';
    break;
  }
}

namespace clang {
  /// \brief Mangles the name of the declaration D and emits that name to the
  /// given output stream.
  ///
  /// If the declaration D requires a mangled name, this routine will emit that
  /// mangled name to \p os and return true. Otherwise, \p os will be unchanged
  /// and this routine will return false. In this case, the caller should just
  /// emit the identifier of the declaration (\c D->getIdentifier()) as its
  /// name.
  bool mangleName(const NamedDecl *D, ASTContext &Context, 
                  llvm::raw_ostream &os) {
    assert(!isa<CXXConstructorDecl>(D) &&
           "Use mangleCXXCtor for constructor decls!");
    assert(!isa<CXXDestructorDecl>(D) &&
           "Use mangleCXXDtor for destructor decls!");
    
    CXXNameMangler Mangler(Context, os);
    if (!Mangler.mangle(D))
      return false;
    
    os.flush();
    return true;
  }
  
  /// \brief Mangles the a thunk with the offset n for the declaration D and
  /// emits that name to the given output stream.
  void mangleThunk(const NamedDecl *D, bool Virtual, int64_t nv, int64_t v,
                   ASTContext &Context, llvm::raw_ostream &os) {
    // FIXME: Hum, we might have to thunk these, fix.
    assert(!isa<CXXConstructorDecl>(D) &&
           "Use mangleCXXCtor for constructor decls!");
    assert(!isa<CXXDestructorDecl>(D) &&
           "Use mangleCXXDtor for destructor decls!");
    
    CXXNameMangler Mangler(Context, os);
    Mangler.mangleThunk(D, Virtual, nv, v);
    os.flush();
  }
  
  /// mangleGuardVariable - Returns the mangled name for a guard variable
  /// for the passed in VarDecl.
  void mangleGuardVariable(const VarDecl *D, ASTContext &Context,
                           llvm::raw_ostream &os) {
    CXXNameMangler Mangler(Context, os);
    Mangler.mangleGuardVariable(D);

    os.flush();
  }
  
  void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                     ASTContext &Context, llvm::raw_ostream &os) {
    CXXNameMangler Mangler(Context, os);
    Mangler.mangleCXXCtor(D, Type);
    
    os.flush();
  }
  
  void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                     ASTContext &Context, llvm::raw_ostream &os) {
    CXXNameMangler Mangler(Context, os);
    Mangler.mangleCXXDtor(D, Type);
    
    os.flush();
  }

  void mangleCXXVtable(QualType Type, ASTContext &Context,
                       llvm::raw_ostream &os) {
    CXXNameMangler Mangler(Context, os);
    Mangler.mangleCXXVtable(Type);

    os.flush();
  }

  void mangleCXXRtti(QualType Type, ASTContext &Context,
                     llvm::raw_ostream &os) {
    CXXNameMangler Mangler(Context, os);
    Mangler.mangleCXXRtti(Type);

    os.flush();
  }
}
