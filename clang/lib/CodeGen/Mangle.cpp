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
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {
  class VISIBILITY_HIDDEN CXXNameMangler {
    ASTContext &Context;
    llvm::raw_ostream &Out;

  public:
    CXXNameMangler(ASTContext &C, llvm::raw_ostream &os)
      : Context(C), Out(os) { }

    bool mangle(const NamedDecl *D);
    void mangleFunctionEncoding(const FunctionDecl *FD);
    void mangleName(const NamedDecl *ND);
    void mangleUnqualifiedName(const NamedDecl *ND);
    void mangleSourceName(const IdentifierInfo *II);
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
  };
}


bool CXXNameMangler::mangle(const NamedDecl *D) {
  // <mangled-name> ::= _Z <encoding>
  //            ::= <data name>
  //            ::= <special-name>

  // FIXME: Actually use a visitor to decode these?
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    bool RequiresMangling = false;
    // Clang's "overloadable" attribute extension to C/C++ implies
    // name mangling (always).
    if (FD->getAttr<OverloadableAttr>())
      RequiresMangling = true;
    // No mangled in an "implicit extern C" header.
    else if (Context.getSourceManager().getFileCharacteristic(FD->getLocation())
          == SrcMgr::C_ExternCSystem)
      RequiresMangling = false;
    else if (Context.getLangOptions().CPlusPlus && !FD->isMain()) {
      // C++ requires name mangling, unless we're in a C linkage
      // specification.
      RequiresMangling = true;

      for (const DeclContext *DC = FD->getDeclContext(); 
           !DC->isTranslationUnit(); DC = DC->getParent()) {
        if (const LinkageSpecDecl *Linkage = dyn_cast<LinkageSpecDecl>(DC)) {
          // extern "C" functions don't use name mangling
          if (Linkage->getLanguage() == LinkageSpecDecl::lang_c)
            RequiresMangling = false;
          break;
        }
      }
    }

    if (RequiresMangling) {
      Out << "_Z";
      mangleFunctionEncoding(FD);
      return true;
    }
  } 

  return false;
}

void CXXNameMangler::mangleFunctionEncoding(const FunctionDecl *FD) {
  // <encoding> ::= <function name> <bare-function-type>
  mangleName(FD);
  mangleBareFunctionType(FD->getType()->getAsFunctionType(), false);
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
  } else {
    mangleNestedName(ND);
  }
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
    // <ctor-dtor-name> ::= C1  # complete object constructor
    //                  ::= C2  # base object constructor
    //                  ::= C3  # complete object allocating constructor
    //
    // FIXME: We don't even have all of these constructors
    // in the AST yet.
    Out << "C1";
    break;

  case DeclarationName::CXXDestructorName:
    // <ctor-dtor-name> ::= D0  # deleting destructor
    //                  ::= D1  # complete object destructor
    //                  ::= D2  # base object destructor
    //
    // FIXME: We don't even have all of these destructors in the AST
    // yet.
    Out << "D0";
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

void CXXNameMangler::manglePrefix(const DeclContext *DC) {
  //  <prefix> ::= <prefix> <unqualified-name>
  //           ::= <template-prefix> <template-args>
  //           ::= <template-param>
  //           ::= # empty
  //           ::= <substitution>
  // FIXME: We only handle mangling of namespaces and classes at the moment.
  if (DC->getParent() != DC)
    manglePrefix(DC);

  if (const NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(DC))
    mangleSourceName(Namespace->getIdentifier());
  else if (const RecordDecl *Record = dyn_cast<RecordDecl>(DC))
    mangleSourceName(Record->getIdentifier());
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
  // UNSUPPORTED:    ::= Di # char32_t
  // UNSUPPORTED:    ::= Ds # char16_t
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
  case BuiltinType::SChar: Out << 'a'; break;
  case BuiltinType::WChar: Out << 'w'; break;
  case BuiltinType::Short: Out << 's'; break;
  case BuiltinType::Int: Out << 'i'; break;
  case BuiltinType::Long: Out << 'l'; break;
  case BuiltinType::LongLong: Out << 'x'; break;
  case BuiltinType::Float: Out << 'f'; break;
  case BuiltinType::Double: Out << 'd'; break;
  case BuiltinType::LongDouble: Out << 'e'; break;

  case BuiltinType::Overload:
  case BuiltinType::Dependent:
    assert(false && 
           "Overloaded and dependent types shouldn't get to name mangling");
    break;
  }
}

void CXXNameMangler::mangleType(const FunctionType *T) {
  // <function-type> ::= F [Y] <bare-function-type> E
  Out << 'F';
  // FIXME: We don't have enough information in the AST to produce the
  // 'Y' encoding for extern "C" function types.
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
  mangleType(T->getPointeeType());
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

namespace clang {
  /// \brief Mangles the name of the declaration D and emits that name
  /// to the given output stream.
  ///
  /// If the declaration D requires a mangled name, this routine will
  /// emit that mangled name to \p os and return true. Otherwise, \p
  /// os will be unchanged and this routine will return false. In this
  /// case, the caller should just emit the identifier of the declaration
  /// (\c D->getIdentifier()) as its name.
  bool mangleName(const NamedDecl *D, ASTContext &Context, 
                  llvm::raw_ostream &os) {
    CXXNameMangler Mangler(Context, os);
    if (!Mangler.mangle(D))
      return false;
    
    os.flush();
    return true;
  }
}

