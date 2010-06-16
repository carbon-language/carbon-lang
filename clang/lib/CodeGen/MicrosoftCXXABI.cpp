//===--- MicrosoftCXXABI.cpp - Emit LLVM Code from ASTs for a Module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targetting the Microsoft Visual C++ ABI.
// The class in this file generates structures that follow the Microsoft
// Visual C++ ABI, which is actually not very well documented at all outside
// of Microsoft.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CodeGenModule.h"
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "CGVTables.h"

using namespace clang;
using namespace CodeGen;

namespace {

/// MicrosoftCXXNameMangler - Manage the mangling of a single name for the
/// Microsoft Visual C++ ABI.
class MicrosoftCXXNameMangler {
  MangleContext &Context;
  llvm::raw_svector_ostream Out;

  ASTContext &getASTContext() const { return Context.getASTContext(); }

public:
  MicrosoftCXXNameMangler(MangleContext &C, llvm::SmallVectorImpl<char> &Res)
  : Context(C), Out(Res) { }

  llvm::raw_svector_ostream &getStream() { return Out; }

  void mangle(const NamedDecl *D, llvm::StringRef Prefix = "?");
  void mangleName(const NamedDecl *ND);
  void mangleFunctionEncoding(const FunctionDecl *FD);
  void mangleVariableEncoding(const VarDecl *VD);
  void mangleType(QualType T);

private:
  void mangleUnqualifiedName(const NamedDecl *ND) {
    mangleUnqualifiedName(ND, ND->getDeclName());
  }
  void mangleUnqualifiedName(const NamedDecl *ND, DeclarationName Name);
  void mangleSourceName(const IdentifierInfo *II);
  void manglePostfix(const DeclContext *DC, bool NoFunction=false);
  void mangleQualifiers(Qualifiers Quals, bool IsMember);

  void mangleObjCMethodName(const ObjCMethodDecl *MD);

  // Declare manglers for every type class.
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) void mangleType(const CLASS##Type *T);
#include "clang/AST/TypeNodes.def"
  
  void mangleType(const FunctionType *T, bool IsStructor);
  void mangleFunctionClass(const FunctionDecl *FD);
  void mangleCallingConvention(const FunctionType *T);
  void mangleThrowSpecification(const FunctionProtoType *T);

};

/// MicrosoftMangleContext - Overrides the default MangleContext for the
/// Microsoft Visual C++ ABI.
class MicrosoftMangleContext : public MangleContext {
public:
  MicrosoftMangleContext(ASTContext &Context,
                         Diagnostic &Diags) : MangleContext(Context, Diags) { }
  virtual bool shouldMangleDeclName(const NamedDecl *D);
  virtual void mangleName(const NamedDecl *D, llvm::SmallVectorImpl<char> &);
  virtual void mangleThunk(const CXXMethodDecl *MD,
                           const ThunkInfo &Thunk,
                           llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXDtorThunk(const CXXDestructorDecl *DD, CXXDtorType Type,
                                  const ThisAdjustment &ThisAdjustment,
                                  llvm::SmallVectorImpl<char> &);
  virtual void mangleGuardVariable(const VarDecl *D,
                                   llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXVTable(const CXXRecordDecl *RD,
                               llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXVTT(const CXXRecordDecl *RD,
                            llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXCtorVTable(const CXXRecordDecl *RD, int64_t Offset,
                                   const CXXRecordDecl *Type,
                                   llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXRTTI(QualType T, llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXRTTIName(QualType T, llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXCtor(const CXXConstructorDecl *D, CXXCtorType Type,
                             llvm::SmallVectorImpl<char> &);
  virtual void mangleCXXDtor(const CXXDestructorDecl *D, CXXDtorType Type,
                             llvm::SmallVectorImpl<char> &);
};

class MicrosoftCXXABI : public CXXABI {
  MicrosoftMangleContext MangleCtx;
public:
  MicrosoftCXXABI(CodeGenModule &CGM)
   : MangleCtx(CGM.getContext(), CGM.getDiags()) {}

  MicrosoftMangleContext &getMangleContext() {
    return MangleCtx;
  }
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

bool MicrosoftMangleContext::shouldMangleDeclName(const NamedDecl *D) {
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

  // Variables at global scope with internal linkage are not mangled.
  if (!FD) {
    const DeclContext *DC = D->getDeclContext();
    if (DC->isTranslationUnit() && D->getLinkage() == InternalLinkage)
      return false;
  }

  // C functions and "main" are not mangled.
  if ((FD && FD->isMain()) || isInCLinkageSpecification(D))
    return false;

  return true;
}

void MicrosoftCXXNameMangler::mangle(const NamedDecl *D,
                                     llvm::StringRef Prefix) {
  // MSVC doesn't mangle C++ names the same way it mangles extern "C" names.
  // Therefore it's really important that we don't decorate the
  // name with leading underscores or leading/trailing at signs. So, emit a
  // asm marker at the start so we get the name right.
  Out << '\01';  // LLVM IR Marker for __asm("foo")

  // Any decl can be declared with __asm("foo") on it, and this takes precedence
  // over all other naming in the .o file.
  if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>()) {
    // If we have an asm name, then we use it as the mangling.
    Out << ALA->getLabel();
    return;
  }

  // <mangled-name> ::= ? <name> <type-encoding>
  Out << Prefix;
  mangleName(D);
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    mangleFunctionEncoding(FD);
  else if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    mangleVariableEncoding(VD);
  // TODO: Fields? Can MSVC even mangle them?
}

void MicrosoftCXXNameMangler::mangleFunctionEncoding(const FunctionDecl *FD) {
  // <type-encoding> ::= <function-class> <function-type>

  // Don't mangle in the type if this isn't a decl we should typically mangle.
  if (!Context.shouldMangleDeclName(FD))
    return;
  
  // We should never ever see a FunctionNoProtoType at this point.
  // We don't even know how to mangle their types anyway :).
  FunctionProtoType *OldType = cast<FunctionProtoType>(FD->getType());

  bool InStructor = false;
  const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD) {
    if (isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD))
      InStructor = true;
  }

  // First, the function class.
  mangleFunctionClass(FD);

  // If this is a C++ instance method, mangle the CVR qualifiers for the
  // this pointer.
  if (MD && MD->isInstance())
    mangleQualifiers(Qualifiers::fromCVRMask(OldType->getTypeQuals()), false);

  // Do the canonicalization out here because parameter types can
  // undergo additional canonicalization (e.g. array decay).
  const FunctionProtoType *FT = cast<FunctionProtoType>(getASTContext()
                                                    .getCanonicalType(OldType));
  // If the function's type had a throw spec, canonicalization removed it.
  // Get it back.
  FT = cast<FunctionProtoType>(getASTContext().getFunctionType(
                                                FT->getResultType(),
                                                FT->arg_type_begin(),
                                                FT->getNumArgs(),
                                                FT->isVariadic(),
                                                FT->getTypeQuals(),
                                                OldType->hasExceptionSpec(),
                                                OldType->hasAnyExceptionSpec(),
                                                OldType->getNumExceptions(),
                                                OldType->exception_begin(),
                                                FT->getExtInfo()).getTypePtr());
  mangleType(FT, InStructor);
}

void MicrosoftCXXNameMangler::mangleVariableEncoding(const VarDecl *VD) {
  // <type-encoding> ::= <storage-class> <variable-type>
  // <storage-class> ::= 0  # private static member
  //                 ::= 1  # protected static member
  //                 ::= 2  # public static member
  //                 ::= 3  # global
  //                 ::= 4  # static local
  
  // The first character in the encoding (after the name) is the storage class.
  if (VD->isStaticDataMember()) {
    // If it's a static member, it also encodes the access level.
    switch (VD->getAccess()) {
      default:
      case AS_private: Out << '0'; break;
      case AS_protected: Out << '1'; break;
      case AS_public: Out << '2'; break;
    }
  }
  else if (!VD->isStaticLocal())
    Out << '3';
  else
    Out << '4';
  // Now mangle the type.
  // <variable-type> ::= <type> <cvr-qualifiers>
  QualType Ty = VD->getType();
  mangleType(Ty.getLocalUnqualifiedType());
  mangleQualifiers(Ty.getLocalQualifiers(), false);
}

void MicrosoftCXXNameMangler::mangleName(const NamedDecl *ND) {
  // <name> ::= <unscoped-name> {[<named-scope>]+ | [<nested-name>]}? @
  const DeclContext *DC = ND->getDeclContext();

  // Always start with the unqualified name.
  mangleUnqualifiedName(ND);    

  // If this is an extern variable declared locally, the relevant DeclContext
  // is that of the containing namespace, or the translation unit.
  if (isa<FunctionDecl>(DC) && ND->hasLinkage())
    while (!DC->isNamespace() && !DC->isTranslationUnit())
      DC = DC->getParent();

  manglePostfix(DC);

  // Terminate the whole name with an '@'.
  Out << '@';
}

void
MicrosoftCXXNameMangler::mangleUnqualifiedName(const NamedDecl *ND,
                                               DeclarationName Name) {
  //  <unqualified-name> ::= <operator-name>
  //                     ::= <ctor-dtor-name>
  //                     ::= <source-name>
  switch (Name.getNameKind()) {
    case DeclarationName::Identifier: {
      if (const IdentifierInfo *II = Name.getAsIdentifierInfo()) {
        mangleSourceName(II);
        break;
      }
      
      // Otherwise, an anonymous entity.  We must have a declaration.
      assert(ND && "mangling empty name without declaration");
      
      if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(ND)) {
        if (NS->isAnonymousNamespace()) {
          Out << "?A";
          break;
        }
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

      // TODO: How does VC mangle anonymous structs?
      assert(false && "Don't know how to mangle anonymous types yet!");
      break;
    }
      
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      assert(false && "Can't mangle Objective-C selector names here!");
      break;
      
    case DeclarationName::CXXConstructorName:
      assert(false && "Can't mangle constructors yet!");
      break;
      
    case DeclarationName::CXXDestructorName:
      assert(false && "Can't mangle destructors yet!");
      break;
      
    case DeclarationName::CXXConversionFunctionName:
      // <operator-name> ::= ?B # (cast)
      // The target type is encoded as the return type.
      Out << "?B";
      break;
      
    case DeclarationName::CXXOperatorName:
      assert(false && "Can't mangle operators yet!");
      
    case DeclarationName::CXXLiteralOperatorName:
      // FIXME: Was this added in VS2010? Does MS even know how to mangle this?
      assert(false && "Don't know how to mangle literal operators yet!");
      break;
      
    case DeclarationName::CXXUsingDirective:
      assert(false && "Can't mangle a using directive name!");
      break;
  }
}

void MicrosoftCXXNameMangler::manglePostfix(const DeclContext *DC,
                                            bool NoFunction) {
  // <postfix> ::= <unqualified-name> [<postfix>]
  //           ::= <template-postfix> <template-args> [<postfix>]
  //           ::= <template-param>
  //           ::= <substitution> [<postfix>]

  if (!DC) return;

  while (isa<LinkageSpecDecl>(DC))
    DC = DC->getParent();

  if (DC->isTranslationUnit())
    return;

  if (const BlockDecl *BD = dyn_cast<BlockDecl>(DC)) {
    llvm::SmallString<64> Name;
    Context.mangleBlock(BD, Name);
    Out << Name << '@';
    return manglePostfix(DC->getParent(), NoFunction);
  }

  if (NoFunction && (isa<FunctionDecl>(DC) || isa<ObjCMethodDecl>(DC)))
    return;
  else if (const ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(DC))
    mangleObjCMethodName(Method);
  else {
    mangleUnqualifiedName(cast<NamedDecl>(DC));
    manglePostfix(DC->getParent(), NoFunction);
  }
}

void MicrosoftCXXNameMangler::mangleSourceName(const IdentifierInfo *II) {
  // <source name> ::= <identifier> @
  Out << II->getName() << '@';
}

void MicrosoftCXXNameMangler::mangleObjCMethodName(const ObjCMethodDecl *MD) {
  llvm::SmallString<64> Buffer;
  MiscNameMangler(Context, Buffer).mangleObjCMethodName(MD);
  Out << Buffer;
}

void MicrosoftCXXNameMangler::mangleQualifiers(Qualifiers Quals,
                                               bool IsMember) {
  // <cvr-qualifiers> ::= [E] [F] [I] <base-cvr-qualifiers>
  // 'E' means __ptr64 (32-bit only); 'F' means __unaligned (32/64-bit only);
  // 'I' means __restrict (32/64-bit).
  // Note that the MSVC __restrict keyword isn't the same as the C99 restrict
  // keyword!
  // <base-cvr-qualifiers> ::= A  # near
  //                       ::= B  # near const
  //                       ::= C  # near volatile
  //                       ::= D  # near const volatile
  //                       ::= E  # far (16-bit)
  //                       ::= F  # far const (16-bit)
  //                       ::= G  # far volatile (16-bit)
  //                       ::= H  # far const volatile (16-bit)
  //                       ::= I  # huge (16-bit)
  //                       ::= J  # huge const (16-bit)
  //                       ::= K  # huge volatile (16-bit)
  //                       ::= L  # huge const volatile (16-bit)
  //                       ::= M <basis> # based
  //                       ::= N <basis> # based const
  //                       ::= O <basis> # based volatile
  //                       ::= P <basis> # based const volatile
  //                       ::= Q  # near member
  //                       ::= R  # near const member
  //                       ::= S  # near volatile member
  //                       ::= T  # near const volatile member
  //                       ::= U  # far member (16-bit)
  //                       ::= V  # far const member (16-bit)
  //                       ::= W  # far volatile member (16-bit)
  //                       ::= X  # far const volatile member (16-bit)
  //                       ::= Y  # huge member (16-bit)
  //                       ::= Z  # huge const member (16-bit)
  //                       ::= 0  # huge volatile member (16-bit)
  //                       ::= 1  # huge const volatile member (16-bit)
  //                       ::= 2 <basis> # based member
  //                       ::= 3 <basis> # based const member
  //                       ::= 4 <basis> # based volatile member
  //                       ::= 5 <basis> # based const volatile member
  //                       ::= 6  # near function (pointers only)
  //                       ::= 7  # far function (pointers only)
  //                       ::= 8  # near method (pointers only)
  //                       ::= 9  # far method (pointers only)
  //                       ::= _A <basis> # based function (pointers only)
  //                       ::= _B <basis> # based function (far?) (pointers only)
  //                       ::= _C <basis> # based method (pointers only)
  //                       ::= _D <basis> # based method (far?) (pointers only)
  // <basis> ::= 0 # __based(void)
  //         ::= 1 # __based(segment)?
  //         ::= 2 <name> # __based(name)
  //         ::= 3 # ?
  //         ::= 4 # ?
  //         ::= 5 # not really based
  if (!IsMember) {
    if (!Quals.hasVolatile()) {
      if (!Quals.hasConst())
        Out << 'A';
      else
        Out << 'B';
    } else {
      if (!Quals.hasConst())
        Out << 'C';
      else
        Out << 'D';
    }
  } else {
    if (!Quals.hasVolatile()) {
      if (!Quals.hasConst())
        Out << 'Q';
      else
        Out << 'R';
    } else {
      if (!Quals.hasConst())
        Out << 'S';
      else
        Out << 'T';
    }
  }

  // FIXME: For now, just drop all extension qualifiers on the floor.
}

void MicrosoftCXXNameMangler::mangleType(QualType T) {
  // Only operate on the canonical type!
  T = getASTContext().getCanonicalType(T);
  
  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define NON_CANONICAL_TYPE(CLASS, PARENT) \
case Type::CLASS: \
llvm_unreachable("can't mangle non-canonical type " #CLASS "Type"); \
return;
#define TYPE(CLASS, PARENT)
#include "clang/AST/TypeNodes.def"
  case Type::Builtin:
    mangleType(static_cast<BuiltinType *>(T.getTypePtr()));
    break;
  default:
    assert(false && "Don't know how to mangle this type!");
    break;
  }
}

void MicrosoftCXXNameMangler::mangleType(const BuiltinType *T) {
  //  <type>         ::= <builtin-type>
  //  <builtin-type> ::= X  # void
  //                 ::= C  # signed char
  //                 ::= D  # char
  //                 ::= E  # unsigned char
  //                 ::= F  # short
  //                 ::= G  # unsigned short (or wchar_t if it's not a builtin)
  //                 ::= H  # int
  //                 ::= I  # unsigned int
  //                 ::= J  # long
  //                 ::= K  # unsigned long
  //                     L  # <none>
  //                 ::= M  # float
  //                 ::= N  # double
  //                 ::= O  # long double (__float80 is mangled differently)
  //                 ::= _D # __int8 (yup, it's a distinct type in MSVC)
  //                 ::= _E # unsigned __int8
  //                 ::= _F # __int16
  //                 ::= _G # unsigned __int16
  //                 ::= _H # __int32
  //                 ::= _I # unsigned __int32
  //                 ::= _J # long long, __int64
  //                 ::= _K # unsigned long long, __int64
  //                 ::= _L # __int128
  //                 ::= _M # unsigned __int128
  //                 ::= _N # bool
  //                     _O # <array in parameter>
  //                 ::= _T # __float80 (Intel)
  //                 ::= _W # wchar_t
  //                 ::= _Z # __float80 (Digital Mars)
  switch (T->getKind()) {
  case BuiltinType::Void: Out << 'X'; break;
  case BuiltinType::SChar: Out << 'C'; break;
  case BuiltinType::Char_U: case BuiltinType::Char_S: Out << 'D'; break;
  case BuiltinType::UChar: Out << 'E'; break;
  case BuiltinType::Short: Out << 'F'; break;
  case BuiltinType::UShort: Out << 'G'; break;
  case BuiltinType::Int: Out << 'H'; break;
  case BuiltinType::UInt: Out << 'I'; break;
  case BuiltinType::Long: Out << 'J'; break;
  case BuiltinType::ULong: Out << 'K'; break;
  case BuiltinType::Float: Out << 'M'; break;
  case BuiltinType::Double: Out << 'N'; break;
  // TODO: Determine size and mangle accordingly
  case BuiltinType::LongDouble: Out << 'O'; break;
  // TODO: __int8 and friends
  case BuiltinType::LongLong: Out << "_J"; break;
  case BuiltinType::ULongLong: Out << "_K"; break;
  case BuiltinType::Int128: Out << "_L"; break;
  case BuiltinType::UInt128: Out << "_M"; break;
  case BuiltinType::Bool: Out << "_N"; break;
  case BuiltinType::WChar: Out << "_W"; break;

  case BuiltinType::Overload:
  case BuiltinType::Dependent:
    assert(false &&
           "Overloaded and dependent types shouldn't get to name mangling");
    break;
  case BuiltinType::UndeducedAuto:
    assert(0 && "Should not see undeduced auto here");
    break;
  case BuiltinType::ObjCId: Out << "PAUobjc_object@@"; break;
  case BuiltinType::ObjCClass: Out << "PAUobjc_class@@"; break;
  case BuiltinType::ObjCSel: Out << "PAUobjc_selector@@"; break;

  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::NullPtr:
    assert(false && "Don't know how to mangle this type");
    break;
  }
}

// <type>          ::= <function-type>
void MicrosoftCXXNameMangler::mangleType(const FunctionProtoType *T) {
  // Structors only appear in decls, so at this point we know it's not a
  // structor type.
  mangleType(T, false);
}
void MicrosoftCXXNameMangler::mangleType(const FunctionNoProtoType *T) {
  llvm_unreachable("Can't mangle K&R function prototypes");
}

void MicrosoftCXXNameMangler::mangleType(const FunctionType *T,
                                         bool IsStructor) {
  // <function-type> ::= <calling-convention> <return-type> <argument-list>
  //                                                              <throw-spec>
  mangleCallingConvention(T);

  const FunctionProtoType *Proto = cast<FunctionProtoType>(T);

  // Structors always have a 'void' return type, but MSVC mangles them as an
  // '@' (because they have no declared return type).
  if (IsStructor)
    Out << '@';
  else
    mangleType(Proto->getResultType());

  // <argument-list> ::= X # void
  //                 ::= <type>+ @
  //                 ::= <type>* Z # varargs
  if (Proto->getNumArgs() == 0 && !Proto->isVariadic()) {
    Out << 'X';
  } else {
    for (FunctionProtoType::arg_type_iterator Arg = Proto->arg_type_begin(),
         ArgEnd = Proto->arg_type_end();
         Arg != ArgEnd; ++Arg)
      mangleType(*Arg);
    
    // <builtin-type>      ::= Z  # ellipsis
    if (Proto->isVariadic())
      Out << 'Z';
    else
      Out << '@';
  }

  mangleThrowSpecification(Proto);
}

void MicrosoftCXXNameMangler::mangleFunctionClass(const FunctionDecl *FD) {
  // <function-class> ::= A # private: near
  //                  ::= B # private: far
  //                  ::= C # private: static near
  //                  ::= D # private: static far
  //                  ::= E # private: virtual near
  //                  ::= F # private: virtual far
  //                  ::= G # private: thunk near
  //                  ::= H # private: thunk far
  //                  ::= I # protected: near
  //                  ::= J # protected: far
  //                  ::= K # protected: static near
  //                  ::= L # protected: static far
  //                  ::= M # protected: virtual near
  //                  ::= N # protected: virtual far
  //                  ::= O # protected: thunk near
  //                  ::= P # protected: thunk far
  //                  ::= Q # public: near
  //                  ::= R # public: far
  //                  ::= S # public: static near
  //                  ::= T # public: static far
  //                  ::= U # public: virtual near
  //                  ::= V # public: virtual far
  //                  ::= W # public: thunk near
  //                  ::= X # public: thunk far
  //                  ::= Y # global near
  //                  ::= Z # global far
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
    switch (MD->getAccess()) {
      default:
      case AS_private:
        if (MD->isStatic())
          Out << 'C';
        else if (MD->isVirtual())
          Out << 'E';
        else
          Out << 'A';
        break;
      case AS_protected:
        if (MD->isStatic())
          Out << 'K';
        else if (MD->isVirtual())
          Out << 'M';
        else
          Out << 'I';
        break;
      case AS_public:
        if (MD->isStatic())
          Out << 'S';
        else if (MD->isVirtual())
          Out << 'U';
        else
          Out << 'Q';
    }
  } else
    Out << 'Y';
}
void MicrosoftCXXNameMangler::mangleCallingConvention(const FunctionType *T) {
  // <calling-convention> ::= A # __cdecl
  //                      ::= B # __export __cdecl
  //                      ::= C # __pascal
  //                      ::= D # __export __pascal
  //                      ::= E # __thiscall
  //                      ::= F # __export __thiscall
  //                      ::= G # __stdcall
  //                      ::= H # __export __stdcall
  //                      ::= I # __fastcall
  //                      ::= J # __export __fastcall
  // The 'export' calling conventions are from a bygone era
  // (*cough*Win16*cough*) when functions were declared for export with
  // that keyword. (It didn't actually export them, it just made them so
  // that they could be in a DLL and somebody from another module could call
  // them.)
  switch (T->getCallConv()) {
    case CC_Default:
    case CC_C: Out << 'A'; break;
    case CC_X86ThisCall: Out << 'E'; break;
    case CC_X86StdCall: Out << 'G'; break;
    case CC_X86FastCall: Out << 'I'; break;
  }
}
void MicrosoftCXXNameMangler::mangleThrowSpecification(
                                                const FunctionProtoType *FT) {
  // <throw-spec> ::= Z # throw(...) (default)
  //              ::= @ # throw() or __declspec/__attribute__((nothrow))
  //              ::= <type>+
  if (!FT->hasExceptionSpec() || FT->hasAnyExceptionSpec())
    Out << 'Z';
  else {
    for (unsigned Exception = 0, NumExceptions = FT->getNumExceptions();
         Exception < NumExceptions;
         ++Exception)
      mangleType(FT->getExceptionType(Exception).getLocalUnqualifiedType());
    Out << '@';
  }
}

void MicrosoftMangleContext::mangleName(const NamedDecl *D,
                                        llvm::SmallVectorImpl<char> &Name) {
  assert((isa<FunctionDecl>(D) || isa<VarDecl>(D)) &&
         "Invalid mangleName() call, argument is not a variable or function!");
  assert(!isa<CXXConstructorDecl>(D) && !isa<CXXDestructorDecl>(D) &&
         "Invalid mangleName() call on 'structor decl!");

  PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                 getASTContext().getSourceManager(),
                                 "Mangling declaration");

  MicrosoftCXXNameMangler Mangler(*this, Name);
  return Mangler.mangle(D);
}
void MicrosoftMangleContext::mangleThunk(const CXXMethodDecl *MD,
                                         const ThunkInfo &Thunk,
                                         llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle thunks!");
}
void MicrosoftMangleContext::mangleCXXDtorThunk(const CXXDestructorDecl *DD,
                                                CXXDtorType Type,
                                                const ThisAdjustment &,
                                                llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle destructor thunks!");
}
void MicrosoftMangleContext::mangleGuardVariable(const VarDecl *D,
                                                 llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle guard variables!");
}
void MicrosoftMangleContext::mangleCXXVTable(const CXXRecordDecl *RD,
                                             llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle virtual tables!");
}
void MicrosoftMangleContext::mangleCXXVTT(const CXXRecordDecl *RD,
                                          llvm::SmallVectorImpl<char> &) {
  llvm_unreachable("The MS C++ ABI does not have virtual table tables!");
}
void MicrosoftMangleContext::mangleCXXCtorVTable(const CXXRecordDecl *RD,
                                                 int64_t Offset,
                                                 const CXXRecordDecl *Type,
                                                 llvm::SmallVectorImpl<char> &) {
  llvm_unreachable("The MS C++ ABI does not have constructor vtables!");
}
void MicrosoftMangleContext::mangleCXXRTTI(QualType T,
                                           llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle RTTI!");
}
void MicrosoftMangleContext::mangleCXXRTTIName(QualType T,
                                               llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle RTTI names!");
}
void MicrosoftMangleContext::mangleCXXCtor(const CXXConstructorDecl *D,
                                           CXXCtorType Type,
                                           llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle constructors!");
}
void MicrosoftMangleContext::mangleCXXDtor(const CXXDestructorDecl *D,
                                           CXXDtorType Type,
                                           llvm::SmallVectorImpl<char> &) {
  assert(false && "Can't yet mangle destructors!");
}

CXXABI *clang::CodeGen::CreateMicrosoftCXXABI(CodeGenModule &CGM) {
  return new MicrosoftCXXABI(CGM);
}

