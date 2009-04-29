//===--- SemaDeclSpec.cpp - Declaration Specifier Semantic Analysis -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declaration specifiers.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLExtras.h"
#include <cstring>
using namespace clang;


static DiagnosticBuilder Diag(Diagnostic &D, SourceLocation Loc,
                              SourceManager &SrcMgr, unsigned DiagID) {
  return D.Report(FullSourceLoc(Loc, SrcMgr), DiagID);
}

/// DeclaratorChunk::getFunction - Return a DeclaratorChunk for a function.
/// "TheDeclarator" is the declarator that this will be added to.
DeclaratorChunk DeclaratorChunk::getFunction(bool hasProto, bool isVariadic,
                                             SourceLocation EllipsisLoc,
                                             ParamInfo *ArgInfo,
                                             unsigned NumArgs,
                                             unsigned TypeQuals,
                                             bool hasExceptionSpec,
                                             bool hasAnyExceptionSpec,
                                             ActionBase::TypeTy **Exceptions,
                                             unsigned NumExceptions,
                                             SourceLocation Loc,
                                             Declarator &TheDeclarator) {
  DeclaratorChunk I;
  I.Kind                 = Function;
  I.Loc                  = Loc;
  I.Fun.hasPrototype     = hasProto;
  I.Fun.isVariadic       = isVariadic;
  I.Fun.EllipsisLoc      = EllipsisLoc.getRawEncoding();
  I.Fun.DeleteArgInfo    = false;
  I.Fun.TypeQuals        = TypeQuals;
  I.Fun.NumArgs          = NumArgs;
  I.Fun.ArgInfo          = 0;
  I.Fun.hasExceptionSpec = hasExceptionSpec;
  I.Fun.hasAnyExceptionSpec = hasAnyExceptionSpec;
  I.Fun.NumExceptions    = NumExceptions;
  I.Fun.Exceptions       = 0;

  // new[] an argument array if needed.
  if (NumArgs) {
    // If the 'InlineParams' in Declarator is unused and big enough, put our
    // parameter list there (in an effort to avoid new/delete traffic).  If it
    // is already used (consider a function returning a function pointer) or too
    // small (function taking too many arguments), go to the heap.
    if (!TheDeclarator.InlineParamsUsed && 
        NumArgs <= llvm::array_lengthof(TheDeclarator.InlineParams)) {
      I.Fun.ArgInfo = TheDeclarator.InlineParams;
      I.Fun.DeleteArgInfo = false;
      TheDeclarator.InlineParamsUsed = true;
    } else {
      I.Fun.ArgInfo = new DeclaratorChunk::ParamInfo[NumArgs];
      I.Fun.DeleteArgInfo = true;
    }
    memcpy(I.Fun.ArgInfo, ArgInfo, sizeof(ArgInfo[0])*NumArgs);
  }
  // new[] an exception array if needed
  if (NumExceptions) {
    I.Fun.Exceptions = new ActionBase::TypeTy*[NumExceptions];
    memcpy(I.Fun.Exceptions, Exceptions,
           sizeof(ActionBase::TypeTy*)*NumExceptions);
  }
  return I;
}

/// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
/// declaration specifier includes.
///
unsigned DeclSpec::getParsedSpecifiers() const {
  unsigned Res = 0;
  if (StorageClassSpec != SCS_unspecified ||
      SCS_thread_specified)
    Res |= PQ_StorageClassSpecifier;

  if (TypeQualifiers != TQ_unspecified)
    Res |= PQ_TypeQualifier;
  
  if (hasTypeSpecifier())
    Res |= PQ_TypeSpecifier;
  
  if (FS_inline_specified || FS_virtual_specified || FS_explicit_specified)
    Res |= PQ_FunctionSpecifier;
  return Res;
}

const char *DeclSpec::getSpecifierName(DeclSpec::SCS S) {
  switch (S) {
  default: assert(0 && "Unknown typespec!");
  case DeclSpec::SCS_unspecified: return "unspecified";
  case DeclSpec::SCS_typedef:     return "typedef";
  case DeclSpec::SCS_extern:      return "extern";
  case DeclSpec::SCS_static:      return "static";
  case DeclSpec::SCS_auto:        return "auto";
  case DeclSpec::SCS_register:    return "register";
  case DeclSpec::SCS_private_extern: return "__private_extern__";
  case DeclSpec::SCS_mutable:     return "mutable";
  }
}

bool DeclSpec::BadSpecifier(SCS S, const char *&PrevSpec) {
  PrevSpec = getSpecifierName(S);
  return true;
}

bool DeclSpec::BadSpecifier(TSW W, const char *&PrevSpec) {
  switch (W) {
  case TSW_unspecified: PrevSpec = "unspecified"; break;
  case TSW_short:       PrevSpec = "short"; break;
  case TSW_long:        PrevSpec = "long"; break;
  case TSW_longlong:    PrevSpec = "long long"; break;
  }
  return true;
}

bool DeclSpec::BadSpecifier(TSC C, const char *&PrevSpec) {
  switch (C) {
  case TSC_unspecified: PrevSpec = "unspecified"; break;
  case TSC_imaginary:   PrevSpec = "imaginary"; break;
  case TSC_complex:     PrevSpec = "complex"; break;
  }
  return true;
}


bool DeclSpec::BadSpecifier(TSS S, const char *&PrevSpec) {
  switch (S) {
  case TSS_unspecified: PrevSpec = "unspecified"; break;
  case TSS_signed:      PrevSpec = "signed"; break;
  case TSS_unsigned:    PrevSpec = "unsigned"; break;
  }
  return true;
}

const char *DeclSpec::getSpecifierName(DeclSpec::TST T) {
  switch (T) {
  default: assert(0 && "Unknown typespec!");
  case DeclSpec::TST_unspecified: return "unspecified";
  case DeclSpec::TST_void:        return "void";
  case DeclSpec::TST_char:        return "char";
  case DeclSpec::TST_wchar:       return "wchar_t";
  case DeclSpec::TST_int:         return "int";
  case DeclSpec::TST_float:       return "float";
  case DeclSpec::TST_double:      return "double";
  case DeclSpec::TST_bool:        return "_Bool";
  case DeclSpec::TST_decimal32:   return "_Decimal32";
  case DeclSpec::TST_decimal64:   return "_Decimal64";
  case DeclSpec::TST_decimal128:  return "_Decimal128";
  case DeclSpec::TST_enum:        return "enum";
  case DeclSpec::TST_class:       return "class";
  case DeclSpec::TST_union:       return "union";
  case DeclSpec::TST_struct:      return "struct";
  case DeclSpec::TST_typename:    return "type-name";
  case DeclSpec::TST_typeofType:
  case DeclSpec::TST_typeofExpr:  return "typeof";
  }
}

bool DeclSpec::BadSpecifier(TST T, const char *&PrevSpec) {
  PrevSpec = getSpecifierName(T);
  return true;
}

bool DeclSpec::BadSpecifier(TQ T, const char *&PrevSpec) {
  switch (T) {
  case DeclSpec::TQ_unspecified: PrevSpec = "unspecified"; break;
  case DeclSpec::TQ_const:       PrevSpec = "const"; break;
  case DeclSpec::TQ_restrict:    PrevSpec = "restrict"; break;
  case DeclSpec::TQ_volatile:    PrevSpec = "volatile"; break;
  }
  return true;
}

bool DeclSpec::SetStorageClassSpec(SCS S, SourceLocation Loc,
                                   const char *&PrevSpec) {
  if (StorageClassSpec != SCS_unspecified)
    return BadSpecifier((SCS)StorageClassSpec, PrevSpec);
  StorageClassSpec = S;
  StorageClassSpecLoc = Loc;
  assert((unsigned)S == StorageClassSpec && "SCS constants overflow bitfield");
  return false;
}

bool DeclSpec::SetStorageClassSpecThread(SourceLocation Loc, 
                                         const char *&PrevSpec) {
  if (SCS_thread_specified) {
    PrevSpec = "__thread";
    return true;
  }
  SCS_thread_specified = true;
  SCS_threadLoc = Loc;
  return false;
}


/// These methods set the specified attribute of the DeclSpec, but return true
/// and ignore the request if invalid (e.g. "extern" then "auto" is
/// specified).
bool DeclSpec::SetTypeSpecWidth(TSW W, SourceLocation Loc,
                                const char *&PrevSpec) {
  if (TypeSpecWidth != TSW_unspecified &&
      // Allow turning long -> long long.
      (W != TSW_longlong || TypeSpecWidth != TSW_long))
    return BadSpecifier((TSW)TypeSpecWidth, PrevSpec);
  TypeSpecWidth = W;
  TSWLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecComplex(TSC C, SourceLocation Loc, 
                                  const char *&PrevSpec) {
  if (TypeSpecComplex != TSC_unspecified)
    return BadSpecifier((TSC)TypeSpecComplex, PrevSpec);
  TypeSpecComplex = C;
  TSCLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecSign(TSS S, SourceLocation Loc, 
                               const char *&PrevSpec) {
  if (TypeSpecSign != TSS_unspecified)
    return BadSpecifier((TSS)TypeSpecSign, PrevSpec);
  TypeSpecSign = S;
  TSSLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec, void *Rep) {
  if (TypeSpecType != TST_unspecified)
    return BadSpecifier((TST)TypeSpecType, PrevSpec);
  TypeSpecType = T;
  TypeRep = Rep;
  TSTLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecError() {
  TypeSpecType = TST_error;
  TypeRep = 0;
  TSTLoc = SourceLocation();
  return false;
}

bool DeclSpec::SetTypeQual(TQ T, SourceLocation Loc, const char *&PrevSpec,
                           const LangOptions &Lang) {
  // Duplicates turn into warnings pre-C99.
  if ((TypeQualifiers & T) && !Lang.C99)
    return BadSpecifier(T, PrevSpec);
  TypeQualifiers |= T;
  
  switch (T) {
  default: assert(0 && "Unknown type qualifier!");
  case TQ_const:    TQ_constLoc = Loc; break;
  case TQ_restrict: TQ_restrictLoc = Loc; break;
  case TQ_volatile: TQ_volatileLoc = Loc; break;
  }
  return false;
}

bool DeclSpec::SetFunctionSpecInline(SourceLocation Loc, const char *&PrevSpec){
  // 'inline inline' is ok.
  FS_inline_specified = true;
  FS_inlineLoc = Loc;
  return false;
}

bool DeclSpec::SetFunctionSpecVirtual(SourceLocation Loc, const char *&PrevSpec){
  // 'virtual virtual' is ok.
  FS_virtual_specified = true;
  FS_virtualLoc = Loc;
  return false;
}

bool DeclSpec::SetFunctionSpecExplicit(SourceLocation Loc, const char *&PrevSpec){
  // 'explicit explicit' is ok.
  FS_explicit_specified = true;
  FS_explicitLoc = Loc;
  return false;
}


/// Finish - This does final analysis of the declspec, rejecting things like
/// "_Imaginary" (lacking an FP type).  This returns a diagnostic to issue or
/// diag::NUM_DIAGNOSTICS if there is no error.  After calling this method,
/// DeclSpec is guaranteed self-consistent, even if an error occurred.
void DeclSpec::Finish(Diagnostic &D, Preprocessor &PP) {
  // Check the type specifier components first.
  SourceManager &SrcMgr = PP.getSourceManager();

  // signed/unsigned are only valid with int/char/wchar_t.
  if (TypeSpecSign != TSS_unspecified) {
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // unsigned -> unsigned int, signed -> signed int.
    else if (TypeSpecType != TST_int  &&
             TypeSpecType != TST_char && TypeSpecType != TST_wchar) {
      Diag(D, TSSLoc, SrcMgr, diag::err_invalid_sign_spec)
        << getSpecifierName((TST)TypeSpecType);
      // signed double -> double.
      TypeSpecSign = TSS_unspecified;
    }
  }

  // Validate the width of the type.
  switch (TypeSpecWidth) {
  case TSW_unspecified: break;
  case TSW_short:    // short int
  case TSW_longlong: // long long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // short -> short int, long long -> long long int.
    else if (TypeSpecType != TST_int) {
      Diag(D, TSWLoc, SrcMgr,
           TypeSpecWidth == TSW_short ? diag::err_invalid_short_spec
                                      : diag::err_invalid_longlong_spec)
        <<  getSpecifierName((TST)TypeSpecType);
      TypeSpecType = TST_int;
    }
    break;
  case TSW_long:  // long double, long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int;  // long -> long int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_double) {
      Diag(D, TSWLoc, SrcMgr, diag::err_invalid_long_spec)
        << getSpecifierName((TST)TypeSpecType);
      TypeSpecType = TST_int;
    }
    break;
  }
  
  // TODO: if the implementation does not implement _Complex or _Imaginary,
  // disallow their use.  Need information about the backend.
  if (TypeSpecComplex != TSC_unspecified) {
    if (TypeSpecType == TST_unspecified) {
      Diag(D, TSCLoc, SrcMgr, diag::ext_plain_complex)
        << CodeModificationHint::CreateInsertion(
                              PP.getLocForEndOfToken(getTypeSpecComplexLoc()),
                                                 " double");
      TypeSpecType = TST_double;   // _Complex -> _Complex double.
    } else if (TypeSpecType == TST_int || TypeSpecType == TST_char) {
      // Note that this intentionally doesn't include _Complex _Bool.
      Diag(D, TSTLoc, SrcMgr, diag::ext_integer_complex);
    } else if (TypeSpecType != TST_float && TypeSpecType != TST_double) {
      Diag(D, TSCLoc, SrcMgr, diag::err_invalid_complex_spec)
        << getSpecifierName((TST)TypeSpecType);
      TypeSpecComplex = TSC_unspecified;
    }
  }

  // Okay, now we can infer the real type.
  
  // TODO: return "auto function" and other bad things based on the real type.
  
  // 'data definition has no type or storage class'?
}

bool DeclSpec::isMissingDeclaratorOk() {
  TST tst = getTypeSpecType();
  return (tst == TST_union
       || tst == TST_struct
       || tst == TST_class
       || tst == TST_enum
          ) && getTypeRep() != 0 && StorageClassSpec != DeclSpec::SCS_typedef;
}
