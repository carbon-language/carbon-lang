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
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
using namespace clang;

/// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
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
  
  if (FS_inline_specified)
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
  case DeclSpec::TST_typedef:     return "typedef";
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
    return BadSpecifier( (SCS)StorageClassSpec, PrevSpec);
  StorageClassSpec = S;
  StorageClassSpecLoc = Loc;
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
    return BadSpecifier( (TSW)TypeSpecWidth, PrevSpec);
  TypeSpecWidth = W;
  TSWLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecComplex(TSC C, SourceLocation Loc, 
                                  const char *&PrevSpec) {
  if (TypeSpecComplex != TSC_unspecified)
    return BadSpecifier( (TSC)TypeSpecComplex, PrevSpec);
  TypeSpecComplex = C;
  TSCLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecSign(TSS S, SourceLocation Loc, 
                               const char *&PrevSpec) {
  if (TypeSpecSign != TSS_unspecified)
    return BadSpecifier( (TSS)TypeSpecSign, PrevSpec);
  TypeSpecSign = S;
  TSSLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec, Action::TypeTy *Rep) {
  if (TypeSpecType != TST_unspecified)
    return BadSpecifier( (TST)TypeSpecType, PrevSpec);
  TypeSpecType = T;
  TypeRep = Rep;
  TSTLoc = Loc;
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


/// Finish - This does final analysis of the declspec, rejecting things like
/// "_Imaginary" (lacking an FP type).  This returns a diagnostic to issue or
/// diag::NUM_DIAGNOSTICS if there is no error.  After calling this method,
/// DeclSpec is guaranteed self-consistent, even if an error occurred.
void DeclSpec::Finish(Diagnostic &D, SourceManager& SrcMgr, 
                      const LangOptions &Lang) {
  // Check the type specifier components first.

  // signed/unsigned are only valid with int/char/wchar_t.
  if (TypeSpecSign != TSS_unspecified) {
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // unsigned -> unsigned int, signed -> signed int.
    else if (TypeSpecType != TST_int  &&
             TypeSpecType != TST_char && TypeSpecType != TST_wchar) {
      Diag(D, TSSLoc, SrcMgr, diag::err_invalid_sign_spec,
           getSpecifierName( (TST)TypeSpecType));
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
                                      : diag::err_invalid_longlong_spec,
           getSpecifierName( (TST)TypeSpecType));
      TypeSpecType = TST_int;
    }
    break;
  case TSW_long:  // long double, long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int;  // long -> long int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_double) {
      Diag(D, TSWLoc, SrcMgr, diag::err_invalid_long_spec,
           getSpecifierName( (TST)TypeSpecType));
      TypeSpecType = TST_int;
    }
    break;
  }
  
  // TODO: if the implementation does not implement _Complex or _Imaginary,
  // disallow their use.  Need information about the backend.
  if (TypeSpecComplex != TSC_unspecified) {
    if (TypeSpecType == TST_unspecified) {
      Diag(D, TSCLoc, SrcMgr, diag::ext_plain_complex);
      TypeSpecType = TST_double;   // _Complex -> _Complex double.
    } else if (TypeSpecType == TST_int || TypeSpecType == TST_char) {
      // Note that this intentionally doesn't include _Complex _Bool.
      Diag(D, TSTLoc, SrcMgr, diag::ext_integer_complex);
    } else if (TypeSpecType != TST_float && TypeSpecType != TST_double) {
      Diag(D, TSCLoc, SrcMgr, diag::err_invalid_complex_spec, 
           getSpecifierName( (TST)TypeSpecType));
      TypeSpecComplex = TSC_unspecified;
    }
  }
  
  // Verify __thread.
  if (SCS_thread_specified) {
    if (StorageClassSpec == SCS_unspecified) {
      StorageClassSpec = SCS_extern; // '__thread int' -> 'extern __thread int'
    } else if (StorageClassSpec != SCS_extern &&
               StorageClassSpec != SCS_static) {
      Diag(D, getStorageClassSpecLoc(), SrcMgr, diag::err_invalid_thread_spec,
           getSpecifierName( (SCS)StorageClassSpec));
      SCS_thread_specified = false;
    }
  }

  // Okay, now we can infer the real type.
  
  // TODO: return "auto function" and other bad things based on the real type.
  
  // 'data definition has no type or storage class'?
}

void DeclSpec::Diag(Diagnostic &D, SourceLocation Loc, SourceManager& SrcMgr, 
                    unsigned DiagID) {
  D.Report(FullSourceLoc(Loc,SrcMgr), DiagID);
}
  
void DeclSpec::Diag(Diagnostic &D, SourceLocation Loc, SourceManager& SrcMgr,
          unsigned DiagID, const std::string &info) {
  D.Report(FullSourceLoc(Loc,SrcMgr), DiagID, &info, 1);
}
