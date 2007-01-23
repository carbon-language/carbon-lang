//===--- SemaDeclSpec.cpp - Declaration Specifier Semantic Analysis -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declaration specifiers.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
using namespace llvm;
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

static bool BadSpecifier(DeclSpec::SCS S, const char *&PrevSpec) {
  PrevSpec = DeclSpec::getSpecifierName(S);
  return true;
}

static bool BadSpecifier(DeclSpec::TSW W, const char *&PrevSpec) {
  switch (W) {
  case DeclSpec::TSW_unspecified: PrevSpec = "unspecified"; break;
  case DeclSpec::TSW_short:       PrevSpec = "short"; break;
  case DeclSpec::TSW_long:        PrevSpec = "long"; break;
  case DeclSpec::TSW_longlong:    PrevSpec = "long long"; break;
  }
  return true;
}

static bool BadSpecifier(DeclSpec::TSC C, const char *&PrevSpec) {
  switch (C) {
  case DeclSpec::TSC_unspecified: PrevSpec = "unspecified"; break;
  case DeclSpec::TSC_imaginary:   PrevSpec = "imaginary"; break;
  case DeclSpec::TSC_complex:     PrevSpec = "complex"; break;
  }
  return true;
}


static bool BadSpecifier(DeclSpec::TSS S, const char *&PrevSpec) {
  switch (S) {
  case DeclSpec::TSS_unspecified: PrevSpec = "unspecified"; break;
  case DeclSpec::TSS_signed:      PrevSpec = "signed"; break;
  case DeclSpec::TSS_unsigned:    PrevSpec = "unsigned"; break;
  }
  return true;
}

const char *DeclSpec::getSpecifierName(DeclSpec::TST T) {
  switch (T) {
  default: assert(0 && "Unknown typespec!");
  case DeclSpec::TST_unspecified: return "unspecified";
  case DeclSpec::TST_void:        return "void";
  case DeclSpec::TST_char:        return "char";
  case DeclSpec::TST_int:         return "int";
  case DeclSpec::TST_float:       return "float";
  case DeclSpec::TST_double:      return "double";
  case DeclSpec::TST_bool:        return "_Bool";
  case DeclSpec::TST_decimal32:   return "_Decimal32";
  case DeclSpec::TST_decimal64:   return "_Decimal64";
  case DeclSpec::TST_decimal128:  return "_Decimal128";
  case DeclSpec::TST_enum:        return "enum";
  case DeclSpec::TST_union:       return "union";
  case DeclSpec::TST_struct:      return "struct";
  case DeclSpec::TST_typedef:     return "typedef";
  }
}

static bool BadSpecifier(DeclSpec::TST T, const char *&PrevSpec) {
  PrevSpec = DeclSpec::getSpecifierName(T);
  return true;
}

static bool BadSpecifier(DeclSpec::TQ T, const char *&PrevSpec) {
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
    return BadSpecifier(StorageClassSpec, PrevSpec);
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
    return BadSpecifier(TypeSpecWidth, PrevSpec);
  TypeSpecWidth = W;
  TSWLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecComplex(TSC C, SourceLocation Loc, 
                                  const char *&PrevSpec) {
  if (TypeSpecComplex != TSC_unspecified)
    return BadSpecifier(TypeSpecComplex, PrevSpec);
  TypeSpecComplex = C;
  TSCLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecSign(TSS S, SourceLocation Loc, 
                               const char *&PrevSpec) {
  if (TypeSpecSign != TSS_unspecified)
    return BadSpecifier(TypeSpecSign, PrevSpec);
  TypeSpecSign = S;
  TSSLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec, void *TypeRep) {
  if (TypeSpecType != TST_unspecified)
    return BadSpecifier(TypeSpecType, PrevSpec);
  TypeSpecType = T;
  TypenameRep = TypeRep;
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
void DeclSpec::Finish(Diagnostic &D, const LangOptions &Lang) {
  // Check the type specifier components first.

  // signed/unsigned are only valid with int/char.
  if (TypeSpecSign != TSS_unspecified) {
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // unsigned -> unsigned int, signed -> signed int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_char) {
      D.Report(TSSLoc, diag::err_invalid_sign_spec,
               getSpecifierName(TypeSpecType));
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
      D.Report(TSWLoc,
               TypeSpecWidth == TSW_short ? diag::err_invalid_short_spec
                                          : diag::err_invalid_longlong_spec,
               getSpecifierName(TypeSpecType));
      TypeSpecType = TST_int;
    }
    break;
  case TSW_long:  // long double, long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int;  // long -> long int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_double) {
      D.Report(TSWLoc, diag::err_invalid_long_spec,
               getSpecifierName(TypeSpecType));
      TypeSpecType = TST_int;
    }
    break;
  }
  
  // TODO: if the implementation does not implement _Complex or _Imaginary,
  // disallow their use.  Need information about the backend.
  if (TypeSpecComplex != TSC_unspecified) {
    if (TypeSpecType == TST_unspecified) {
      D.Report(TSCLoc, diag::ext_plain_complex);
      TypeSpecType = TST_double;   // _Complex -> _Complex double.
    } else if (TypeSpecType == TST_int || TypeSpecType == TST_char) {
      // Note that this intentionally doesn't include _Complex _Bool.
      D.Report(TSTLoc, diag::ext_integer_complex);
    } else if (TypeSpecType != TST_float && TypeSpecType != TST_double) {
      D.Report(TSCLoc, diag::err_invalid_complex_spec, 
               getSpecifierName(TypeSpecType));
      TypeSpecComplex = TSC_unspecified;
    }
  }
  
  // Verify __thread.
  if (SCS_thread_specified) {
    if (StorageClassSpec == SCS_unspecified) {
      StorageClassSpec = SCS_extern; // '__thread int' -> 'extern __thread int'
    } else if (StorageClassSpec != SCS_extern &&
               StorageClassSpec != SCS_static) {
      D.Report(getStorageClassSpecLoc(), diag::err_invalid_thread_spec,
               getSpecifierName(StorageClassSpec));
      SCS_thread_specified = false;
    }
  }

  // Okay, now we can infer the real type.
  // TODO: infer real type.
  
  // TODO: return "auto function" and other bad things based on the real type.
  
  // 'data definition has no type or storage class'?
}
