//===--- Declarations.cpp - Declaration Representation Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Declaration representation classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Declarations.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
using namespace llvm;
using namespace clang;

/// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
///
unsigned DeclSpec::getParsedSpecifiers() const {
  unsigned Res = 0;
  if (StorageClassSpec != SCS_unspecified)
    Res |= PQ_StorageClassSpecifier;
  
  if (TypeQualifiers   != TQ_unspecified)
    Res |= PQ_TypeQualifier;
  
  if (TypeSpecWidth    != TSW_unspecified ||
      TypeSpecComplex  != TSC_unspecified ||
      TypeSpecSign     != TSS_unspecified ||
      TypeSpecType     != TST_unspecified)
    Res |= PQ_TypeSpecifier;
  
  if (FuncSpec         != FS_unspecified)
    Res |= PQ_FunctionSpecifier;
  return Res;
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

static const char *getSpecifierName(DeclSpec::TST T) {
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
  }
}

static bool BadSpecifier(DeclSpec::TST T, const char *&PrevSpec) {
  PrevSpec = getSpecifierName(T);
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

/// These methods set the specified attribute of the DeclSpec, but return true
/// and ignore the request if invalid (e.g. "extern" then "auto" is
/// specified).
bool DeclSpec::SetTypeSpecWidth(TSW W, const char *&PrevSpec) {
  if (TypeSpecWidth != TSW_unspecified)
    return BadSpecifier(TypeSpecWidth, PrevSpec);
  TypeSpecWidth = W;
  return false;
}

bool DeclSpec::SetTypeSpecComplex(TSC C, const char *&PrevSpec) {
  if (TypeSpecComplex != TSC_unspecified)
    return BadSpecifier(TypeSpecComplex, PrevSpec);
  TypeSpecComplex = C;
  return false;
}

bool DeclSpec::SetTypeSpecSign(TSS S, const char *&PrevSpec) {
  if (TypeSpecSign != TSS_unspecified)
    return BadSpecifier(TypeSpecSign, PrevSpec);
  TypeSpecSign = S;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, const char *&PrevSpec) {
  if (TypeSpecType != TST_unspecified)
    return BadSpecifier(TypeSpecType, PrevSpec);
  TypeSpecType = T;
  return false;
}

bool DeclSpec::SetTypeQual(TQ T, const char *&PrevSpec,
                           const LangOptions &Lang) {
  // Duplicates turn into warnings pre-C99.
  if ((TypeQualifiers & T) && !Lang.C99)
    return BadSpecifier(T, PrevSpec);
  TypeQualifiers |= T;
  return false;
}

bool DeclSpec::SetFuncSpec(FS F, const char *&PrevSpec) {
  // 'inline inline' is ok.
  FuncSpec = F;
  return false;
}

/// Finish - This does final analysis of the declspec, rejecting things like
/// "_Imaginary" (lacking an FP type).  This returns a diagnostic to issue or
/// diag::NUM_DIAGNOSTICS if there is no error.  After calling this method,
/// DeclSpec is guaranteed self-consistent, even if an error occurred.
void DeclSpec::Finish(SourceLocation Loc, Diagnostic &D,
                      const LangOptions &Lang) {
  // Check the type specifier components first.

  // signed/unsigned are only valid with int/char.
  if (TypeSpecSign != TSS_unspecified) {
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // unsigned -> unsigned int, signed -> signed int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_char) {
      D.Report(Loc, diag::err_invalid_sign_spec,getSpecifierName(TypeSpecType));
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
      D.Report(Loc, TypeSpecWidth == TSW_short ? diag::err_invalid_short_spec :
               diag::err_invalid_longlong_spec,
               getSpecifierName(TypeSpecType));
      TypeSpecType = TST_int;
    }
    break;
  case TSW_long:  // long double, long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int;  // long -> long int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_double) {
      D.Report(Loc, diag::err_invalid_long_spec,
               getSpecifierName(TypeSpecType));
      TypeSpecType = TST_int;
    }
    break;
  }
  
  // FIXME: if the implementation does not implement _Complex or _Imaginary,
  // disallow their use.  Need information about the backend.
  if (TypeSpecComplex != TSC_unspecified) {
    if (TypeSpecType == TST_unspecified) {
      D.Report(Loc, diag::ext_plain_complex);
      TypeSpecType = TST_double;   // _Complex -> _Complex double.
    } else if (TypeSpecType == TST_int || TypeSpecType == TST_char) {
      // Note that GCC doesn't support _Complex _Bool.
      D.Report(Loc, diag::ext_integer_complex);
    } else if (TypeSpecType != TST_float && TypeSpecType != TST_double) {
      D.Report(Loc, diag::err_invalid_complex_spec, 
               getSpecifierName(TypeSpecType));
      TypeSpecComplex = TSC_unspecified;
    }
  }
}
