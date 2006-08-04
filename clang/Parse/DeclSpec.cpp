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

static bool BadSpecifier(DeclSpec::TST T, const char *&PrevSpec) {
  switch (T) {
  case DeclSpec::TST_unspecified: PrevSpec = "unspecified"; break;
  case DeclSpec::TST_void:        PrevSpec = "void"; break;
  case DeclSpec::TST_char:        PrevSpec = "char"; break;
  case DeclSpec::TST_int:         PrevSpec = "int"; break;
  case DeclSpec::TST_float:       PrevSpec = "float"; break;
  case DeclSpec::TST_double:      PrevSpec = "double"; break;
  case DeclSpec::TST_bool:        PrevSpec = "_Bool"; break;
  case DeclSpec::TST_decimal32:   PrevSpec = "_Decimal32"; break;
  case DeclSpec::TST_decimal64:   PrevSpec = "_Decimal64"; break;
  case DeclSpec::TST_decimal128:  PrevSpec = "_Decimal128"; break;
  }
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
diag::kind DeclSpec::Finish(const LangOptions &Lang) {
  // FIXME: implement this.
  
  return diag::NUM_DIAGNOSTICS;
}
