// FormatString.cpp - Common stuff for handling printf/scanf formats -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared details for processing format strings of printf and scanf
// (and friends).
//
//===----------------------------------------------------------------------===//

#include "FormatStringParsing.h"

using clang::analyze_format_string::ArgTypeResult;
using clang::analyze_format_string::FormatStringHandler;
using clang::analyze_format_string::FormatSpecifier;
using clang::analyze_format_string::LengthModifier;
using clang::analyze_format_string::OptionalAmount;
using clang::analyze_format_string::PositionContext;
using clang::analyze_format_string::ConversionSpecifier;
using namespace clang;

// Key function to FormatStringHandler.
FormatStringHandler::~FormatStringHandler() {}

//===----------------------------------------------------------------------===//
// Functions for parsing format strings components in both printf and
// scanf format strings.
//===----------------------------------------------------------------------===//

OptionalAmount
clang::analyze_format_string::ParseAmount(const char *&Beg, const char *E) {
  const char *I = Beg;
  UpdateOnReturn <const char*> UpdateBeg(Beg, I);

  unsigned accumulator = 0;
  bool hasDigits = false;

  for ( ; I != E; ++I) {
    char c = *I;
    if (c >= '0' && c <= '9') {
      hasDigits = true;
      accumulator = (accumulator * 10) + (c - '0');
      continue;
    }

    if (hasDigits)
      return OptionalAmount(OptionalAmount::Constant, accumulator, Beg, I - Beg,
          false);

    break;
  }

  return OptionalAmount();
}

OptionalAmount
clang::analyze_format_string::ParseNonPositionAmount(const char *&Beg,
                                                     const char *E,
                                                     unsigned &argIndex) {
  if (*Beg == '*') {
    ++Beg;
    return OptionalAmount(OptionalAmount::Arg, argIndex++, Beg, 0, false);
  }

  return ParseAmount(Beg, E);
}

OptionalAmount
clang::analyze_format_string::ParsePositionAmount(FormatStringHandler &H,
                                                  const char *Start,
                                                  const char *&Beg,
                                                  const char *E,
                                                  PositionContext p) {
  if (*Beg == '*') {
    const char *I = Beg + 1;
    const OptionalAmount &Amt = ParseAmount(I, E);

    if (Amt.getHowSpecified() == OptionalAmount::NotSpecified) {
      H.HandleInvalidPosition(Beg, I - Beg, p);
      return OptionalAmount(false);
    }

    if (I == E) {
      // No more characters left?
      H.HandleIncompleteSpecifier(Start, E - Start);
      return OptionalAmount(false);
    }

    assert(Amt.getHowSpecified() == OptionalAmount::Constant);

    if (*I == '$') {
      // Handle positional arguments

      // Special case: '*0$', since this is an easy mistake.
      if (Amt.getConstantAmount() == 0) {
        H.HandleZeroPosition(Beg, I - Beg + 1);
        return OptionalAmount(false);
      }

      const char *Tmp = Beg;
      Beg = ++I;

      return OptionalAmount(OptionalAmount::Arg, Amt.getConstantAmount() - 1,
                            Tmp, 0, true);
    }

    H.HandleInvalidPosition(Beg, I - Beg, p);
    return OptionalAmount(false);
  }

  return ParseAmount(Beg, E);
}


bool
clang::analyze_format_string::ParseFieldWidth(FormatStringHandler &H,
                                              FormatSpecifier &CS,
                                              const char *Start,
                                              const char *&Beg, const char *E,
                                              unsigned *argIndex) {
  // FIXME: Support negative field widths.
  if (argIndex) {
    CS.setFieldWidth(ParseNonPositionAmount(Beg, E, *argIndex));
  }
  else {
    const OptionalAmount Amt =
      ParsePositionAmount(H, Start, Beg, E,
                          analyze_format_string::FieldWidthPos);

    if (Amt.isInvalid())
      return true;
    CS.setFieldWidth(Amt);
  }
  return false;
}

bool
clang::analyze_format_string::ParseArgPosition(FormatStringHandler &H,
                                               FormatSpecifier &FS,
                                               const char *Start,
                                               const char *&Beg,
                                               const char *E) {
  const char *I = Beg;

  const OptionalAmount &Amt = ParseAmount(I, E);

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }

  if (Amt.getHowSpecified() == OptionalAmount::Constant && *(I++) == '$') {
    // Special case: '%0$', since this is an easy mistake.
    if (Amt.getConstantAmount() == 0) {
      H.HandleZeroPosition(Start, I - Start);
      return true;
    }

    FS.setArgIndex(Amt.getConstantAmount() - 1);
    FS.setUsesPositionalArg();
    // Update the caller's pointer if we decided to consume
    // these characters.
    Beg = I;
    return false;
  }

  return false;
}

bool
clang::analyze_format_string::ParseLengthModifier(FormatSpecifier &FS,
                                                  const char *&I,
                                                  const char *E) {
  LengthModifier::Kind lmKind = LengthModifier::None;
  const char *lmPosition = I;
  switch (*I) {
    default:
      return false;
    case 'h':
      ++I;
      lmKind = (I != E && *I == 'h') ?
      ++I, LengthModifier::AsChar : LengthModifier::AsShort;
      break;
    case 'l':
      ++I;
      lmKind = (I != E && *I == 'l') ?
      ++I, LengthModifier::AsLongLong : LengthModifier::AsLong;
      break;
    case 'j': lmKind = LengthModifier::AsIntMax;     ++I; break;
    case 'z': lmKind = LengthModifier::AsSizeT;      ++I; break;
    case 't': lmKind = LengthModifier::AsPtrDiff;    ++I; break;
    case 'L': lmKind = LengthModifier::AsLongDouble; ++I; break;
    case 'q': lmKind = LengthModifier::AsLongLong;   ++I; break;
  }
  LengthModifier lm(lmPosition, lmKind);
  FS.setLengthModifier(lm);
  return true;
}

//===----------------------------------------------------------------------===//
// Methods on ArgTypeResult.
//===----------------------------------------------------------------------===//

bool ArgTypeResult::matchesType(ASTContext &C, QualType argTy) const {
  switch (K) {
    case InvalidTy:
      assert(false && "ArgTypeResult must be valid");
      return true;

    case UnknownTy:
      return true;

    case SpecificTy: {
      argTy = C.getCanonicalType(argTy).getUnqualifiedType();
      if (T == argTy)
        return true;
      if (const BuiltinType *BT = argTy->getAs<BuiltinType>())
        switch (BT->getKind()) {
          default:
            break;
          case BuiltinType::Char_S:
          case BuiltinType::SChar:
            return T == C.SignedCharTy;
          case BuiltinType::Char_U:
          case BuiltinType::UChar:
            return T == C.UnsignedCharTy;
          case BuiltinType::Short:
            return T == C.UnsignedShortTy;
          case BuiltinType::UShort:
            return T == C.ShortTy;
          case BuiltinType::Int:
            return T == C.UnsignedIntTy;
          case BuiltinType::UInt:
            return T == C.IntTy;
          case BuiltinType::Long:
            return T == C.UnsignedLongTy;
          case BuiltinType::ULong:
            return T == C.LongTy;
          case BuiltinType::LongLong:
            return T == C.UnsignedLongLongTy;
          case BuiltinType::ULongLong:
            return T == C.LongLongTy;
        }
      return false;
    }

    case CStrTy: {
      const PointerType *PT = argTy->getAs<PointerType>();
      if (!PT)
        return false;
      QualType pointeeTy = PT->getPointeeType();
      if (const BuiltinType *BT = pointeeTy->getAs<BuiltinType>())
        switch (BT->getKind()) {
          case BuiltinType::Void:
          case BuiltinType::Char_U:
          case BuiltinType::UChar:
          case BuiltinType::Char_S:
          case BuiltinType::SChar:
            return true;
          default:
            break;
        }

      return false;
    }

    case WCStrTy: {
      const PointerType *PT = argTy->getAs<PointerType>();
      if (!PT)
        return false;
      QualType pointeeTy =
        C.getCanonicalType(PT->getPointeeType()).getUnqualifiedType();
      return pointeeTy == C.getWCharType();
    }
    
    case WIntTy: {
      // Instead of doing a lookup for the definition of 'wint_t' (which
      // is defined by the system headers) instead see if wchar_t and
      // the argument type promote to the same type.
      QualType PromoWChar =
        C.getWCharType()->isPromotableIntegerType() 
          ? C.getPromotedIntegerType(C.getWCharType()) : C.getWCharType();
      QualType PromoArg = 
        argTy->isPromotableIntegerType()
          ? C.getPromotedIntegerType(argTy) : argTy;
      
      PromoWChar = C.getCanonicalType(PromoWChar).getUnqualifiedType();
      PromoArg = C.getCanonicalType(PromoArg).getUnqualifiedType();
      
      return PromoWChar == PromoArg;
    }

    case CPointerTy:
      return argTy->isPointerType() || argTy->isObjCObjectPointerType() ||
        argTy->isNullPtrType();

    case ObjCPointerTy:
      return argTy->getAs<ObjCObjectPointerType>() != NULL;
  }

  // FIXME: Should be unreachable, but Clang is currently emitting
  // a warning.
  return false;
}

QualType ArgTypeResult::getRepresentativeType(ASTContext &C) const {
  switch (K) {
    case InvalidTy:
      assert(false && "No representative type for Invalid ArgTypeResult");
      // Fall-through.
    case UnknownTy:
      return QualType();
    case SpecificTy:
      return T;
    case CStrTy:
      return C.getPointerType(C.CharTy);
    case WCStrTy:
      return C.getPointerType(C.getWCharType());
    case ObjCPointerTy:
      return C.ObjCBuiltinIdTy;
    case CPointerTy:
      return C.VoidPtrTy;
    case WIntTy: {
      QualType WC = C.getWCharType();
      return WC->isPromotableIntegerType() ? C.getPromotedIntegerType(WC) : WC;
    }
  }

  // FIXME: Should be unreachable, but Clang is currently emitting
  // a warning.
  return QualType();
}

//===----------------------------------------------------------------------===//
// Methods on OptionalAmount.
//===----------------------------------------------------------------------===//

ArgTypeResult
analyze_format_string::OptionalAmount::getArgType(ASTContext &Ctx) const {
  return Ctx.IntTy;
}

//===----------------------------------------------------------------------===//
// Methods on LengthModifier.
//===----------------------------------------------------------------------===//

const char *
analyze_format_string::LengthModifier::toString() const {
  switch (kind) {
  case AsChar:
    return "hh";
  case AsShort:
    return "h";
  case AsLong: // or AsWideChar
    return "l";
  case AsLongLong:
    return "ll";
  case AsIntMax:
    return "j";
  case AsSizeT:
    return "z";
  case AsPtrDiff:
    return "t";
  case AsLongDouble:
    return "L";
  case None:
    return "";
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Methods on OptionalAmount.
//===----------------------------------------------------------------------===//

void OptionalAmount::toString(llvm::raw_ostream &os) const {
  switch (hs) {
  case Invalid:
  case NotSpecified:
    return;
  case Arg:
    if (UsesDotPrefix)
        os << ".";
    if (usesPositionalArg())
      os << "*" << getPositionalArgIndex() << "$";
    else
      os << "*";
    break;
  case Constant:
    if (UsesDotPrefix)
        os << ".";
    os << amt;
    break;
  }
}

//===----------------------------------------------------------------------===//
// Methods on ConversionSpecifier.
//===----------------------------------------------------------------------===//

bool FormatSpecifier::hasValidLengthModifier() const {
  switch (LM.getKind()) {
    case LengthModifier::None:
      return true;
      
        // Handle most integer flags
    case LengthModifier::AsChar:
    case LengthModifier::AsShort:
    case LengthModifier::AsLongLong:
    case LengthModifier::AsIntMax:
    case LengthModifier::AsSizeT:
    case LengthModifier::AsPtrDiff:
      switch (CS.getKind()) {
        case ConversionSpecifier::dArg:
        case ConversionSpecifier::iArg:
        case ConversionSpecifier::oArg:
        case ConversionSpecifier::uArg:
        case ConversionSpecifier::xArg:
        case ConversionSpecifier::XArg:
        case ConversionSpecifier::nArg:
          return true;
        default:
          return false;
      }
      
        // Handle 'l' flag
    case LengthModifier::AsLong:
      switch (CS.getKind()) {
        case ConversionSpecifier::dArg:
        case ConversionSpecifier::iArg:
        case ConversionSpecifier::oArg:
        case ConversionSpecifier::uArg:
        case ConversionSpecifier::xArg:
        case ConversionSpecifier::XArg:
        case ConversionSpecifier::aArg:
        case ConversionSpecifier::AArg:
        case ConversionSpecifier::fArg:
        case ConversionSpecifier::FArg:
        case ConversionSpecifier::eArg:
        case ConversionSpecifier::EArg:
        case ConversionSpecifier::gArg:
        case ConversionSpecifier::GArg:
        case ConversionSpecifier::nArg:
        case ConversionSpecifier::cArg:
        case ConversionSpecifier::sArg:
          return true;
        default:
          return false;
      }
      
    case LengthModifier::AsLongDouble:
      switch (CS.getKind()) {
        case ConversionSpecifier::aArg:
        case ConversionSpecifier::AArg:
        case ConversionSpecifier::fArg:
        case ConversionSpecifier::FArg:
        case ConversionSpecifier::eArg:
        case ConversionSpecifier::EArg:
        case ConversionSpecifier::gArg:
        case ConversionSpecifier::GArg:
          return true;
        default:
          return false;
      }
  }
  return false;
}


