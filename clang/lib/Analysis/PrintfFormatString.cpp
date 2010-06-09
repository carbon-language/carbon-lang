//= PrintfFormatStrings.cpp - Analysis of printf format strings --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Handling of format string in printf and friends.  The structure of format
// strings for fprintf() are described in C99 7.19.6.1.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/PrintfFormatString.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "llvm/Support/raw_ostream.h"

using clang::analyze_printf::ArgTypeResult;
using clang::analyze_printf::FormatSpecifier;
using clang::analyze_printf::FormatStringHandler;
using clang::analyze_printf::OptionalAmount;
using clang::analyze_printf::PositionContext;
using clang::analyze_printf::ConversionSpecifier;
using clang::analyze_printf::LengthModifier;

using namespace clang;

namespace {
class FormatSpecifierResult {
  FormatSpecifier FS;
  const char *Start;
  bool Stop;
public:
  FormatSpecifierResult(bool stop = false)
    : Start(0), Stop(stop) {}
  FormatSpecifierResult(const char *start,
                        const FormatSpecifier &fs)
    : FS(fs), Start(start), Stop(false) {}


  const char *getStart() const { return Start; }
  bool shouldStop() const { return Stop; }
  bool hasValue() const { return Start != 0; }
  const FormatSpecifier &getValue() const {
    assert(hasValue());
    return FS;
  }
  const FormatSpecifier &getValue() { return FS; }
};
} // end anonymous namespace

template <typename T>
class UpdateOnReturn {
  T &ValueToUpdate;
  const T &ValueToCopy;
public:
  UpdateOnReturn(T &valueToUpdate, const T &valueToCopy)
    : ValueToUpdate(valueToUpdate), ValueToCopy(valueToCopy) {}

  ~UpdateOnReturn() {
    ValueToUpdate = ValueToCopy;
  }
};

//===----------------------------------------------------------------------===//
// Methods for parsing format strings.
//===----------------------------------------------------------------------===//

static OptionalAmount ParseAmount(const char *&Beg, const char *E) {
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
      return OptionalAmount(OptionalAmount::Constant, accumulator, Beg, 0);

    break;
  }

  return OptionalAmount();
}

static OptionalAmount ParseNonPositionAmount(const char *&Beg, const char *E,
                                             unsigned &argIndex) {
  if (*Beg == '*') {
    ++Beg;
    return OptionalAmount(OptionalAmount::Arg, argIndex++, Beg, 0);
  }

  return ParseAmount(Beg, E);
}

static OptionalAmount ParsePositionAmount(FormatStringHandler &H,
                                          const char *Start,
                                          const char *&Beg, const char *E,
                                          PositionContext p) {
  if (*Beg == '*') {
    const char *I = Beg + 1;
    const OptionalAmount &Amt = ParseAmount(I, E);

    if (Amt.getHowSpecified() == OptionalAmount::NotSpecified) {
      H.HandleInvalidPosition(Beg, I - Beg, p);
      return OptionalAmount(false);
    }

    if (I== E) {
      // No more characters left?
      H.HandleIncompleteFormatSpecifier(Start, E - Start);
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
                            Tmp, 1);
    }

    H.HandleInvalidPosition(Beg, I - Beg, p);
    return OptionalAmount(false);
  }

  return ParseAmount(Beg, E);
}

static bool ParsePrecision(FormatStringHandler &H, FormatSpecifier &FS,
                           const char *Start, const char *&Beg, const char *E,
                           unsigned *argIndex) {
  if (argIndex) {
    FS.setPrecision(ParseNonPositionAmount(Beg, E, *argIndex));
  }
  else {
    const OptionalAmount Amt = ParsePositionAmount(H, Start, Beg, E,
                                                  analyze_printf::PrecisionPos);
    if (Amt.isInvalid())
      return true;
    FS.setPrecision(Amt);
  }
  return false;
}

static bool ParseFieldWidth(FormatStringHandler &H, FormatSpecifier &FS,
                            const char *Start, const char *&Beg, const char *E,
                            unsigned *argIndex) {
  // FIXME: Support negative field widths.
  if (argIndex) {
    FS.setFieldWidth(ParseNonPositionAmount(Beg, E, *argIndex));
  }
  else {
    const OptionalAmount Amt = ParsePositionAmount(H, Start, Beg, E,
                                                 analyze_printf::FieldWidthPos);
    if (Amt.isInvalid())
      return true;
    FS.setFieldWidth(Amt);
  }
  return false;
}


static bool ParseArgPosition(FormatStringHandler &H,
                             FormatSpecifier &FS, const char *Start,
                             const char *&Beg, const char *E) {

  using namespace clang::analyze_printf;
  const char *I = Beg;

  const OptionalAmount &Amt = ParseAmount(I, E);

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteFormatSpecifier(Start, E - Start);
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

static FormatSpecifierResult ParseFormatSpecifier(FormatStringHandler &H,
                                                  const char *&Beg,
                                                  const char *E,
                                                  unsigned &argIndex) {

  using namespace clang::analyze_printf;

  const char *I = Beg;
  const char *Start = 0;
  UpdateOnReturn <const char*> UpdateBeg(Beg, I);

  // Look for a '%' character that indicates the start of a format specifier.
  for ( ; I != E ; ++I) {
    char c = *I;
    if (c == '\0') {
      // Detect spurious null characters, which are likely errors.
      H.HandleNullChar(I);
      return true;
    }
    if (c == '%') {
      Start = I++;  // Record the start of the format specifier.
      break;
    }
  }

  // No format specifier found?
  if (!Start)
    return false;

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteFormatSpecifier(Start, E - Start);
    return true;
  }

  FormatSpecifier FS;
  if (ParseArgPosition(H, FS, Start, I, E))
    return true;

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteFormatSpecifier(Start, E - Start);
    return true;
  }

  // Look for flags (if any).
  bool hasMore = true;
  for ( ; I != E; ++I) {
    switch (*I) {
      default: hasMore = false; break;
      case '-': FS.setIsLeftJustified(); break;
      case '+': FS.setHasPlusPrefix(); break;
      case ' ': FS.setHasSpacePrefix(); break;
      case '#': FS.setHasAlternativeForm(); break;
      case '0': FS.setHasLeadingZeros(); break;
    }
    if (!hasMore)
      break;
  }

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteFormatSpecifier(Start, E - Start);
    return true;
  }

  // Look for the field width (if any).
  if (ParseFieldWidth(H, FS, Start, I, E,
                      FS.usesPositionalArg() ? 0 : &argIndex))
    return true;

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteFormatSpecifier(Start, E - Start);
    return true;
  }

  // Look for the precision (if any).
  if (*I == '.') {
    ++I;
    if (I == E) {
      H.HandleIncompleteFormatSpecifier(Start, E - Start);
      return true;
    }

    if (ParsePrecision(H, FS, Start, I, E,
                       FS.usesPositionalArg() ? 0 : &argIndex))
      return true;

    if (I == E) {
      // No more characters left?
      H.HandleIncompleteFormatSpecifier(Start, E - Start);
      return true;
    }
  }

  // Look for the length modifier.
  LengthModifier::Kind lmKind = LengthModifier::None;
  const char *lmPosition = I;
  switch (*I) {
    default:
      break;
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

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteFormatSpecifier(Start, E - Start);
    return true;
  }

  if (*I == '\0') {
    // Detect spurious null characters, which are likely errors.
    H.HandleNullChar(I);
    return true;
  }

  // Finally, look for the conversion specifier.
  const char *conversionPosition = I++;
  ConversionSpecifier::Kind k = ConversionSpecifier::InvalidSpecifier;
  switch (*conversionPosition) {
    default:
      break;
    // C99: 7.19.6.1 (section 8).
    case '%': k = ConversionSpecifier::PercentArg;   break;
    case 'A': k = ConversionSpecifier::AArg; break;
    case 'E': k = ConversionSpecifier::EArg; break;
    case 'F': k = ConversionSpecifier::FArg; break;
    case 'G': k = ConversionSpecifier::GArg; break;
    case 'X': k = ConversionSpecifier::XArg; break;
    case 'a': k = ConversionSpecifier::aArg; break;
    case 'c': k = ConversionSpecifier::IntAsCharArg; break;
    case 'd': k = ConversionSpecifier::dArg; break;
    case 'e': k = ConversionSpecifier::eArg; break;
    case 'f': k = ConversionSpecifier::fArg; break;
    case 'g': k = ConversionSpecifier::gArg; break;
    case 'i': k = ConversionSpecifier::iArg; break;
    case 'n': k = ConversionSpecifier::OutIntPtrArg; break;
    case 'o': k = ConversionSpecifier::oArg; break;
    case 'p': k = ConversionSpecifier::VoidPtrArg;   break;
    case 's': k = ConversionSpecifier::CStrArg;      break;
    case 'u': k = ConversionSpecifier::uArg; break;
    case 'x': k = ConversionSpecifier::xArg; break;
    // Mac OS X (unicode) specific
    case 'C': k = ConversionSpecifier::CArg; break;
    case 'S': k = ConversionSpecifier::UnicodeStrArg; break;
    // Objective-C.
    case '@': k = ConversionSpecifier::ObjCObjArg; break;
    // Glibc specific.
    case 'm': k = ConversionSpecifier::PrintErrno; break;
  }
  ConversionSpecifier CS(conversionPosition, k);
  FS.setConversionSpecifier(CS);
  if (CS.consumesDataArgument() && !FS.usesPositionalArg())
    FS.setArgIndex(argIndex++);

  if (k == ConversionSpecifier::InvalidSpecifier) {
    // Assume the conversion takes one argument.
    return !H.HandleInvalidConversionSpecifier(FS, Beg, I - Beg);
  }
  return FormatSpecifierResult(Start, FS);
}

bool clang::analyze_printf::ParseFormatString(FormatStringHandler &H,
                       const char *I, const char *E) {

  unsigned argIndex = 0;

  // Keep looking for a format specifier until we have exhausted the string.
  while (I != E) {
    const FormatSpecifierResult &FSR = ParseFormatSpecifier(H, I, E, argIndex);
    // Did a fail-stop error of any kind occur when parsing the specifier?
    // If so, don't do any more processing.
    if (FSR.shouldStop())
      return true;;
    // Did we exhaust the string or encounter an error that
    // we can recover from?
    if (!FSR.hasValue())
      continue;
    // We have a format specifier.  Pass it to the callback.
    if (!H.HandleFormatSpecifier(FSR.getValue(), FSR.getStart(),
                                 I - FSR.getStart()))
      return true;
  }
  assert(I == E && "Format string not exhausted");
  return false;
}

FormatStringHandler::~FormatStringHandler() {}

//===----------------------------------------------------------------------===//
// Methods on ArgTypeResult.
//===----------------------------------------------------------------------===//

bool ArgTypeResult::matchesType(ASTContext &C, QualType argTy) const {
  assert(isValid());

  if (K == UnknownTy)
    return true;

  if (K == SpecificTy) {
    argTy = C.getCanonicalType(argTy).getUnqualifiedType();

    if (T == argTy)
      return true;

    if (const BuiltinType *BT = argTy->getAs<BuiltinType>())
      switch (BT->getKind()) {
        default:
          break;
        case BuiltinType::Char_S:
        case BuiltinType::SChar:
          return T == C.UnsignedCharTy;
        case BuiltinType::Char_U:
        case BuiltinType::UChar:
          return T == C.SignedCharTy;
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

  if (K == CStrTy) {
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

  if (K == WCStrTy) {
    const PointerType *PT = argTy->getAs<PointerType>();
    if (!PT)
      return false;

    QualType pointeeTy =
      C.getCanonicalType(PT->getPointeeType()).getUnqualifiedType();

    return pointeeTy == C.getWCharType();
  }

  return false;
}

QualType ArgTypeResult::getRepresentativeType(ASTContext &C) const {
  assert(isValid());
  if (K == SpecificTy)
    return T;
  if (K == CStrTy)
    return C.getPointerType(C.CharTy);
  if (K == WCStrTy)
    return C.getPointerType(C.getWCharType());
  if (K == ObjCPointerTy)
    return C.ObjCBuiltinIdTy;

  return QualType();
}

//===----------------------------------------------------------------------===//
// Methods on OptionalAmount.
//===----------------------------------------------------------------------===//

ArgTypeResult OptionalAmount::getArgType(ASTContext &Ctx) const {
  return Ctx.IntTy;
}

//===----------------------------------------------------------------------===//
// Methods on ConversionSpecifier.
//===----------------------------------------------------------------------===//
const char *ConversionSpecifier::toString() const {
  switch (kind) {
  case dArg: return "d";
  case iArg: return "i";
  case oArg: return "o";
  case uArg: return "u";
  case xArg: return "x";
  case XArg: return "X";
  case fArg: return "f";
  case FArg: return "F";
  case eArg: return "e";
  case EArg: return "E";
  case gArg: return "g";
  case GArg: return "G";
  case aArg: return "a";
  case AArg: return "A";
  case IntAsCharArg:     return "c";
  case CStrArg:          return "s";
  case VoidPtrArg:       return "p";
  case OutIntPtrArg:     return "n";
  case PercentArg:       return "%";
  case InvalidSpecifier: return NULL;

  // MacOS X unicode extensions.
  case CArg:          return "C";
  case UnicodeStrArg: return "S";

  // Objective-C specific specifiers.
  case ObjCObjArg: return "@";

  // GlibC specific specifiers.
  case PrintErrno: return "m";
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Methods on LengthModifier.
//===----------------------------------------------------------------------===//

const char *LengthModifier::toString() const {
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
    if (usesPositionalArg())
      os << ".*" << getPositionalArgIndex() << "$";
    else
      os << ".*";
    break;
  case Constant:
    os << "." << amt;
    break;
  }
}

//===----------------------------------------------------------------------===//
// Methods on FormatSpecifier.
//===----------------------------------------------------------------------===//

ArgTypeResult FormatSpecifier::getArgType(ASTContext &Ctx) const {
  if (!CS.consumesDataArgument())
    return ArgTypeResult::Invalid();

  if (CS.isIntArg())
    switch (LM.getKind()) {
      case LengthModifier::AsLongDouble:
        return ArgTypeResult::Invalid();
      case LengthModifier::None: return Ctx.IntTy;
      case LengthModifier::AsChar: return Ctx.SignedCharTy;
      case LengthModifier::AsShort: return Ctx.ShortTy;
      case LengthModifier::AsLong: return Ctx.LongTy;
      case LengthModifier::AsLongLong: return Ctx.LongLongTy;
      case LengthModifier::AsIntMax:
        // FIXME: Return unknown for now.
        return ArgTypeResult();
      case LengthModifier::AsSizeT: return Ctx.getSizeType();
      case LengthModifier::AsPtrDiff: return Ctx.getPointerDiffType();
    }

  if (CS.isUIntArg())
    switch (LM.getKind()) {
      case LengthModifier::AsLongDouble:
        return ArgTypeResult::Invalid();
      case LengthModifier::None: return Ctx.UnsignedIntTy;
      case LengthModifier::AsChar: return Ctx.UnsignedCharTy;
      case LengthModifier::AsShort: return Ctx.UnsignedShortTy;
      case LengthModifier::AsLong: return Ctx.UnsignedLongTy;
      case LengthModifier::AsLongLong: return Ctx.UnsignedLongLongTy;
      case LengthModifier::AsIntMax:
        // FIXME: Return unknown for now.
        return ArgTypeResult();
      case LengthModifier::AsSizeT:
        // FIXME: How to get the corresponding unsigned
        // version of size_t?
        return ArgTypeResult();
      case LengthModifier::AsPtrDiff:
        // FIXME: How to get the corresponding unsigned
        // version of ptrdiff_t?
        return ArgTypeResult();
    }

  if (CS.isDoubleArg()) {
    if (LM.getKind() == LengthModifier::AsLongDouble)
      return Ctx.LongDoubleTy;
    return Ctx.DoubleTy;
  }

  switch (CS.getKind()) {
    case ConversionSpecifier::CStrArg:
      return ArgTypeResult(LM.getKind() == LengthModifier::AsWideChar ?
          ArgTypeResult::WCStrTy : ArgTypeResult::CStrTy);
    case ConversionSpecifier::UnicodeStrArg:
      // FIXME: This appears to be Mac OS X specific.
      return ArgTypeResult::WCStrTy;
    case ConversionSpecifier::CArg:
      return Ctx.WCharTy;
    default:
      break;
  }

  // FIXME: Handle other cases.
  return ArgTypeResult();
}

bool FormatSpecifier::fixType(QualType QT) {
  // Handle strings first (char *, wchar_t *)
  if (QT->isPointerType() && (QT->getPointeeType()->isAnyCharacterType())) {
    CS.setKind(ConversionSpecifier::CStrArg);

    // Set the long length modifier for wide characters
    if (QT->getPointeeType()->isWideCharType())
      LM.setKind(LengthModifier::AsWideChar);

    return true;
  }

  // We can only work with builtin types.
  if (!QT->isBuiltinType())
    return false;

  // Everything else should be a base type
  const BuiltinType *BT = QT->getAs<BuiltinType>();
  // Set length modifier
  switch (BT->getKind()) {
  default:
    break;
  case BuiltinType::WChar:
  case BuiltinType::Long:
  case BuiltinType::ULong:
    LM.setKind(LengthModifier::AsLong);
    break;

  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    LM.setKind(LengthModifier::AsLongLong);
    break;

  case BuiltinType::LongDouble:
    LM.setKind(LengthModifier::AsLongDouble);
    break;
  }

  // Set conversion specifier and disable any flags which do not apply to it.
  if (QT->isAnyCharacterType()) {
    CS.setKind(ConversionSpecifier::IntAsCharArg);
    Precision.setHowSpecified(OptionalAmount::NotSpecified);
    HasAlternativeForm = 0;
    HasLeadingZeroes = 0;
  }
  // Test for Floating type first as LongDouble can pass isUnsignedIntegerType
  else if (QT->isFloatingType()) {
    CS.setKind(ConversionSpecifier::fArg);
  }
  else if (QT->isPointerType()) {
    CS.setKind(ConversionSpecifier::VoidPtrArg);
    Precision.setHowSpecified(OptionalAmount::NotSpecified);
    HasAlternativeForm = 0;
    HasLeadingZeroes = 0;
  }
  else if (QT->isSignedIntegerType()) {
    CS.setKind(ConversionSpecifier::dArg);
    HasAlternativeForm = 0;
  }
  else if (QT->isUnsignedIntegerType()) {
    CS.setKind(ConversionSpecifier::uArg);
    HasAlternativeForm = 0;
  }
  else {
    return false;
  }

  return true;
}

void FormatSpecifier::toString(llvm::raw_ostream &os) const {
  // Whilst some features have no defined order, we are using the order
  // appearing in the C99 standard (ISO/IEC 9899:1999 (E) ¤7.19.6.1)
  os << "%";

  // Positional args
  if (usesPositionalArg()) {
    os << getPositionalArgIndex() << "$";
  }

  // Conversion flags
  if (IsLeftJustified)    os << "-";
  if (HasPlusPrefix)      os << "+";
  if (HasSpacePrefix)     os << " ";
  if (HasAlternativeForm) os << "#";
  if (HasLeadingZeroes)   os << "0";

  // Minimum field width
  FieldWidth.toString(os);
  // Precision
  Precision.toString(os);
  // Length modifier
  os << LM.toString();
  // Conversion specifier
  os << CS.toString();
}
