//== PrintfFormatString.cpp - Analysis of printf format strings --*- C++ -*-==//
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

#include "clang/Analysis/Analyses/FormatString.h"
#include "FormatStringParsing.h"

using clang::analyze_format_string::ArgTypeResult;
using clang::analyze_format_string::FormatStringHandler;
using clang::analyze_format_string::LengthModifier;
using clang::analyze_format_string::OptionalAmount;
using clang::analyze_format_string::ConversionSpecifier;
using clang::analyze_printf::PrintfSpecifier;

using namespace clang;

typedef clang::analyze_format_string::SpecifierResult<PrintfSpecifier>
        PrintfSpecifierResult;

//===----------------------------------------------------------------------===//
// Methods for parsing format strings.
//===----------------------------------------------------------------------===//

using analyze_format_string::ParseNonPositionAmount;

static bool ParsePrecision(FormatStringHandler &H, PrintfSpecifier &FS,
                           const char *Start, const char *&Beg, const char *E,
                           unsigned *argIndex) {
  if (argIndex) {
    FS.setPrecision(ParseNonPositionAmount(Beg, E, *argIndex));
  }
  else {
    const OptionalAmount Amt = ParsePositionAmount(H, Start, Beg, E,
                                           analyze_format_string::PrecisionPos);
    if (Amt.isInvalid())
      return true;
    FS.setPrecision(Amt);
  }
  return false;
}

static PrintfSpecifierResult ParsePrintfSpecifier(FormatStringHandler &H,
                                                  const char *&Beg,
                                                  const char *E,
                                                  unsigned &argIndex) {

  using namespace clang::analyze_format_string;
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
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }

  PrintfSpecifier FS;
  if (ParseArgPosition(H, FS, Start, I, E))
    return true;

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }

  // Look for flags (if any).
  bool hasMore = true;
  for ( ; I != E; ++I) {
    switch (*I) {
      default: hasMore = false; break;
      case '-': FS.setIsLeftJustified(I); break;
      case '+': FS.setHasPlusPrefix(I); break;
      case ' ': FS.setHasSpacePrefix(I); break;
      case '#': FS.setHasAlternativeForm(I); break;
      case '0': FS.setHasLeadingZeros(I); break;
    }
    if (!hasMore)
      break;
  }

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }

  // Look for the field width (if any).
  if (ParseFieldWidth(H, FS, Start, I, E,
                      FS.usesPositionalArg() ? 0 : &argIndex))
    return true;

  if (I == E) {
    // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
    return true;
  }

  // Look for the precision (if any).
  if (*I == '.') {
    ++I;
    if (I == E) {
      H.HandleIncompleteSpecifier(Start, E - Start);
      return true;
    }

    if (ParsePrecision(H, FS, Start, I, E,
                       FS.usesPositionalArg() ? 0 : &argIndex))
      return true;

    if (I == E) {
      // No more characters left?
      H.HandleIncompleteSpecifier(Start, E - Start);
      return true;
    }
  }

  // Look for the length modifier.
  if (ParseLengthModifier(FS, I, E) && I == E) {
    // No more characters left?
    H.HandleIncompleteSpecifier(Start, E - Start);
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
    case 'c': k = ConversionSpecifier::cArg; break;
    case 'd': k = ConversionSpecifier::dArg; break;
    case 'e': k = ConversionSpecifier::eArg; break;
    case 'f': k = ConversionSpecifier::fArg; break;
    case 'g': k = ConversionSpecifier::gArg; break;
    case 'i': k = ConversionSpecifier::iArg; break;
    case 'n': k = ConversionSpecifier::nArg; break;
    case 'o': k = ConversionSpecifier::oArg; break;
    case 'p': k = ConversionSpecifier::pArg;   break;
    case 's': k = ConversionSpecifier::sArg;      break;
    case 'u': k = ConversionSpecifier::uArg; break;
    case 'x': k = ConversionSpecifier::xArg; break;
    // Mac OS X (unicode) specific
    case 'C': k = ConversionSpecifier::CArg; break;
    case 'S': k = ConversionSpecifier::SArg; break;
    // Objective-C.
    case '@': k = ConversionSpecifier::ObjCObjArg; break;
    // Glibc specific.
    case 'm': k = ConversionSpecifier::PrintErrno; break;
  }
  PrintfConversionSpecifier CS(conversionPosition, k);
  FS.setConversionSpecifier(CS);
  if (CS.consumesDataArgument() && !FS.usesPositionalArg())
    FS.setArgIndex(argIndex++);

  if (k == ConversionSpecifier::InvalidSpecifier) {
    // Assume the conversion takes one argument.
    return !H.HandleInvalidPrintfConversionSpecifier(FS, Beg, I - Beg);
  }
  return PrintfSpecifierResult(Start, FS);
}

bool clang::analyze_format_string::ParsePrintfString(FormatStringHandler &H,
                                                     const char *I,
                                                     const char *E) {

  unsigned argIndex = 0;

  // Keep looking for a format specifier until we have exhausted the string.
  while (I != E) {
    const PrintfSpecifierResult &FSR = ParsePrintfSpecifier(H, I, E, argIndex);
    // Did a fail-stop error of any kind occur when parsing the specifier?
    // If so, don't do any more processing.
    if (FSR.shouldStop())
      return true;;
    // Did we exhaust the string or encounter an error that
    // we can recover from?
    if (!FSR.hasValue())
      continue;
    // We have a format specifier.  Pass it to the callback.
    if (!H.HandlePrintfSpecifier(FSR.getValue(), FSR.getStart(),
                                 I - FSR.getStart()))
      return true;
  }
  assert(I == E && "Format string not exhausted");
  return false;
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
  case cArg: return "c";
  case sArg: return "s";
  case pArg: return "p";
  case nArg: return "n";
  case PercentArg:  return "%";
  case ScanListArg: return "[";
  case InvalidSpecifier: return NULL;

  // MacOS X unicode extensions.
  case CArg: return "C";
  case SArg: return "S";

  // Objective-C specific specifiers.
  case ObjCObjArg: return "@";

  // GlibC specific specifiers.
  case PrintErrno: return "m";
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Methods on PrintfSpecifier.
//===----------------------------------------------------------------------===//

ArgTypeResult PrintfSpecifier::getArgType(ASTContext &Ctx) const {
  const PrintfConversionSpecifier &CS = getConversionSpecifier();
  
  if (!CS.consumesDataArgument())
    return ArgTypeResult::Invalid();

  if (CS.getKind() == ConversionSpecifier::cArg)
    switch (LM.getKind()) {
      case LengthModifier::None: return Ctx.IntTy;
      case LengthModifier::AsLong: return ArgTypeResult::WIntTy;
      default:
        return ArgTypeResult::Invalid();
    }
  
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
    case ConversionSpecifier::sArg:
      return ArgTypeResult(LM.getKind() == LengthModifier::AsWideChar ?
          ArgTypeResult::WCStrTy : ArgTypeResult::CStrTy);
    case ConversionSpecifier::SArg:
      // FIXME: This appears to be Mac OS X specific.
      return ArgTypeResult::WCStrTy;
    case ConversionSpecifier::CArg:
      return Ctx.WCharTy;
    case ConversionSpecifier::pArg:
      return ArgTypeResult::CPointerTy;
    default:
      break;
  }

  // FIXME: Handle other cases.
  return ArgTypeResult();
}

bool PrintfSpecifier::fixType(QualType QT) {
  // Handle strings first (char *, wchar_t *)
  if (QT->isPointerType() && (QT->getPointeeType()->isAnyCharacterType())) {
    CS.setKind(ConversionSpecifier::sArg);

    // Disable irrelevant flags
    HasAlternativeForm = 0;
    HasLeadingZeroes = 0;

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
    // The rest of the conversions are either optional or for non-builtin types
    LM.setKind(LengthModifier::None);
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
    CS.setKind(ConversionSpecifier::cArg);
    Precision.setHowSpecified(OptionalAmount::NotSpecified);
    HasAlternativeForm = 0;
    HasLeadingZeroes = 0;
    HasPlusPrefix = 0;
  }
  // Test for Floating type first as LongDouble can pass isUnsignedIntegerType
  else if (QT->isRealFloatingType()) {
    CS.setKind(ConversionSpecifier::fArg);
  }
  else if (QT->isPointerType()) {
    CS.setKind(ConversionSpecifier::pArg);
    Precision.setHowSpecified(OptionalAmount::NotSpecified);
    HasAlternativeForm = 0;
    HasLeadingZeroes = 0;
    HasPlusPrefix = 0;
  }
  else if (QT->isSignedIntegerType()) {
    CS.setKind(ConversionSpecifier::dArg);
    HasAlternativeForm = 0;
  }
  else if (QT->isUnsignedIntegerType()) {
    CS.setKind(ConversionSpecifier::uArg);
    HasAlternativeForm = 0;
    HasPlusPrefix = 0;
  }
  else {
    return false;
  }

  return true;
}

void PrintfSpecifier::toString(llvm::raw_ostream &os) const {
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

bool PrintfSpecifier::hasValidPlusPrefix() const {
  if (!HasPlusPrefix)
    return true;

  // The plus prefix only makes sense for signed conversions
  switch (CS.getKind()) {
  case ConversionSpecifier::dArg:
  case ConversionSpecifier::iArg:
  case ConversionSpecifier::fArg:
  case ConversionSpecifier::FArg:
  case ConversionSpecifier::eArg:
  case ConversionSpecifier::EArg:
  case ConversionSpecifier::gArg:
  case ConversionSpecifier::GArg:
  case ConversionSpecifier::aArg:
  case ConversionSpecifier::AArg:
    return true;

  default:
    return false;
  }
}

bool PrintfSpecifier::hasValidAlternativeForm() const {
  if (!HasAlternativeForm)
    return true;

  // Alternate form flag only valid with the oxaAeEfFgG conversions
  switch (CS.getKind()) {
  case ConversionSpecifier::oArg:
  case ConversionSpecifier::xArg:
  case ConversionSpecifier::aArg:
  case ConversionSpecifier::AArg:
  case ConversionSpecifier::eArg:
  case ConversionSpecifier::EArg:
  case ConversionSpecifier::fArg:
  case ConversionSpecifier::FArg:
  case ConversionSpecifier::gArg:
  case ConversionSpecifier::GArg:
    return true;

  default:
    return false;
  }
}

bool PrintfSpecifier::hasValidLeadingZeros() const {
  if (!HasLeadingZeroes)
    return true;

  // Leading zeroes flag only valid with the diouxXaAeEfFgG conversions
  switch (CS.getKind()) {
  case ConversionSpecifier::dArg:
  case ConversionSpecifier::iArg:
  case ConversionSpecifier::oArg:
  case ConversionSpecifier::uArg:
  case ConversionSpecifier::xArg:
  case ConversionSpecifier::XArg:
  case ConversionSpecifier::aArg:
  case ConversionSpecifier::AArg:
  case ConversionSpecifier::eArg:
  case ConversionSpecifier::EArg:
  case ConversionSpecifier::fArg:
  case ConversionSpecifier::FArg:
  case ConversionSpecifier::gArg:
  case ConversionSpecifier::GArg:
    return true;

  default:
    return false;
  }
}

bool PrintfSpecifier::hasValidSpacePrefix() const {
  if (!HasSpacePrefix)
    return true;

  // The space prefix only makes sense for signed conversions
  switch (CS.getKind()) {
  case ConversionSpecifier::dArg:
  case ConversionSpecifier::iArg:
  case ConversionSpecifier::fArg:
  case ConversionSpecifier::FArg:
  case ConversionSpecifier::eArg:
  case ConversionSpecifier::EArg:
  case ConversionSpecifier::gArg:
  case ConversionSpecifier::GArg:
  case ConversionSpecifier::aArg:
  case ConversionSpecifier::AArg:
    return true;

  default:
    return false;
  }
}

bool PrintfSpecifier::hasValidLeftJustified() const {
  if (!IsLeftJustified)
    return true;

  // The left justified flag is valid for all conversions except n
  switch (CS.getKind()) {
  case ConversionSpecifier::nArg:
    return false;

  default:
    return true;
  }
}

bool PrintfSpecifier::hasValidPrecision() const {
  if (Precision.getHowSpecified() == OptionalAmount::NotSpecified)
    return true;

  // Precision is only valid with the diouxXaAeEfFgGs conversions
  switch (CS.getKind()) {
  case ConversionSpecifier::dArg:
  case ConversionSpecifier::iArg:
  case ConversionSpecifier::oArg:
  case ConversionSpecifier::uArg:
  case ConversionSpecifier::xArg:
  case ConversionSpecifier::XArg:
  case ConversionSpecifier::aArg:
  case ConversionSpecifier::AArg:
  case ConversionSpecifier::eArg:
  case ConversionSpecifier::EArg:
  case ConversionSpecifier::fArg:
  case ConversionSpecifier::FArg:
  case ConversionSpecifier::gArg:
  case ConversionSpecifier::GArg:
  case ConversionSpecifier::sArg:
    return true;

  default:
    return false;
  }
}
bool PrintfSpecifier::hasValidFieldWidth() const {
  if (FieldWidth.getHowSpecified() == OptionalAmount::NotSpecified)
      return true;

  // The field width is valid for all conversions except n
  switch (CS.getKind()) {
  case ConversionSpecifier::nArg:
    return false;

  default:
    return true;
  }
}
