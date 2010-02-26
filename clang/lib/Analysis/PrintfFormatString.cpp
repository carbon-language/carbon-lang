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

using clang::analyze_printf::FormatSpecifier;
using clang::analyze_printf::OptionalAmount;
using clang::analyze_printf::ArgTypeResult;
using clang::analyze_printf::FormatStringHandler;
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

static OptionalAmount ParseAmount(const char *&Beg, const char *E,
                                  unsigned &argIndex) {
  const char *I = Beg;
  UpdateOnReturn <const char*> UpdateBeg(Beg, I);

  bool foundDigits = false;
  unsigned accumulator = 0;

  for ( ; I != E; ++I) {
    char c = *I;
    if (c >= '0' && c <= '9') {
      foundDigits = true;
      accumulator += (accumulator * 10) + (c - '0');
      continue;
    }

    if (foundDigits)
      return OptionalAmount(OptionalAmount::Constant, accumulator, Beg);

    if (c == '*') {
      ++I;
      return OptionalAmount(OptionalAmount::Arg, argIndex++, Beg);
    }

    break;
  }

  return OptionalAmount();
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
  FS.setFieldWidth(ParseAmount(I, E, argIndex));

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

    FS.setPrecision(ParseAmount(I, E, argIndex));

    if (I == E) {
      // No more characters left?
      H.HandleIncompleteFormatSpecifier(Start, E - Start);
      return true;
    }
  }

  // Look for the length modifier.
  LengthModifier lm = None;
  switch (*I) {
    default:
      break;
    case 'h':
      ++I;
      lm = (I != E && *I == 'h') ? ++I, AsChar : AsShort;
      break;
    case 'l':
      ++I;
      lm = (I != E && *I == 'l') ? ++I, AsLongLong : AsLong;
      break;
    case 'j': lm = AsIntMax;     ++I; break;
    case 'z': lm = AsSizeT;      ++I; break;
    case 't': lm = AsPtrDiff;    ++I; break;
    case 'L': lm = AsLongDouble; ++I; break;
    case 'q': lm = AsLongLong;   ++I; break;
  }
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
  if (CS.consumesDataArgument())
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
// Methods on FormatSpecifier.
//===----------------------------------------------------------------------===//

ArgTypeResult FormatSpecifier::getArgType(ASTContext &Ctx) const {
  if (!CS.consumesDataArgument())
    return ArgTypeResult::Invalid();

  if (CS.isIntArg())
    switch (LM) {
      case AsLongDouble:
        return ArgTypeResult::Invalid();
      case None: return Ctx.IntTy;
      case AsChar: return Ctx.SignedCharTy;
      case AsShort: return Ctx.ShortTy;
      case AsLong: return Ctx.LongTy;
      case AsLongLong: return Ctx.LongLongTy;
      case AsIntMax:
        // FIXME: Return unknown for now.
        return ArgTypeResult();
      case AsSizeT: return Ctx.getSizeType();
      case AsPtrDiff: return Ctx.getPointerDiffType();
    }

  if (CS.isUIntArg())
    switch (LM) {
      case AsLongDouble:
        return ArgTypeResult::Invalid();
      case None: return Ctx.UnsignedIntTy;
      case AsChar: return Ctx.UnsignedCharTy;
      case AsShort: return Ctx.UnsignedShortTy;
      case AsLong: return Ctx.UnsignedLongTy;
      case AsLongLong: return Ctx.UnsignedLongLongTy;
      case AsIntMax:
        // FIXME: Return unknown for now.
        return ArgTypeResult();
      case AsSizeT:
        // FIXME: How to get the corresponding unsigned
        // version of size_t?
        return ArgTypeResult();
      case AsPtrDiff:
        // FIXME: How to get the corresponding unsigned
        // version of ptrdiff_t?
        return ArgTypeResult();
    }

  if (CS.isDoubleArg()) {
    if (LM == AsLongDouble)
      return Ctx.LongDoubleTy;
    return Ctx.DoubleTy;
  }

  switch (CS.getKind()) {
    case ConversionSpecifier::CStrArg:
      return ArgTypeResult(LM == AsWideChar ? ArgTypeResult::WCStrTy                                            : ArgTypeResult::CStrTy);
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

