//==- PrintfFormatStrings.h - Analysis of printf format strings --*- C++ -*-==//
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

#ifndef LLVM_CLANG_FPRINTF_FORMAT_H
#define LLVM_CLANG_FPRINTF_FORMAT_H

#include "clang/AST/CanonicalType.h"

namespace clang {

class ASTContext;

namespace analyze_printf {

class ArgTypeResult {
public:
  enum Kind { UnknownTy, InvalidTy, SpecificTy, ObjCPointerTy, CPointerTy,
              CStrTy, WCStrTy };
private:
  const Kind K;
  QualType T;
  ArgTypeResult(bool) : K(InvalidTy) {}
public:
  ArgTypeResult(Kind k = UnknownTy) : K(k) {}
  ArgTypeResult(QualType t) : K(SpecificTy), T(t) {}
  ArgTypeResult(CanQualType t) : K(SpecificTy), T(t) {}

  static ArgTypeResult Invalid() { return ArgTypeResult(true); }

  bool isValid() const { return K != InvalidTy; }

  const QualType *getSpecificType() const {
    return K == SpecificTy ? &T : 0;
  }

  bool matchesType(ASTContext &C, QualType argTy) const;

  bool matchesAnyObjCObjectRef() const { return K == ObjCPointerTy; }

  QualType getRepresentativeType(ASTContext &C) const;
};

class ConversionSpecifier {
public:
  enum Kind {
   InvalidSpecifier = 0,
    // C99 conversion specifiers.
   dArg, // 'd'
   iArg, // 'i',
   oArg, // 'o',
   uArg, // 'u',
   xArg, // 'x',
   XArg, // 'X',
   fArg, // 'f',
   FArg, // 'F',
   eArg, // 'e',
   EArg, // 'E',
   gArg, // 'g',
   GArg, // 'G',
   aArg, // 'a',
   AArg, // 'A',
   IntAsCharArg,  // 'c'
   CStrArg,       // 's'
   VoidPtrArg,    // 'p'
   OutIntPtrArg,  // 'n'
   PercentArg,    // '%'
   // MacOS X unicode extensions.
   CArg, // 'C'
   UnicodeStrArg, // 'S'
   // Objective-C specific specifiers.
   ObjCObjArg,    // '@'
   // GlibC specific specifiers.
   PrintErrno,    // 'm'
   // Specifier ranges.
   IntArgBeg = dArg,
   IntArgEnd = iArg,
   UIntArgBeg = oArg,
   UIntArgEnd = XArg,
   DoubleArgBeg = fArg,
   DoubleArgEnd = AArg,
   C99Beg = IntArgBeg,
   C99End = DoubleArgEnd,
   ObjCBeg = ObjCObjArg,
   ObjCEnd = ObjCObjArg
  };

  ConversionSpecifier()
    : Position(0), kind(InvalidSpecifier) {}

  ConversionSpecifier(const char *pos, Kind k)
    : Position(pos), kind(k) {}

  const char *getStart() const {
    return Position;
  }

  llvm::StringRef getCharacters() const {
    return llvm::StringRef(getStart(), getLength());
  }

  bool consumesDataArgument() const {
    switch (kind) {
  	  case PercentArg:
	  case PrintErrno:
		return false;
	  default:
		return true;
	}
  }

  bool isObjCArg() const { return kind >= ObjCBeg && kind <= ObjCEnd; }
  bool isIntArg() const { return kind >= dArg && kind <= iArg; }
  bool isUIntArg() const { return kind >= oArg && kind <= XArg; }
  bool isDoubleArg() const { return kind >= fArg && kind <= AArg; }
  Kind getKind() const { return kind; }
  void setKind(Kind k) { kind = k; }
  unsigned getLength() const {
    // Conversion specifiers currently only are represented by
    // single characters, but we be flexible.
    return 1;
  }
  const char *toString() const;

private:
  const char *Position;
  Kind kind;
};

class LengthModifier {
public:
  enum Kind {
   None,
   AsChar,       // 'hh'
   AsShort,      // 'h'
   AsLong,       // 'l'
   AsLongLong,   // 'll', 'q' (BSD, deprecated)
   AsIntMax,     // 'j'
   AsSizeT,      // 'z'
   AsPtrDiff,    // 't'
   AsLongDouble, // 'L'
   AsWideChar = AsLong // for '%ls'
  };

  LengthModifier()
    : Position(0), kind(None) {}
  LengthModifier(const char *pos, Kind k)
    : Position(pos), kind(k) {}

  const char *getStart() const {
    return Position;
  }

  unsigned getLength() const {
    switch (kind) {
    default:
      return 1;
    case AsLongLong:
      return 2;
    case None:
      return 0;
    }
  }

  Kind getKind() const { return kind; }
  void setKind(Kind k) { kind = k; }

  const char *toString() const;

private:
  const char *Position;
  Kind kind;
};

class OptionalAmount {
public:
  enum HowSpecified { NotSpecified, Constant, Arg, Invalid };

  OptionalAmount(HowSpecified howSpecified,
                 unsigned amount,
                 const char *amountStart,
                 bool usesPositionalArg)
    : start(amountStart), hs(howSpecified), amt(amount),
      UsesPositionalArg(usesPositionalArg) {}

  OptionalAmount(bool valid = true)
    : start(0), hs(valid ? NotSpecified : Invalid), amt(0),
      UsesPositionalArg(0) {}

  bool isInvalid() const {
    return hs == Invalid;
  }

  HowSpecified getHowSpecified() const { return hs; }
  void setHowSpecified(HowSpecified h) { hs = h; }

  bool hasDataArgument() const { return hs == Arg; }

  unsigned getArgIndex() const {
    assert(hasDataArgument());
    return amt;
  }

  unsigned getConstantAmount() const {
    assert(hs == Constant);
    return amt;
  }

  const char *getStart() const {
    return start;
  }

  ArgTypeResult getArgType(ASTContext &Ctx) const;

  void toString(llvm::raw_ostream &os) const;

  bool usesPositionalArg() const { return (bool) UsesPositionalArg; }
  unsigned getPositionalArgIndex() const {
    assert(hasDataArgument());
    return amt + 1;
  }

private:
  const char *start;
  HowSpecified hs;
  unsigned amt;
  bool UsesPositionalArg : 1;
};

class FormatSpecifier {
  LengthModifier LM;
  unsigned IsLeftJustified : 1;
  unsigned HasPlusPrefix : 1;
  unsigned HasSpacePrefix : 1;
  unsigned HasAlternativeForm : 1;
  unsigned HasLeadingZeroes : 1;
  /// Positional arguments, an IEEE extension:
  ///  IEEE Std 1003.1, 2004 Edition
  ///  http://www.opengroup.org/onlinepubs/009695399/functions/printf.html
  unsigned UsesPositionalArg : 1;
  unsigned argIndex;
  ConversionSpecifier CS;
  OptionalAmount FieldWidth;
  OptionalAmount Precision;
public:
  FormatSpecifier() :
    IsLeftJustified(0), HasPlusPrefix(0), HasSpacePrefix(0),
    HasAlternativeForm(0), HasLeadingZeroes(0), UsesPositionalArg(0),
    argIndex(0) {}

  static FormatSpecifier Parse(const char *beg, const char *end);

  // Methods for incrementally constructing the FormatSpecifier.
  void setConversionSpecifier(const ConversionSpecifier &cs) {
    CS = cs;
  }
  void setLengthModifier(LengthModifier lm) {
    LM = lm;
  }
  void setIsLeftJustified() { IsLeftJustified = 1; }
  void setHasPlusPrefix() { HasPlusPrefix = 1; }
  void setHasSpacePrefix() { HasSpacePrefix = 1; }
  void setHasAlternativeForm() { HasAlternativeForm = 1; }
  void setHasLeadingZeros() { HasLeadingZeroes = 1; }
  void setUsesPositionalArg() { UsesPositionalArg = 1; }

  void setArgIndex(unsigned i) {
    assert(CS.consumesDataArgument());
    argIndex = i;
  }

  unsigned getArgIndex() const {
    assert(CS.consumesDataArgument());
    return argIndex;
  }

  unsigned getPositionalArgIndex() const {
    assert(CS.consumesDataArgument());
    return argIndex + 1;
  }

  // Methods for querying the format specifier.

  const ConversionSpecifier &getConversionSpecifier() const {
    return CS;
  }

  LengthModifier getLengthModifier() const {
    return LM;
  }

  const OptionalAmount &getFieldWidth() const {
    return FieldWidth;
  }

  void setFieldWidth(const OptionalAmount &Amt) {
    FieldWidth = Amt;
  }

  void setPrecision(const OptionalAmount &Amt) {
    Precision = Amt;
  }

  const OptionalAmount &getPrecision() const {
    return Precision;
  }

  /// \brief Returns the builtin type that a data argument
  /// paired with this format specifier should have.  This method
  /// will return null if the format specifier does not have
  /// a matching data argument or the matching argument matches
  /// more than one type.
  ArgTypeResult getArgType(ASTContext &Ctx) const;

  bool isLeftJustified() const { return (bool) IsLeftJustified; }
  bool hasPlusPrefix() const { return (bool) HasPlusPrefix; }
  bool hasAlternativeForm() const { return (bool) HasAlternativeForm; }
  bool hasLeadingZeros() const { return (bool) HasLeadingZeroes; }
  bool hasSpacePrefix() const { return (bool) HasSpacePrefix; }
  bool usesPositionalArg() const { return (bool) UsesPositionalArg; }

  /// Changes the specifier and length according to a QualType, retaining any
  /// flags or options. Returns true on success, or false when a conversion
  /// was not successful.
  bool fixType(QualType QT);

  void toString(llvm::raw_ostream &os) const;
};

enum PositionContext { FieldWidthPos = 0, PrecisionPos = 1 };

class FormatStringHandler {
public:
  FormatStringHandler() {}
  virtual ~FormatStringHandler();

  virtual void HandleIncompleteFormatSpecifier(const char *startSpecifier,
                                               unsigned specifierLen) {}

  virtual void HandleNullChar(const char *nullCharacter) {}

  virtual void HandleInvalidPosition(const char *startPos, unsigned posLen,
                                     PositionContext p) {}

  virtual void HandleZeroPosition(const char *startPos, unsigned posLen) {}

  virtual bool
    HandleInvalidConversionSpecifier(const analyze_printf::FormatSpecifier &FS,
                                     const char *startSpecifier,
                                     unsigned specifierLen) { return true; }

  virtual bool HandleFormatSpecifier(const analyze_printf::FormatSpecifier &FS,
                                     const char *startSpecifier,
                                     unsigned specifierLen) {
    return true;
  }
};

bool ParseFormatString(FormatStringHandler &H,
                       const char *beg, const char *end);

} // end printf namespace
} // end clang namespace
#endif
