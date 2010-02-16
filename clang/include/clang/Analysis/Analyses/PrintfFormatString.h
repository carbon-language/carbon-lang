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
  unsigned getLength() const {
    // Conversion specifiers currently only are represented by
    // single characters, but we be flexible.
    return 1;
  }
  
private:
  const char *Position;
  Kind kind;
};

enum LengthModifier {
 None,
 AsChar,      // 'hh'
 AsShort,     // 'h'
 AsLong,      // 'l'
 AsLongLong,  // 'll', 'q' (BSD, deprecated)
 AsIntMax,    // 'j'
 AsSizeT,     // 'z'
 AsPtrDiff,   // 't'
 AsLongDouble, // 'L'
 AsWideChar = AsLong // for '%ls'
};

class OptionalAmount {
public:
  enum HowSpecified { NotSpecified, Constant, Arg };

  OptionalAmount(HowSpecified h, const char *st) 
    : start(st), hs(h), amt(0) {}

  OptionalAmount()
    : start(0), hs(NotSpecified), amt(0) {}
  
  OptionalAmount(unsigned i, const char *st) 
    : start(st), hs(Constant), amt(i) {}

  HowSpecified getHowSpecified() const { return hs; }
  bool hasDataArgument() const { return hs == Arg; }

  unsigned getConstantAmount() const { 
    assert(hs == Constant);
    return amt;
  }

  const char *getStart() const {
    return start;
  }
  
private:
  const char *start;
  HowSpecified hs;
  unsigned amt;
};
  
class ArgTypeResult {
  enum Kind { UnknownTy, InvalidTy, SpecificTy, ObjCPointerTy };
  const Kind K;
  QualType T;
  ArgTypeResult(bool) : K(InvalidTy) {}  
public:
  ArgTypeResult() : K(UnknownTy) {}
  ArgTypeResult(QualType t) : K(SpecificTy), T(t) {}
  ArgTypeResult(CanQualType t) : K(SpecificTy), T(t) {}
  
  static ArgTypeResult Invalid() { return ArgTypeResult(true); }
  
  bool isValid() const { return K != InvalidTy; }
  
  const QualType *getSpecificType() const { 
    return K == SpecificTy ? &T : 0;
  }
    
  bool matchesAnyObjCObjectRef() const { return K == ObjCPointerTy; }
};    

class FormatSpecifier {
  LengthModifier LM;
  unsigned IsLeftJustified : 1;
  unsigned HasPlusPrefix : 1;
  unsigned HasSpacePrefix : 1;
  unsigned HasAlternativeForm : 1;
  unsigned HasLeadingZeroes : 1;
  unsigned flags : 5;
  ConversionSpecifier CS;
  OptionalAmount FieldWidth;
  OptionalAmount Precision;
public:
  FormatSpecifier() : LM(None),
    IsLeftJustified(0), HasPlusPrefix(0), HasSpacePrefix(0),
    HasAlternativeForm(0), HasLeadingZeroes(0) {}
  
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
};

class FormatStringHandler {
public:
  FormatStringHandler() {}
  virtual ~FormatStringHandler();
  
  virtual void HandleIncompleteFormatSpecifier(const char *startSpecifier,
                                               unsigned specifierLen) {}

  virtual void HandleNullChar(const char *nullCharacter) {}
    
  virtual void 
    HandleInvalidConversionSpecifier(const analyze_printf::FormatSpecifier &FS,
                                     const char *startSpecifier,
                                     unsigned specifierLen) {}
  
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
