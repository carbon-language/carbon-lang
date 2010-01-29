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
 AsLongLong,  // 'll'
 AsIntMax,    // 'j'
 AsSizeT,     // 'z'
 AsPtrDiff,   // 't'
 AsLongDouble // 'L'
};

enum Flags {
 LeftJustified = 0x1,
 PlusPrefix = 0x2,
 SpacePrefix = 0x4,
 AlternativeForm = 0x8,
 LeadingZeroes = 0x16
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
    
  bool matchesAnyObjCObjectRef() const { return K == ObjCPointerTy; };
};    

class FormatSpecifier {
  LengthModifier LM;
  unsigned flags : 5;
  ConversionSpecifier CS;
  OptionalAmount FieldWidth;
  OptionalAmount Precision;
public:
  FormatSpecifier() : LM(None), flags(0) {}
  
  static FormatSpecifier Parse(const char *beg, const char *end);

  // Methods for incrementally constructing the FormatSpecifier.
  void setConversionSpecifier(const ConversionSpecifier &cs) {
    CS = cs;
  }
  void setLengthModifier(LengthModifier lm) {
    LM = lm;
  }
  void setIsLeftJustified() { flags |= LeftJustified; }
  void setHasPlusPrefix() { flags |= PlusPrefix; }
  void setHasSpacePrefix() { flags |= SpacePrefix; }
  void setHasAlternativeForm() { flags |= AlternativeForm; }
  void setHasLeadingZeros() { flags |= LeadingZeroes; }

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

  bool isLeftJustified() const { return flags & LeftJustified; }
  bool hasPlusPrefix() const { return flags & PlusPrefix; }
  bool hasAlternativeForm() const { return flags & AlternativeForm; }
  bool hasLeadingZeros() const { return flags & LeadingZeroes; }  
};

} // end printf namespace

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


} // end clang namespace
#endif
