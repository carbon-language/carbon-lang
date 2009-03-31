//===--- TargetInfo.h - Expose information about the target -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TargetInfo interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETINFO_H
#define LLVM_CLANG_BASIC_TARGETINFO_H

#include "llvm/Support/DataTypes.h"
#include <vector>
#include <string>

namespace llvm { struct fltSemantics; }

namespace clang {

class Diagnostic;
class SourceManager;
class LangOptions;
  
namespace Builtin { struct Info; }
  
/// TargetInfo - This class exposes information about the current target.
///
class TargetInfo {
  std::string Triple;
protected:
  // Target values set by the ctor of the actual target implementation.  Default
  // values are specified by the TargetInfo constructor.
  bool CharIsSigned;
  unsigned char PointerWidth, PointerAlign;
  unsigned char WCharWidth, WCharAlign;
  unsigned char IntWidth, IntAlign;
  unsigned char FloatWidth, FloatAlign;
  unsigned char DoubleWidth, DoubleAlign;
  unsigned char LongDoubleWidth, LongDoubleAlign;
  unsigned char LongWidth, LongAlign;
  unsigned char LongLongWidth, LongLongAlign;
  unsigned char IntMaxTWidth;
  const char *DescriptionString;
  const char *UserLabelPrefix;
  const llvm::fltSemantics *FloatFormat, *DoubleFormat, *LongDoubleFormat;

  // TargetInfo Constructor.  Default initializes all fields.
  TargetInfo(const std::string &T);
  
public:  
  /// CreateTargetInfo - Return the target info object for the specified target
  /// triple.
  static TargetInfo* CreateTargetInfo(const std::string &Triple);

  virtual ~TargetInfo();

  ///===---- Target Data Type Query Methods -------------------------------===//
  enum IntType {
    NoInt = 0,
    SignedShort,
    UnsignedShort,
    SignedInt,
    UnsignedInt,
    SignedLong,
    UnsignedLong,
    SignedLongLong,
    UnsignedLongLong
  };
protected:
  IntType SizeType, IntMaxType, UIntMaxType, PtrDiffType, IntPtrType, WCharType;
public:
  IntType getSizeType() const { return SizeType; }
  IntType getIntMaxType() const { return IntMaxType; }
  IntType getUIntMaxType() const { return UIntMaxType; }
  IntType getPtrDiffType(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PtrDiffType : getPtrDiffTypeV(AddrSpace);
  }
  IntType getIntPtrType() const { return IntPtrType; }
  IntType getWCharType() const { return WCharType; }

  /// isCharSigned - Return true if 'char' is 'signed char' or false if it is
  /// treated as 'unsigned char'.  This is implementation defined according to
  /// C99 6.2.5p15.  In our implementation, this is target-specific.
  bool isCharSigned() const { return CharIsSigned; }
  
  /// getPointerWidth - Return the width of pointers on this target, for the
  /// specified address space.
  uint64_t getPointerWidth(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PointerWidth : getPointerWidthV(AddrSpace);
  }
  uint64_t getPointerAlign(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PointerAlign : getPointerAlignV(AddrSpace);
  }
  
  /// getBoolWidth/Align - Return the size of '_Bool' and C++ 'bool' for this
  /// target, in bits.
  unsigned getBoolWidth(bool isWide = false) const { return 8; }  // FIXME
  unsigned getBoolAlign(bool isWide = false) const { return 8; }  // FIXME
  
  unsigned getCharWidth(bool isWide = false) const {
    return isWide ? getWCharWidth() : 8; // FIXME
  }
  unsigned getCharAlign(bool isWide = false) const {
    return isWide ? getWCharAlign() : 8; // FIXME
  }
  
  /// getShortWidth/Align - Return the size of 'signed short' and
  /// 'unsigned short' for this target, in bits.  
  unsigned getShortWidth() const { return 16; } // FIXME
  unsigned getShortAlign() const { return 16; } // FIXME
  
  /// getIntWidth/Align - Return the size of 'signed int' and 'unsigned int' for
  /// this target, in bits.
  unsigned getIntWidth() const { return IntWidth; }
  unsigned getIntAlign() const { return IntAlign; }
  
  /// getLongWidth/Align - Return the size of 'signed long' and 'unsigned long'
  /// for this target, in bits.
  unsigned getLongWidth() const { return LongWidth; }
  unsigned getLongAlign() const { return LongAlign; }
  
  /// getLongLongWidth/Align - Return the size of 'signed long long' and
  /// 'unsigned long long' for this target, in bits.
  unsigned getLongLongWidth() const { return LongLongWidth; }
  unsigned getLongLongAlign() const { return LongLongAlign; }
  
  /// getWcharWidth/Align - Return the size of 'wchar_t' for this target, in
  /// bits.
  unsigned getWCharWidth() const { return WCharWidth; }
  unsigned getWCharAlign() const { return WCharAlign; }

  /// getFloatWidth/Align/Format - Return the size/align/format of 'float'.
  unsigned getFloatWidth() const { return FloatWidth; }
  unsigned getFloatAlign() const { return FloatAlign; }
  const llvm::fltSemantics &getFloatFormat() const { return *FloatFormat; }

  /// getDoubleWidth/Align/Format - Return the size/align/format of 'double'.
  unsigned getDoubleWidth() const { return DoubleWidth; }
  unsigned getDoubleAlign() const { return DoubleAlign; }
  const llvm::fltSemantics &getDoubleFormat() const { return *DoubleFormat; }

  /// getLongDoubleWidth/Align/Format - Return the size/align/format of 'long
  /// double'.
  unsigned getLongDoubleWidth() const { return LongDoubleWidth; }
  unsigned getLongDoubleAlign() const { return LongDoubleAlign; }
  const llvm::fltSemantics &getLongDoubleFormat() const {
    return *LongDoubleFormat;
  }
  
  /// getIntMaxTWidth - Return the size of intmax_t and uintmax_t for this
  /// target, in bits.  
  unsigned getIntMaxTWidth() const {
    return IntMaxTWidth;
  }
  
  /// getUserLabelPrefix - This returns the default value of the
  /// __USER_LABEL_PREFIX__ macro, which is the prefix given to user symbols by
  /// default.  On most platforms this is "_", but it is "" on some, and "." on
  /// others.
  const char *getUserLabelPrefix() const {
    return UserLabelPrefix;
  }
  
  /// getTypeName - Return the user string for the specified integer type enum.
  /// For example, SignedShort -> "short".
  static const char *getTypeName(IntType T);
  
  ///===---- Other target property query methods --------------------------===//
  
  /// getTargetDefines - Appends the target-specific #define values for this
  /// target set to the specified buffer.
  virtual void getTargetDefines(const LangOptions &Opts,
                                std::vector<char> &DefineBuffer) const = 0;
  
  /// getTargetBuiltins - Return information about target-specific builtins for
  /// the current primary target, and info about which builtins are non-portable
  /// across the current set of primary and secondary targets.
  virtual void getTargetBuiltins(const Builtin::Info *&Records, 
                                 unsigned &NumRecords) const = 0;

  /// getVAListDeclaration - Return the declaration to use for
  /// __builtin_va_list, which is target-specific.
  virtual const char *getVAListDeclaration() const = 0;

  /// isValidGCCRegisterName - Returns whether the passed in string
  /// is a valid register name according to GCC. This is used by Sema for
  /// inline asm statements.
  bool isValidGCCRegisterName(const char *Name) const;

  // getNormalizedGCCRegisterName - Returns the "normalized" GCC register name.
  // For example, on x86 it will return "ax" when "eax" is passed in.
  const char *getNormalizedGCCRegisterName(const char *Name) const;
  
  enum ConstraintInfo {
    CI_None = 0x00,
    CI_AllowsMemory = 0x01,
    CI_AllowsRegister = 0x02,
    CI_ReadWrite = 0x04
  };

  // validateOutputConstraint, validateInputConstraint - Checks that
  // a constraint is valid and provides information about it.
  // FIXME: These should return a real error instead of just true/false.
  bool validateOutputConstraint(const char *Name, ConstraintInfo &Info) const;
  bool validateInputConstraint(const char *Name, 
                               const std::string *OutputNamesBegin,
                               const std::string *OutputNamesEnd,
                               ConstraintInfo* OutputConstraints,
                               ConstraintInfo &info) const;
  bool resolveSymbolicName(const char *&Name,
                           const std::string *OutputNamesBegin,
                           const std::string *OutputNamesEnd,
                           unsigned &Index) const;
  
  virtual std::string convertConstraint(const char Constraint) const {
    return std::string(1, Constraint);
  }
  
  // Returns a string of target-specific clobbers, in LLVM format.
  virtual const char *getClobbers() const = 0;
  

  /// getTargetPrefix - Return the target prefix used for identifying
  /// llvm intrinsics.
  virtual const char *getTargetPrefix() const = 0;
    
  /// getTargetTriple - Return the target triple of the primary target.
  const char *getTargetTriple() const {
    return Triple.c_str();
  }
  
  const char *getTargetDescription() const {
    return DescriptionString;
  }

  struct GCCRegAlias {
    const char * const Aliases[5];
    const char * const Register;
  };

  virtual bool useGlobalsForAutomaticVariables() const { return false; }

  /// getStringSymbolPrefix - Get the default symbol prefix to
  /// use for string literals.
  virtual const char *getStringSymbolPrefix(bool IsConstant) const { 
    return ".str";
  }

  /// getCFStringSymbolPrefix - Get the default symbol prefix
  /// to use for CFString literals.
  virtual const char *getCFStringSymbolPrefix() const { 
    return "";
  }

  /// getCFStringSection - Return the section to use for the CFString
  /// literals, or 0 if no special section is used.
  virtual const char *getCFStringSection() const { 
    return "__DATA,__cfstring";
  }

  /// getCFStringDataSection - Return the section to use for the
  /// constant string data associated with a CFString literal, or 0 if
  /// no special section is used.
  virtual const char *getCFStringDataSection() const { 
    return "__TEXT,__cstring,cstring_literals";
  }

  /// getDefaultLangOptions - Allow the target to specify default settings for
  /// various language options.  These may be overridden by command line
  /// options. 
  virtual void getDefaultLangOptions(LangOptions &Opts) {}

  /// HandleTargetFeatures - Handle target-specific options like -mattr=+sse2
  /// and friends.  An array of arguments is passed in: if they are all valid,
  /// this should handle them and return -1.  If there is an error, the index of
  /// the invalid argument should be returned along with an optional error
  /// string.
  ///
  /// Note that the driver should have already consolidated all the
  /// target-feature settings and passed them to us in the -mattr list.  The
  /// -mattr list is treated by the code generator as a diff against the -mcpu
  /// setting, but the driver should pass all enabled options as "+" settings.
  /// This means that the target should only look at + settings.
  virtual int HandleTargetFeatures(std::string *StrArray, unsigned NumStrs,
                                   std::string &ErrorReason) {
    if (NumStrs == 0)
      return -1;
    return 0;
  }
protected:
  virtual uint64_t getPointerWidthV(unsigned AddrSpace) const {
    return PointerWidth;
  }
  virtual uint64_t getPointerAlignV(unsigned AddrSpace) const {
    return PointerAlign;
  }
  virtual enum IntType getPtrDiffTypeV(unsigned AddrSpace) const {
    return PtrDiffType;
  }
  virtual void getGCCRegNames(const char * const *&Names, 
                              unsigned &NumNames) const = 0;
  virtual void getGCCRegAliases(const GCCRegAlias *&Aliases, 
                                unsigned &NumAliases) const = 0;
  virtual bool validateAsmConstraint(const char *&Name, 
                                     TargetInfo::ConstraintInfo &info) const= 0;
};

}  // end namespace clang

#endif
