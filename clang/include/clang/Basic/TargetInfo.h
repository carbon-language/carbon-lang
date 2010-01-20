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

// FIXME: Daniel isn't smart enough to use a prototype for this.
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/DataTypes.h"
#include <cassert>
#include <vector>
#include <string>

namespace llvm {
struct fltSemantics;
class StringRef;
}

namespace clang {
class Diagnostic;
class LangOptions;
class SourceLocation;
class SourceManager;
class TargetOptions;

namespace Builtin { struct Info; }

class MacroBuilder {
  llvm::raw_ostream &Out;
public:
  MacroBuilder(llvm::raw_ostream &Output) : Out(Output) {}

  /// Append a #define line for macro of the form "#define Name Value\n".
  void defineMacro(const llvm::Twine &Name, const llvm::Twine &Value = "1") {
    Out << "#define " << Name << ' ' << Value << '\n';
  }

  /// Append a #undef line for Name.  Name should be of the form XXX
  /// and we emit "#undef XXX".
  void undefineMacro(const llvm::Twine &Name) {
    Out << "#undef " << Name << '\n';
  }

  /// Directly append Str and a newline to the underlying buffer.
  void append(const llvm::Twine &Str) {
    Out << Str << '\n';
  }
};

/// TargetInfo - This class exposes information about the current target.
///
class TargetInfo {
  llvm::Triple Triple;
protected:
  // Target values set by the ctor of the actual target implementation.  Default
  // values are specified by the TargetInfo constructor.
  bool TLSSupported;
  unsigned char PointerWidth, PointerAlign;
  unsigned char IntWidth, IntAlign;
  unsigned char FloatWidth, FloatAlign;
  unsigned char DoubleWidth, DoubleAlign;
  unsigned char LongDoubleWidth, LongDoubleAlign;
  unsigned char LongWidth, LongAlign;
  unsigned char LongLongWidth, LongLongAlign;
  const char *DescriptionString;
  const char *UserLabelPrefix;
  const llvm::fltSemantics *FloatFormat, *DoubleFormat, *LongDoubleFormat;
  unsigned char RegParmMax, SSERegParmMax;

  // TargetInfo Constructor.  Default initializes all fields.
  TargetInfo(const std::string &T);

public:
  /// CreateTargetInfo - Construct a target for the given options.
  ///
  /// \param Opts - The options to use to initialize the target. The target may
  /// modify the options to canonicalize the target feature information to match
  /// what the backend expects.
  static TargetInfo* CreateTargetInfo(Diagnostic &Diags, TargetOptions &Opts);

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
  IntType SizeType, IntMaxType, UIntMaxType, PtrDiffType, IntPtrType, WCharType,
          WIntType, Char16Type, Char32Type, Int64Type, SigAtomicType;
public:
  IntType getSizeType() const { return SizeType; }
  IntType getIntMaxType() const { return IntMaxType; }
  IntType getUIntMaxType() const { return UIntMaxType; }
  IntType getPtrDiffType(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PtrDiffType : getPtrDiffTypeV(AddrSpace);
  }
  IntType getIntPtrType() const { return IntPtrType; }
  IntType getWCharType() const { return WCharType; }
  IntType getWIntType() const { return WIntType; }
  IntType getChar16Type() const { return Char16Type; }
  IntType getChar32Type() const { return Char32Type; }
  IntType getInt64Type() const { return Int64Type; }
  IntType getSigAtomicType() const { return SigAtomicType; }


  /// getTypeWidth - Return the width (in bits) of the specified integer type 
  /// enum. For example, SignedInt -> getIntWidth().
  unsigned getTypeWidth(IntType T) const;

  /// getTypeAlign - Return the alignment (in bits) of the specified integer
  /// type enum. For example, SignedInt -> getIntAlign().
  unsigned getTypeAlign(IntType T) const;

  /// isTypeSigned - Return whether an integer types is signed. Returns true if
  /// the type is signed; false otherwise.
  bool isTypeSigned(IntType T) const;

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

  unsigned getCharWidth() const { return 8; } // FIXME
  unsigned getCharAlign() const { return 8; } // FIXME

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

  /// getWCharWidth/Align - Return the size of 'wchar_t' for this target, in
  /// bits.
  unsigned getWCharWidth() const { return getTypeWidth(WCharType); }
  unsigned getWCharAlign() const { return getTypeAlign(WCharType); }

  /// getChar16Width/Align - Return the size of 'char16_t' for this target, in
  /// bits.
  unsigned getChar16Width() const { return getTypeWidth(Char16Type); }
  unsigned getChar16Align() const { return getTypeAlign(Char16Type); }

  /// getChar32Width/Align - Return the size of 'char32_t' for this target, in
  /// bits.
  unsigned getChar32Width() const { return getTypeWidth(Char32Type); }
  unsigned getChar32Align() const { return getTypeAlign(Char32Type); }

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
    return getTypeWidth(IntMaxType);
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

  /// getTypeConstantSuffix - Return the constant suffix for the specified
  /// integer type enum. For example, SignedLong -> "L".
  static const char *getTypeConstantSuffix(IntType T);

  ///===---- Other target property query methods --------------------------===//

  /// getTargetDefines - Appends the target-specific #define values for this
  /// target set to the specified buffer.
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const = 0;


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

  struct ConstraintInfo {
    enum {
      CI_None = 0x00,
      CI_AllowsMemory = 0x01,
      CI_AllowsRegister = 0x02,
      CI_ReadWrite = 0x04,       // "+r" output constraint (read and write).
      CI_HasMatchingInput = 0x08 // This output operand has a matching input.
    };
    unsigned Flags;
    int TiedOperand;

    std::string ConstraintStr;  // constraint: "=rm"
    std::string Name;           // Operand name: [foo] with no []'s.
  public:
    ConstraintInfo(const char *str, unsigned strlen, const std::string &name)
      : Flags(0), TiedOperand(-1), ConstraintStr(str, str+strlen), Name(name) {}
    explicit ConstraintInfo(const std::string &Str, const std::string &name)
      : Flags(0), TiedOperand(-1), ConstraintStr(Str), Name(name) {}

    const std::string &getConstraintStr() const { return ConstraintStr; }
    const std::string &getName() const { return Name; }
    bool isReadWrite() const { return (Flags & CI_ReadWrite) != 0; }
    bool allowsRegister() const { return (Flags & CI_AllowsRegister) != 0; }
    bool allowsMemory() const { return (Flags & CI_AllowsMemory) != 0; }

    /// hasMatchingInput - Return true if this output operand has a matching
    /// (tied) input operand.
    bool hasMatchingInput() const { return (Flags & CI_HasMatchingInput) != 0; }

    /// hasTiedOperand() - Return true if this input operand is a matching
    /// constraint that ties it to an output operand.  If this returns true,
    /// then getTiedOperand will indicate which output operand this is tied to.
    bool hasTiedOperand() const { return TiedOperand != -1; }
    unsigned getTiedOperand() const {
      assert(hasTiedOperand() && "Has no tied operand!");
      return (unsigned)TiedOperand;
    }

    void setIsReadWrite() { Flags |= CI_ReadWrite; }
    void setAllowsMemory() { Flags |= CI_AllowsMemory; }
    void setAllowsRegister() { Flags |= CI_AllowsRegister; }
    void setHasMatchingInput() { Flags |= CI_HasMatchingInput; }

    /// setTiedOperand - Indicate that this is an input operand that is tied to
    /// the specified output operand.  Copy over the various constraint
    /// information from the output.
    void setTiedOperand(unsigned N, ConstraintInfo &Output) {
      Output.setHasMatchingInput();
      Flags = Output.Flags;
      TiedOperand = N;
      // Don't copy Name or constraint string.
    }
  };

  // validateOutputConstraint, validateInputConstraint - Checks that
  // a constraint is valid and provides information about it.
  // FIXME: These should return a real error instead of just true/false.
  bool validateOutputConstraint(ConstraintInfo &Info) const;
  bool validateInputConstraint(ConstraintInfo *OutputConstraints,
                               unsigned NumOutputs,
                               ConstraintInfo &info) const;
  bool resolveSymbolicName(const char *&Name,
                           ConstraintInfo *OutputConstraints,
                           unsigned NumOutputs, unsigned &Index) const;

  virtual std::string convertConstraint(const char Constraint) const {
    return std::string(1, Constraint);
  }

  // Returns a string of target-specific clobbers, in LLVM format.
  virtual const char *getClobbers() const = 0;


  /// getTriple - Return the target triple of the primary target.
  const llvm::Triple &getTriple() const {
    return Triple;
  }

  const char *getTargetDescription() const {
    return DescriptionString;
  }

  struct GCCRegAlias {
    const char * const Aliases[5];
    const char * const Register;
  };

  virtual bool useGlobalsForAutomaticVariables() const { return false; }

  /// getUnicodeStringSection - Return the section to use for unicode
  /// string literals, or 0 if no special section is used.
  virtual const char *getUnicodeStringSection() const {
    return 0;
  }

  /// getCFStringSection - Return the section to use for CFString
  /// literals, or 0 if no special section is used.
  virtual const char *getCFStringSection() const {
    return "__DATA,__cfstring";
  }

  /// isValidSectionSpecifier - This is an optional hook that targets can
  /// implement to perform semantic checking on attribute((section("foo")))
  /// specifiers.  In this case, "foo" is passed in to be checked.  If the
  /// section specifier is invalid, the backend should return a non-empty string
  /// that indicates the problem.
  ///
  /// This hook is a simple quality of implementation feature to catch errors
  /// and give good diagnostics in cases when the assembler or code generator
  /// would otherwise reject the section specifier.
  ///
  virtual std::string isValidSectionSpecifier(const llvm::StringRef &SR) const {
    return "";
  }

  /// setForcedLangOptions - Set forced language options.
  /// Apply changes to the target information with respect to certain
  /// language options which change the target configuration.
  virtual void setForcedLangOptions(LangOptions &Opts);

  /// getDefaultFeatures - Get the default set of target features for
  /// the \args CPU; this should include all legal feature strings on
  /// the target.
  virtual void getDefaultFeatures(const std::string &CPU,
                                  llvm::StringMap<bool> &Features) const {
  }

  /// getABI - Get the ABI in use.
  virtual const char *getABI() const {
    return "";
  }

  /// setCPU - Target the specific CPU.
  ///
  /// \return - False on error (invalid CPU name).
  //
  // FIXME: Remove this.
  virtual bool setCPU(const std::string &Name) {
    return true;
  }

  /// setABI - Use the specific ABI.
  ///
  /// \return - False on error (invalid ABI name).
  virtual bool setABI(const std::string &Name) {
    return false;
  }

  /// setFeatureEnabled - Enable or disable a specific target feature,
  /// the feature name must be valid.
  ///
  /// \return - False on error (invalid feature name).
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 const std::string &Name,
                                 bool Enabled) const {
    return false;
  }

  /// HandleTargetOptions - Perform initialization based on the user configured
  /// set of features (e.g., +sse4). The list is guaranteed to have at most one
  /// entry per feature.
  ///
  /// The target may modify the features list, to change which options are
  /// passed onwards to the backend.
  virtual void HandleTargetFeatures(std::vector<std::string> &Features) {
  }

  // getRegParmMax - Returns maximal number of args passed in registers.
  unsigned getRegParmMax() const {
    return RegParmMax;
  }

  /// isTLSSupported - Whether the target supports thread-local storage.
  bool isTLSSupported() const {
    return TLSSupported;
  }
  
  /// getEHDataRegisterNumber - Return the register number that
  /// __builtin_eh_return_regno would return with the specified argument.
  virtual int getEHDataRegisterNumber(unsigned RegNo) const {
    return -1; 
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
