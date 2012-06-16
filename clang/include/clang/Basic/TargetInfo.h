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

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/DataTypes.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/VersionTuple.h"
#include <cassert>
#include <vector>
#include <string>

namespace llvm {
struct fltSemantics;
}

namespace clang {
class DiagnosticsEngine;
class LangOptions;
class MacroBuilder;
class SourceLocation;
class SourceManager;
class TargetOptions;

namespace Builtin { struct Info; }

/// TargetCXXABI - The types of C++ ABIs for which we can generate code.
enum TargetCXXABI {
  /// The generic ("Itanium") C++ ABI, documented at:
  ///   http://www.codesourcery.com/public/cxx-abi/
  CXXABI_Itanium,

  /// The ARM C++ ABI, based largely on the Itanium ABI but with
  /// significant differences.
  ///    http://infocenter.arm.com
  ///                    /help/topic/com.arm.doc.ihi0041c/IHI0041C_cppabi.pdf
  CXXABI_ARM,

  /// The Visual Studio ABI.  Only scattered official documentation exists.
  CXXABI_Microsoft
};

/// TargetInfo - This class exposes information about the current target.
///
class TargetInfo : public RefCountedBase<TargetInfo> {
  llvm::Triple Triple;
protected:
  // Target values set by the ctor of the actual target implementation.  Default
  // values are specified by the TargetInfo constructor.
  bool BigEndian;
  bool TLSSupported;
  bool NoAsmVariants;  // True if {|} are normal characters.
  unsigned char PointerWidth, PointerAlign;
  unsigned char BoolWidth, BoolAlign;
  unsigned char IntWidth, IntAlign;
  unsigned char HalfWidth, HalfAlign;
  unsigned char FloatWidth, FloatAlign;
  unsigned char DoubleWidth, DoubleAlign;
  unsigned char LongDoubleWidth, LongDoubleAlign;
  unsigned char LargeArrayMinWidth, LargeArrayAlign;
  unsigned char LongWidth, LongAlign;
  unsigned char LongLongWidth, LongLongAlign;
  unsigned char SuitableAlign;
  unsigned char MaxAtomicPromoteWidth, MaxAtomicInlineWidth;
  const char *DescriptionString;
  const char *UserLabelPrefix;
  const char *MCountName;
  const llvm::fltSemantics *HalfFormat, *FloatFormat, *DoubleFormat,
    *LongDoubleFormat;
  unsigned char RegParmMax, SSERegParmMax;
  TargetCXXABI CXXABI;
  const LangAS::Map *AddrSpaceMap;

  mutable StringRef PlatformName;
  mutable VersionTuple PlatformMinVersion;

  unsigned HasAlignMac68kSupport : 1;
  unsigned RealTypeUsesObjCFPRet : 3;
  unsigned ComplexLongDoubleUsesFP2Ret : 1;

  // TargetInfo Constructor.  Default initializes all fields.
  TargetInfo(const std::string &T);

public:
  /// CreateTargetInfo - Construct a target for the given options.
  ///
  /// \param Opts - The options to use to initialize the target. The target may
  /// modify the options to canonicalize the target feature information to match
  /// what the backend expects.
  static TargetInfo* CreateTargetInfo(DiagnosticsEngine &Diags,
                                      TargetOptions &Opts);

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

  enum RealType {
    Float = 0,
    Double,
    LongDouble
  };

  /// BuiltinVaListKind - The different kinds of __builtin_va_list types
  /// defined by the target implementation.
  enum BuiltinVaListKind {
    /// typedef char* __builtin_va_list;
    CharPtrBuiltinVaList = 0,

    /// typedef void* __builtin_va_list;
    VoidPtrBuiltinVaList,

    /// __builtin_va_list as defined by the PNaCl ABI:
    /// http://www.chromium.org/nativeclient/pnacl/bitcode-abi#TOC-Machine-Types
    PNaClABIBuiltinVaList,

    /// __builtin_va_list as defined by the Power ABI:
    /// https://www.power.org
    ///        /resources/downloads/Power-Arch-32-bit-ABI-supp-1.0-Embedded.pdf
    PowerABIBuiltinVaList,

    /// __builtin_va_list as defined by the x86-64 ABI:
    /// http://www.x86-64.org/documentation/abi.pdf
    X86_64ABIBuiltinVaList
  };

protected:
  IntType SizeType, IntMaxType, UIntMaxType, PtrDiffType, IntPtrType, WCharType,
          WIntType, Char16Type, Char32Type, Int64Type, SigAtomicType;

  /// Flag whether the Objective-C built-in boolean type should be signed char.
  /// Otherwise, when this flag is not set, the normal built-in boolean type is
  /// used.
  unsigned UseSignedCharForObjCBool : 1;

  /// Control whether the alignment of bit-field types is respected when laying
  /// out structures. If true, then the alignment of the bit-field type will be
  /// used to (a) impact the alignment of the containing structure, and (b)
  /// ensure that the individual bit-field will not straddle an alignment
  /// boundary.
  unsigned UseBitFieldTypeAlignment : 1;

  /// Control whether zero length bitfields (e.g., int : 0;) force alignment of
  /// the next bitfield.  If the alignment of the zero length bitfield is 
  /// greater than the member that follows it, `bar', `bar' will be aligned as
  /// the type of the zero-length bitfield.
  unsigned UseZeroLengthBitfieldAlignment : 1;

  /// If non-zero, specifies a fixed alignment value for bitfields that follow
  /// zero length bitfield, regardless of the zero length bitfield type.
  unsigned ZeroLengthBitfieldBoundary;

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
  static bool isTypeSigned(IntType T);

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
  unsigned getBoolWidth() const { return BoolWidth; }
  unsigned getBoolAlign() const { return BoolAlign; }

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

  /// getSuitableAlign - Return the alignment that is suitable for storing any
  /// object with a fundamental alignment requirement.
  unsigned getSuitableAlign() const { return SuitableAlign; }

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

  /// getHalfWidth/Align/Format - Return the size/align/format of 'half'.
  unsigned getHalfWidth() const { return HalfWidth; }
  unsigned getHalfAlign() const { return HalfAlign; }
  const llvm::fltSemantics &getHalfFormat() const { return *HalfFormat; }

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

  /// getFloatEvalMethod - Return the value for the C99 FLT_EVAL_METHOD macro.
  virtual unsigned getFloatEvalMethod() const { return 0; }

  // getLargeArrayMinWidth/Align - Return the minimum array size that is
  // 'large' and its alignment.
  unsigned getLargeArrayMinWidth() const { return LargeArrayMinWidth; }
  unsigned getLargeArrayAlign() const { return LargeArrayAlign; }

  /// getMaxAtomicPromoteWidth - Return the maximum width lock-free atomic
  /// operation which will ever be supported for the given target
  unsigned getMaxAtomicPromoteWidth() const { return MaxAtomicPromoteWidth; }
  /// getMaxAtomicInlineWidth - Return the maximum width lock-free atomic
  /// operation which can be inlined given the supported features of the
  /// given target.
  unsigned getMaxAtomicInlineWidth() const { return MaxAtomicInlineWidth; }

  /// getIntMaxTWidth - Return the size of intmax_t and uintmax_t for this
  /// target, in bits.
  unsigned getIntMaxTWidth() const {
    return getTypeWidth(IntMaxType);
  }

  /// getRegisterWidth - Return the "preferred" register width on this target.
  uint64_t getRegisterWidth() const {
    // Currently we assume the register width on the target matches the pointer
    // width, we can introduce a new variable for this if/when some target wants
    // it.
    return LongWidth; 
  }

  /// getUserLabelPrefix - This returns the default value of the
  /// __USER_LABEL_PREFIX__ macro, which is the prefix given to user symbols by
  /// default.  On most platforms this is "_", but it is "" on some, and "." on
  /// others.
  const char *getUserLabelPrefix() const {
    return UserLabelPrefix;
  }

  /// MCountName - This returns name of the mcount instrumentation function.
  const char *getMCountName() const {
    return MCountName;
  }

  /// useSignedCharForObjCBool - Check if the Objective-C built-in boolean
  /// type should be signed char.  Otherwise, if this returns false, the
  /// normal built-in boolean type should also be used for Objective-C.
  bool useSignedCharForObjCBool() const {
    return UseSignedCharForObjCBool;
  }
  void noSignedCharForObjCBool() {
    UseSignedCharForObjCBool = false;
  }

  /// useBitFieldTypeAlignment() - Check whether the alignment of bit-field 
  /// types is respected when laying out structures.
  bool useBitFieldTypeAlignment() const {
    return UseBitFieldTypeAlignment;
  }

  /// useZeroLengthBitfieldAlignment() - Check whether zero length bitfields 
  /// should force alignment of the next member.
  bool useZeroLengthBitfieldAlignment() const {
    return UseZeroLengthBitfieldAlignment;
  }

  /// getZeroLengthBitfieldBoundary() - Get the fixed alignment value in bits
  /// for a member that follows a zero length bitfield.
  unsigned getZeroLengthBitfieldBoundary() const {
    return ZeroLengthBitfieldBoundary;
  }

  /// hasAlignMac68kSupport - Check whether this target support '#pragma options
  /// align=mac68k'.
  bool hasAlignMac68kSupport() const {
    return HasAlignMac68kSupport;
  }

  /// getTypeName - Return the user string for the specified integer type enum.
  /// For example, SignedShort -> "short".
  static const char *getTypeName(IntType T);

  /// getTypeConstantSuffix - Return the constant suffix for the specified
  /// integer type enum. For example, SignedLong -> "L".
  static const char *getTypeConstantSuffix(IntType T);

  /// \brief Check whether the given real type should use the "fpret" flavor of
  /// Obj-C message passing on this target.
  bool useObjCFPRetForRealType(RealType T) const {
    return RealTypeUsesObjCFPRet & (1 << T);
  }

  /// \brief Check whether _Complex long double should use the "fp2ret" flavor
  /// of Obj-C message passing on this target.
  bool useObjCFP2RetForComplexLongDouble() const {
    return ComplexLongDoubleUsesFP2Ret;
  }

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

  /// isCLZForZeroUndef - The __builtin_clz* and __builtin_ctz* built-in
  /// functions are specified to have undefined results for zero inputs, but
  /// on targets that support these operations in a way that provides
  /// well-defined results for zero without loss of performance, it is a good
  /// idea to avoid optimizing based on that undef behavior.
  virtual bool isCLZForZeroUndef() const { return true; }

  /// getBuiltinVaListKind - Returns the kind of __builtin_va_list
  /// type that should be used with this target.
  virtual BuiltinVaListKind getBuiltinVaListKind() const = 0;

  /// isValidClobber - Returns whether the passed in string is
  /// a valid clobber in an inline asm statement. This is used by
  /// Sema.
  bool isValidClobber(StringRef Name) const;

  /// isValidGCCRegisterName - Returns whether the passed in string
  /// is a valid register name according to GCC. This is used by Sema for
  /// inline asm statements.
  bool isValidGCCRegisterName(StringRef Name) const;

  // getNormalizedGCCRegisterName - Returns the "normalized" GCC register name.
  // For example, on x86 it will return "ax" when "eax" is passed in.
  StringRef getNormalizedGCCRegisterName(StringRef Name) const;

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
    ConstraintInfo(StringRef ConstraintStr, StringRef Name)
      : Flags(0), TiedOperand(-1), ConstraintStr(ConstraintStr.str()),
      Name(Name.str()) {}

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

  // Constraint parm will be left pointing at the last character of
  // the constraint.  In practice, it won't be changed unless the
  // constraint is longer than one character.
  virtual std::string convertConstraint(const char *&Constraint) const {
    // 'p' defaults to 'r', but can be overridden by targets.
    if (*Constraint == 'p')
      return std::string("r");
    return std::string(1, *Constraint);
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

  struct AddlRegName {
    const char * const Names[5];
    const unsigned RegNum;
  };

  /// hasProtectedVisibility - Does this target support "protected"
  /// visibility?
  ///
  /// Any target which dynamic libraries will naturally support
  /// something like "default" (meaning that the symbol is visible
  /// outside this shared object) and "hidden" (meaning that it isn't)
  /// visibilities, but "protected" is really an ELF-specific concept
  /// with weird semantics designed around the convenience of dynamic
  /// linker implementations.  Which is not to suggest that there's
  /// consistent target-independent semantics for "default" visibility
  /// either; the entire thing is pretty badly mangled.
  virtual bool hasProtectedVisibility() const { return true; }

  virtual bool useGlobalsForAutomaticVariables() const { return false; }

  /// getCFStringSection - Return the section to use for CFString
  /// literals, or 0 if no special section is used.
  virtual const char *getCFStringSection() const {
    return "__DATA,__cfstring";
  }

  /// getNSStringSection - Return the section to use for NSString
  /// literals, or 0 if no special section is used.
  virtual const char *getNSStringSection() const {
    return "__OBJC,__cstring_object,regular,no_dead_strip";
  }

  /// getNSStringNonFragileABISection - Return the section to use for
  /// NSString literals, or 0 if no special section is used (NonFragile ABI).
  virtual const char *getNSStringNonFragileABISection() const {
    return "__DATA, __objc_stringobj, regular, no_dead_strip";
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
  virtual std::string isValidSectionSpecifier(StringRef SR) const {
    return "";
  }

  /// setForcedLangOptions - Set forced language options.
  /// Apply changes to the target information with respect to certain
  /// language options which change the target configuration.
  virtual void setForcedLangOptions(LangOptions &Opts);

  /// getDefaultFeatures - Get the default set of target features for the CPU;
  /// this should include all legal feature strings on the target.
  virtual void getDefaultFeatures(llvm::StringMap<bool> &Features) const {
  }

  /// getABI - Get the ABI in use.
  virtual const char *getABI() const {
    return "";
  }

  /// getCXXABI - Get the C++ ABI in use.
  virtual TargetCXXABI getCXXABI() const {
    return CXXABI;
  }

  /// setCPU - Target the specific CPU.
  ///
  /// \return - False on error (invalid CPU name).
  virtual bool setCPU(const std::string &Name) {
    return false;
  }

  /// setABI - Use the specific ABI.
  ///
  /// \return - False on error (invalid ABI name).
  virtual bool setABI(const std::string &Name) {
    return false;
  }

  /// setCXXABI - Use this specific C++ ABI.
  ///
  /// \return - False on error (invalid C++ ABI name).
  bool setCXXABI(const std::string &Name) {
    static const TargetCXXABI Unknown = static_cast<TargetCXXABI>(-1);
    TargetCXXABI ABI = llvm::StringSwitch<TargetCXXABI>(Name)
      .Case("arm", CXXABI_ARM)
      .Case("itanium", CXXABI_Itanium)
      .Case("microsoft", CXXABI_Microsoft)
      .Default(Unknown);
    if (ABI == Unknown) return false;
    return setCXXABI(ABI);
  }

  /// setCXXABI - Set the C++ ABI to be used by this implementation.
  ///
  /// \return - False on error (ABI not valid on this target)
  virtual bool setCXXABI(TargetCXXABI ABI) {
    CXXABI = ABI;
    return true;
  }

  /// setFeatureEnabled - Enable or disable a specific target feature,
  /// the feature name must be valid.
  ///
  /// \return - False on error (invalid feature name).
  virtual bool setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
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

  /// \brief Determine whether the given target has the given feature.
  virtual bool hasFeature(StringRef Feature) const {
    return false;
  }
  
  // getRegParmMax - Returns maximal number of args passed in registers.
  unsigned getRegParmMax() const {
    assert(RegParmMax < 7 && "RegParmMax value is larger than AST can handle");
    return RegParmMax;
  }

  /// isTLSSupported - Whether the target supports thread-local storage.
  bool isTLSSupported() const {
    return TLSSupported;
  }

  /// hasNoAsmVariants - Return true if {|} are normal characters in the
  /// asm string.  If this returns false (the default), then {abc|xyz} is syntax
  /// that says that when compiling for asm variant #0, "abc" should be
  /// generated, but when compiling for asm variant #1, "xyz" should be
  /// generated.
  bool hasNoAsmVariants() const {
    return NoAsmVariants;
  }

  /// getEHDataRegisterNumber - Return the register number that
  /// __builtin_eh_return_regno would return with the specified argument.
  virtual int getEHDataRegisterNumber(unsigned RegNo) const {
    return -1;
  }

  /// getStaticInitSectionSpecifier - Return the section to use for C++ static
  /// initialization functions.
  virtual const char *getStaticInitSectionSpecifier() const {
    return 0;
  }

  const LangAS::Map &getAddressSpaceMap() const {
    return *AddrSpaceMap;
  }

  /// \brief Retrieve the name of the platform as it is used in the
  /// availability attribute.
  StringRef getPlatformName() const { return PlatformName; }

  /// \brief Retrieve the minimum desired version of the platform, to
  /// which the program should be compiled.
  VersionTuple getPlatformMinVersion() const { return PlatformMinVersion; }

  bool isBigEndian() const { return BigEndian; }

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
  virtual void getGCCAddlRegNames(const AddlRegName *&Addl,
				  unsigned &NumAddl) const {
    Addl = 0;
    NumAddl = 0;
  }
  virtual bool validateAsmConstraint(const char *&Name,
                                     TargetInfo::ConstraintInfo &info) const= 0;
};

}  // end namespace clang

#endif
