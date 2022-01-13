//===--- TargetInfo.h - Expose information about the target -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the clang::TargetInfo interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETINFO_H
#define LLVM_CLANG_BASIC_TARGETINFO_H

#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetCXXABI.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <string>
#include <vector>

namespace llvm {
struct fltSemantics;
}

namespace clang {
class DiagnosticsEngine;
class LangOptions;
class CodeGenOptions;
class MacroBuilder;

namespace Builtin { struct Info; }

enum class FloatModeKind {
  NoFloat = 255,
  Float = 0,
  Double,
  LongDouble,
  Float128,
  Ibm128
};

/// Fields controlling how types are laid out in memory; these may need to
/// be copied for targets like AMDGPU that base their ABIs on an auxiliary
/// CPU target.
struct TransferrableTargetInfo {
  unsigned char PointerWidth, PointerAlign;
  unsigned char BoolWidth, BoolAlign;
  unsigned char IntWidth, IntAlign;
  unsigned char HalfWidth, HalfAlign;
  unsigned char BFloat16Width, BFloat16Align;
  unsigned char FloatWidth, FloatAlign;
  unsigned char DoubleWidth, DoubleAlign;
  unsigned char LongDoubleWidth, LongDoubleAlign, Float128Align, Ibm128Align;
  unsigned char LargeArrayMinWidth, LargeArrayAlign;
  unsigned char LongWidth, LongAlign;
  unsigned char LongLongWidth, LongLongAlign;

  // Fixed point bit widths
  unsigned char ShortAccumWidth, ShortAccumAlign;
  unsigned char AccumWidth, AccumAlign;
  unsigned char LongAccumWidth, LongAccumAlign;
  unsigned char ShortFractWidth, ShortFractAlign;
  unsigned char FractWidth, FractAlign;
  unsigned char LongFractWidth, LongFractAlign;

  // If true, unsigned fixed point types have the same number of fractional bits
  // as their signed counterparts, forcing the unsigned types to have one extra
  // bit of padding. Otherwise, unsigned fixed point types have
  // one more fractional bit than its corresponding signed type. This is false
  // by default.
  bool PaddingOnUnsignedFixedPoint;

  // Fixed point integral and fractional bit sizes
  // Saturated types share the same integral/fractional bits as their
  // corresponding unsaturated types.
  // For simplicity, the fractional bits in a _Fract type will be one less the
  // width of that _Fract type. This leaves all signed _Fract types having no
  // padding and unsigned _Fract types will only have 1 bit of padding after the
  // sign if PaddingOnUnsignedFixedPoint is set.
  unsigned char ShortAccumScale;
  unsigned char AccumScale;
  unsigned char LongAccumScale;

  unsigned char SuitableAlign;
  unsigned char DefaultAlignForAttributeAligned;
  unsigned char MinGlobalAlign;

  unsigned short NewAlign;
  unsigned MaxVectorAlign;
  unsigned MaxTLSAlign;

  const llvm::fltSemantics *HalfFormat, *BFloat16Format, *FloatFormat,
      *DoubleFormat, *LongDoubleFormat, *Float128Format, *Ibm128Format;

  ///===---- Target Data Type Query Methods -------------------------------===//
  enum IntType {
    NoInt = 0,
    SignedChar,
    UnsignedChar,
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
  IntType SizeType, IntMaxType, PtrDiffType, IntPtrType, WCharType, WIntType,
      Char16Type, Char32Type, Int64Type, Int16Type, SigAtomicType,
      ProcessIDType;

  /// Whether Objective-C's built-in boolean type should be signed char.
  ///
  /// Otherwise, when this flag is not set, the normal built-in boolean type is
  /// used.
  unsigned UseSignedCharForObjCBool : 1;

  /// Control whether the alignment of bit-field types is respected when laying
  /// out structures. If true, then the alignment of the bit-field type will be
  /// used to (a) impact the alignment of the containing structure, and (b)
  /// ensure that the individual bit-field will not straddle an alignment
  /// boundary.
  unsigned UseBitFieldTypeAlignment : 1;

  /// Whether zero length bitfields (e.g., int : 0;) force alignment of
  /// the next bitfield.
  ///
  /// If the alignment of the zero length bitfield is greater than the member
  /// that follows it, `bar', `bar' will be aligned as the type of the
  /// zero-length bitfield.
  unsigned UseZeroLengthBitfieldAlignment : 1;

  /// Whether zero length bitfield alignment is respected if they are the
  /// leading members.
  unsigned UseLeadingZeroLengthBitfield : 1;

  ///  Whether explicit bit field alignment attributes are honored.
  unsigned UseExplicitBitFieldAlignment : 1;

  /// If non-zero, specifies a fixed alignment value for bitfields that follow
  /// zero length bitfield, regardless of the zero length bitfield type.
  unsigned ZeroLengthBitfieldBoundary;

  /// If non-zero, specifies a maximum alignment to truncate alignment
  /// specified in the aligned attribute of a static variable to this value.
  unsigned MaxAlignedAttribute;
};

/// OpenCL type kinds.
enum OpenCLTypeKind : uint8_t {
  OCLTK_Default,
  OCLTK_ClkEvent,
  OCLTK_Event,
  OCLTK_Image,
  OCLTK_Pipe,
  OCLTK_Queue,
  OCLTK_ReserveID,
  OCLTK_Sampler,
};

/// Exposes information about the current target.
///
class TargetInfo : public virtual TransferrableTargetInfo,
                   public RefCountedBase<TargetInfo> {
  std::shared_ptr<TargetOptions> TargetOpts;
  llvm::Triple Triple;
protected:
  // Target values set by the ctor of the actual target implementation.  Default
  // values are specified by the TargetInfo constructor.
  bool BigEndian;
  bool TLSSupported;
  bool VLASupported;
  bool NoAsmVariants;  // True if {|} are normal characters.
  bool HasLegalHalfType; // True if the backend supports operations on the half
                         // LLVM IR type.
  bool HasFloat128;
  bool HasFloat16;
  bool HasBFloat16;
  bool HasIbm128;
  bool HasLongDouble;
  bool HasFPReturn;
  bool HasStrictFP;

  unsigned char MaxAtomicPromoteWidth, MaxAtomicInlineWidth;
  unsigned short SimdDefaultAlign;
  std::string DataLayoutString;
  const char *UserLabelPrefix;
  const char *MCountName;
  unsigned char RegParmMax, SSERegParmMax;
  TargetCXXABI TheCXXABI;
  const LangASMap *AddrSpaceMap;

  mutable StringRef PlatformName;
  mutable VersionTuple PlatformMinVersion;

  unsigned HasAlignMac68kSupport : 1;
  unsigned RealTypeUsesObjCFPRet : 3;
  unsigned ComplexLongDoubleUsesFP2Ret : 1;

  unsigned HasBuiltinMSVaList : 1;

  unsigned IsRenderScriptTarget : 1;

  unsigned HasAArch64SVETypes : 1;

  unsigned HasRISCVVTypes : 1;

  unsigned AllowAMDGPUUnsafeFPAtomics : 1;

  unsigned ARMCDECoprocMask : 8;

  unsigned MaxOpenCLWorkGroupSize;

  // TargetInfo Constructor.  Default initializes all fields.
  TargetInfo(const llvm::Triple &T);

  // UserLabelPrefix must match DL's getGlobalPrefix() when interpreted
  // as a DataLayout object.
  void resetDataLayout(StringRef DL, const char *UserLabelPrefix = "");

public:
  /// Construct a target for the given options.
  ///
  /// \param Opts - The options to use to initialize the target. The target may
  /// modify the options to canonicalize the target feature information to match
  /// what the backend expects.
  static TargetInfo *
  CreateTargetInfo(DiagnosticsEngine &Diags,
                   const std::shared_ptr<TargetOptions> &Opts);

  virtual ~TargetInfo();

  /// Retrieve the target options.
  TargetOptions &getTargetOpts() const {
    assert(TargetOpts && "Missing target options");
    return *TargetOpts;
  }

  /// The different kinds of __builtin_va_list types defined by
  /// the target implementation.
  enum BuiltinVaListKind {
    /// typedef char* __builtin_va_list;
    CharPtrBuiltinVaList = 0,

    /// typedef void* __builtin_va_list;
    VoidPtrBuiltinVaList,

    /// __builtin_va_list as defined by the AArch64 ABI
    /// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055a/IHI0055A_aapcs64.pdf
    AArch64ABIBuiltinVaList,

    /// __builtin_va_list as defined by the PNaCl ABI:
    /// http://www.chromium.org/nativeclient/pnacl/bitcode-abi#TOC-Machine-Types
    PNaClABIBuiltinVaList,

    /// __builtin_va_list as defined by the Power ABI:
    /// https://www.power.org
    ///        /resources/downloads/Power-Arch-32-bit-ABI-supp-1.0-Embedded.pdf
    PowerABIBuiltinVaList,

    /// __builtin_va_list as defined by the x86-64 ABI:
    /// http://refspecs.linuxbase.org/elf/x86_64-abi-0.21.pdf
    X86_64ABIBuiltinVaList,

    /// __builtin_va_list as defined by ARM AAPCS ABI
    /// http://infocenter.arm.com
    //        /help/topic/com.arm.doc.ihi0042d/IHI0042D_aapcs.pdf
    AAPCSABIBuiltinVaList,

    // typedef struct __va_list_tag
    //   {
    //     long __gpr;
    //     long __fpr;
    //     void *__overflow_arg_area;
    //     void *__reg_save_area;
    //   } va_list[1];
    SystemZBuiltinVaList,

    // typedef struct __va_list_tag {
    //    void *__current_saved_reg_area_pointer;
    //    void *__saved_reg_area_end_pointer;
    //    void *__overflow_area_pointer;
    //} va_list;
    HexagonBuiltinVaList
  };

protected:
  /// Specify if mangling based on address space map should be used or
  /// not for language specific address spaces
  bool UseAddrSpaceMapMangling;

public:
  IntType getSizeType() const { return SizeType; }
  IntType getSignedSizeType() const {
    switch (SizeType) {
    case UnsignedShort:
      return SignedShort;
    case UnsignedInt:
      return SignedInt;
    case UnsignedLong:
      return SignedLong;
    case UnsignedLongLong:
      return SignedLongLong;
    default:
      llvm_unreachable("Invalid SizeType");
    }
  }
  IntType getIntMaxType() const { return IntMaxType; }
  IntType getUIntMaxType() const {
    return getCorrespondingUnsignedType(IntMaxType);
  }
  IntType getPtrDiffType(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PtrDiffType : getPtrDiffTypeV(AddrSpace);
  }
  IntType getUnsignedPtrDiffType(unsigned AddrSpace) const {
    return getCorrespondingUnsignedType(getPtrDiffType(AddrSpace));
  }
  IntType getIntPtrType() const { return IntPtrType; }
  IntType getUIntPtrType() const {
    return getCorrespondingUnsignedType(IntPtrType);
  }
  IntType getWCharType() const { return WCharType; }
  IntType getWIntType() const { return WIntType; }
  IntType getChar16Type() const { return Char16Type; }
  IntType getChar32Type() const { return Char32Type; }
  IntType getInt64Type() const { return Int64Type; }
  IntType getUInt64Type() const {
    return getCorrespondingUnsignedType(Int64Type);
  }
  IntType getInt16Type() const { return Int16Type; }
  IntType getUInt16Type() const {
    return getCorrespondingUnsignedType(Int16Type);
  }
  IntType getSigAtomicType() const { return SigAtomicType; }
  IntType getProcessIDType() const { return ProcessIDType; }

  static IntType getCorrespondingUnsignedType(IntType T) {
    switch (T) {
    case SignedChar:
      return UnsignedChar;
    case SignedShort:
      return UnsignedShort;
    case SignedInt:
      return UnsignedInt;
    case SignedLong:
      return UnsignedLong;
    case SignedLongLong:
      return UnsignedLongLong;
    default:
      llvm_unreachable("Unexpected signed integer type");
    }
  }

  /// In the event this target uses the same number of fractional bits for its
  /// unsigned types as it does with its signed counterparts, there will be
  /// exactly one bit of padding.
  /// Return true if unsigned fixed point types have padding for this target.
  bool doUnsignedFixedPointTypesHavePadding() const {
    return PaddingOnUnsignedFixedPoint;
  }

  /// Return the width (in bits) of the specified integer type enum.
  ///
  /// For example, SignedInt -> getIntWidth().
  unsigned getTypeWidth(IntType T) const;

  /// Return integer type with specified width.
  virtual IntType getIntTypeByWidth(unsigned BitWidth, bool IsSigned) const;

  /// Return the smallest integer type with at least the specified width.
  virtual IntType getLeastIntTypeByWidth(unsigned BitWidth,
                                         bool IsSigned) const;

  /// Return floating point type with specified width. On PPC, there are
  /// three possible types for 128-bit floating point: "PPC double-double",
  /// IEEE 754R quad precision, and "long double" (which under the covers
  /// is represented as one of those two). At this time, there is no support
  /// for an explicit "PPC double-double" type (i.e. __ibm128) so we only
  /// need to differentiate between "long double" and IEEE quad precision.
  FloatModeKind getRealTypeByWidth(unsigned BitWidth,
                                   FloatModeKind ExplicitType) const;

  /// Return the alignment (in bits) of the specified integer type enum.
  ///
  /// For example, SignedInt -> getIntAlign().
  unsigned getTypeAlign(IntType T) const;

  /// Returns true if the type is signed; false otherwise.
  static bool isTypeSigned(IntType T);

  /// Return the width of pointers on this target, for the
  /// specified address space.
  uint64_t getPointerWidth(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PointerWidth : getPointerWidthV(AddrSpace);
  }
  uint64_t getPointerAlign(unsigned AddrSpace) const {
    return AddrSpace == 0 ? PointerAlign : getPointerAlignV(AddrSpace);
  }

  /// Return the maximum width of pointers on this target.
  virtual uint64_t getMaxPointerWidth() const {
    return PointerWidth;
  }

  /// Get integer value for null pointer.
  /// \param AddrSpace address space of pointee in source language.
  virtual uint64_t getNullPointerValue(LangAS AddrSpace) const { return 0; }

  /// Return the size of '_Bool' and C++ 'bool' for this target, in bits.
  unsigned getBoolWidth() const { return BoolWidth; }

  /// Return the alignment of '_Bool' and C++ 'bool' for this target.
  unsigned getBoolAlign() const { return BoolAlign; }

  unsigned getCharWidth() const { return 8; } // FIXME
  unsigned getCharAlign() const { return 8; } // FIXME

  /// Return the size of 'signed short' and 'unsigned short' for this
  /// target, in bits.
  unsigned getShortWidth() const { return 16; } // FIXME

  /// Return the alignment of 'signed short' and 'unsigned short' for
  /// this target.
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

  /// getShortAccumWidth/Align - Return the size of 'signed short _Accum' and
  /// 'unsigned short _Accum' for this target, in bits.
  unsigned getShortAccumWidth() const { return ShortAccumWidth; }
  unsigned getShortAccumAlign() const { return ShortAccumAlign; }

  /// getAccumWidth/Align - Return the size of 'signed _Accum' and
  /// 'unsigned _Accum' for this target, in bits.
  unsigned getAccumWidth() const { return AccumWidth; }
  unsigned getAccumAlign() const { return AccumAlign; }

  /// getLongAccumWidth/Align - Return the size of 'signed long _Accum' and
  /// 'unsigned long _Accum' for this target, in bits.
  unsigned getLongAccumWidth() const { return LongAccumWidth; }
  unsigned getLongAccumAlign() const { return LongAccumAlign; }

  /// getShortFractWidth/Align - Return the size of 'signed short _Fract' and
  /// 'unsigned short _Fract' for this target, in bits.
  unsigned getShortFractWidth() const { return ShortFractWidth; }
  unsigned getShortFractAlign() const { return ShortFractAlign; }

  /// getFractWidth/Align - Return the size of 'signed _Fract' and
  /// 'unsigned _Fract' for this target, in bits.
  unsigned getFractWidth() const { return FractWidth; }
  unsigned getFractAlign() const { return FractAlign; }

  /// getLongFractWidth/Align - Return the size of 'signed long _Fract' and
  /// 'unsigned long _Fract' for this target, in bits.
  unsigned getLongFractWidth() const { return LongFractWidth; }
  unsigned getLongFractAlign() const { return LongFractAlign; }

  /// getShortAccumScale/IBits - Return the number of fractional/integral bits
  /// in a 'signed short _Accum' type.
  unsigned getShortAccumScale() const { return ShortAccumScale; }
  unsigned getShortAccumIBits() const {
    return ShortAccumWidth - ShortAccumScale - 1;
  }

  /// getAccumScale/IBits - Return the number of fractional/integral bits
  /// in a 'signed _Accum' type.
  unsigned getAccumScale() const { return AccumScale; }
  unsigned getAccumIBits() const { return AccumWidth - AccumScale - 1; }

  /// getLongAccumScale/IBits - Return the number of fractional/integral bits
  /// in a 'signed long _Accum' type.
  unsigned getLongAccumScale() const { return LongAccumScale; }
  unsigned getLongAccumIBits() const {
    return LongAccumWidth - LongAccumScale - 1;
  }

  /// getUnsignedShortAccumScale/IBits - Return the number of
  /// fractional/integral bits in a 'unsigned short _Accum' type.
  unsigned getUnsignedShortAccumScale() const {
    return PaddingOnUnsignedFixedPoint ? ShortAccumScale : ShortAccumScale + 1;
  }
  unsigned getUnsignedShortAccumIBits() const {
    return PaddingOnUnsignedFixedPoint
               ? getShortAccumIBits()
               : ShortAccumWidth - getUnsignedShortAccumScale();
  }

  /// getUnsignedAccumScale/IBits - Return the number of fractional/integral
  /// bits in a 'unsigned _Accum' type.
  unsigned getUnsignedAccumScale() const {
    return PaddingOnUnsignedFixedPoint ? AccumScale : AccumScale + 1;
  }
  unsigned getUnsignedAccumIBits() const {
    return PaddingOnUnsignedFixedPoint ? getAccumIBits()
                                       : AccumWidth - getUnsignedAccumScale();
  }

  /// getUnsignedLongAccumScale/IBits - Return the number of fractional/integral
  /// bits in a 'unsigned long _Accum' type.
  unsigned getUnsignedLongAccumScale() const {
    return PaddingOnUnsignedFixedPoint ? LongAccumScale : LongAccumScale + 1;
  }
  unsigned getUnsignedLongAccumIBits() const {
    return PaddingOnUnsignedFixedPoint
               ? getLongAccumIBits()
               : LongAccumWidth - getUnsignedLongAccumScale();
  }

  /// getShortFractScale - Return the number of fractional bits
  /// in a 'signed short _Fract' type.
  unsigned getShortFractScale() const { return ShortFractWidth - 1; }

  /// getFractScale - Return the number of fractional bits
  /// in a 'signed _Fract' type.
  unsigned getFractScale() const { return FractWidth - 1; }

  /// getLongFractScale - Return the number of fractional bits
  /// in a 'signed long _Fract' type.
  unsigned getLongFractScale() const { return LongFractWidth - 1; }

  /// getUnsignedShortFractScale - Return the number of fractional bits
  /// in a 'unsigned short _Fract' type.
  unsigned getUnsignedShortFractScale() const {
    return PaddingOnUnsignedFixedPoint ? getShortFractScale()
                                       : getShortFractScale() + 1;
  }

  /// getUnsignedFractScale - Return the number of fractional bits
  /// in a 'unsigned _Fract' type.
  unsigned getUnsignedFractScale() const {
    return PaddingOnUnsignedFixedPoint ? getFractScale() : getFractScale() + 1;
  }

  /// getUnsignedLongFractScale - Return the number of fractional bits
  /// in a 'unsigned long _Fract' type.
  unsigned getUnsignedLongFractScale() const {
    return PaddingOnUnsignedFixedPoint ? getLongFractScale()
                                       : getLongFractScale() + 1;
  }

  /// Determine whether the __int128 type is supported on this target.
  virtual bool hasInt128Type() const {
    return (getPointerWidth(0) >= 64) || getTargetOpts().ForceEnableInt128;
  } // FIXME

  /// Determine whether the _BitInt type is supported on this target. This
  /// limitation is put into place for ABI reasons.
  /// FIXME: _BitInt is a required type in C23, so there's not much utility in
  /// asking whether the target supported it or not; I think this should be
  /// removed once backends have been alerted to the type and have had the
  /// chance to do implementation work if needed.
  virtual bool hasBitIntType() const {
    return false;
  }

  /// Determine whether _Float16 is supported on this target.
  virtual bool hasLegalHalfType() const { return HasLegalHalfType; }

  /// Determine whether the __float128 type is supported on this target.
  virtual bool hasFloat128Type() const { return HasFloat128; }

  /// Determine whether the _Float16 type is supported on this target.
  virtual bool hasFloat16Type() const { return HasFloat16; }

  /// Determine whether the _BFloat16 type is supported on this target.
  virtual bool hasBFloat16Type() const { return HasBFloat16; }

  /// Determine whether the __ibm128 type is supported on this target.
  virtual bool hasIbm128Type() const { return HasIbm128; }

  /// Determine whether the long double type is supported on this target.
  virtual bool hasLongDoubleType() const { return HasLongDouble; }

  /// Determine whether return of a floating point value is supported
  /// on this target.
  virtual bool hasFPReturn() const { return HasFPReturn; }

  /// Determine whether constrained floating point is supported on this target.
  virtual bool hasStrictFP() const { return HasStrictFP; }

  /// Return the alignment that is the largest alignment ever used for any
  /// scalar/SIMD data type on the target machine you are compiling for
  /// (including types with an extended alignment requirement).
  unsigned getSuitableAlign() const { return SuitableAlign; }

  /// Return the default alignment for __attribute__((aligned)) on
  /// this target, to be used if no alignment value is specified.
  unsigned getDefaultAlignForAttributeAligned() const {
    return DefaultAlignForAttributeAligned;
  }

  /// getMinGlobalAlign - Return the minimum alignment of a global variable,
  /// unless its alignment is explicitly reduced via attributes.
  virtual unsigned getMinGlobalAlign (uint64_t) const {
    return MinGlobalAlign;
  }

  /// Return the largest alignment for which a suitably-sized allocation with
  /// '::operator new(size_t)' or 'malloc' is guaranteed to produce a
  /// correctly-aligned pointer.
  unsigned getNewAlign() const {
    return NewAlign ? NewAlign : std::max(LongDoubleAlign, LongLongAlign);
  }

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

  /// getBFloat16Width/Align/Format - Return the size/align/format of '__bf16'.
  unsigned getBFloat16Width() const { return BFloat16Width; }
  unsigned getBFloat16Align() const { return BFloat16Align; }
  const llvm::fltSemantics &getBFloat16Format() const { return *BFloat16Format; }

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

  /// getFloat128Width/Align/Format - Return the size/align/format of
  /// '__float128'.
  unsigned getFloat128Width() const { return 128; }
  unsigned getFloat128Align() const { return Float128Align; }
  const llvm::fltSemantics &getFloat128Format() const {
    return *Float128Format;
  }

  /// getIbm128Width/Align/Format - Return the size/align/format of
  /// '__ibm128'.
  unsigned getIbm128Width() const { return 128; }
  unsigned getIbm128Align() const { return Ibm128Align; }
  const llvm::fltSemantics &getIbm128Format() const { return *Ibm128Format; }

  /// Return the mangled code of long double.
  virtual const char *getLongDoubleMangling() const { return "e"; }

  /// Return the mangled code of __float128.
  virtual const char *getFloat128Mangling() const { return "g"; }

  /// Return the mangled code of __ibm128.
  virtual const char *getIbm128Mangling() const {
    llvm_unreachable("ibm128 not implemented on this target");
  }

  /// Return the mangled code of bfloat.
  virtual const char *getBFloat16Mangling() const {
    llvm_unreachable("bfloat not implemented on this target");
  }

  /// Return the value for the C99 FLT_EVAL_METHOD macro.
  virtual unsigned getFloatEvalMethod() const { return 0; }

  // getLargeArrayMinWidth/Align - Return the minimum array size that is
  // 'large' and its alignment.
  unsigned getLargeArrayMinWidth() const { return LargeArrayMinWidth; }
  unsigned getLargeArrayAlign() const { return LargeArrayAlign; }

  /// Return the maximum width lock-free atomic operation which will
  /// ever be supported for the given target
  unsigned getMaxAtomicPromoteWidth() const { return MaxAtomicPromoteWidth; }
  /// Return the maximum width lock-free atomic operation which can be
  /// inlined given the supported features of the given target.
  unsigned getMaxAtomicInlineWidth() const { return MaxAtomicInlineWidth; }
  /// Set the maximum inline or promote width lock-free atomic operation
  /// for the given target.
  virtual void setMaxAtomicWidth() {}
  /// Returns true if the given target supports lock-free atomic
  /// operations at the specified width and alignment.
  virtual bool hasBuiltinAtomic(uint64_t AtomicSizeInBits,
                                uint64_t AlignmentInBits) const {
    return AtomicSizeInBits <= AlignmentInBits &&
           AtomicSizeInBits <= getMaxAtomicInlineWidth() &&
           (AtomicSizeInBits <= getCharWidth() ||
            llvm::isPowerOf2_64(AtomicSizeInBits / getCharWidth()));
  }

  /// Return the maximum vector alignment supported for the given target.
  unsigned getMaxVectorAlign() const { return MaxVectorAlign; }
  /// Return default simd alignment for the given target. Generally, this
  /// value is type-specific, but this alignment can be used for most of the
  /// types for the given target.
  unsigned getSimdDefaultAlign() const { return SimdDefaultAlign; }

  unsigned getMaxOpenCLWorkGroupSize() const { return MaxOpenCLWorkGroupSize; }

  /// Return the alignment (in bits) of the thrown exception object. This is
  /// only meaningful for targets that allocate C++ exceptions in a system
  /// runtime, such as those using the Itanium C++ ABI.
  virtual unsigned getExnObjectAlignment() const {
    // Itanium says that an _Unwind_Exception has to be "double-word"
    // aligned (and thus the end of it is also so-aligned), meaning 16
    // bytes.  Of course, that was written for the actual Itanium,
    // which is a 64-bit platform.  Classically, the ABI doesn't really
    // specify the alignment on other platforms, but in practice
    // libUnwind declares the struct with __attribute__((aligned)), so
    // we assume that alignment here.  (It's generally 16 bytes, but
    // some targets overwrite it.)
    return getDefaultAlignForAttributeAligned();
  }

  /// Return the size of intmax_t and uintmax_t for this target, in bits.
  unsigned getIntMaxTWidth() const {
    return getTypeWidth(IntMaxType);
  }

  // Return the size of unwind_word for this target.
  virtual unsigned getUnwindWordWidth() const { return getPointerWidth(0); }

  /// Return the "preferred" register width on this target.
  virtual unsigned getRegisterWidth() const {
    // Currently we assume the register width on the target matches the pointer
    // width, we can introduce a new variable for this if/when some target wants
    // it.
    return PointerWidth;
  }

  /// \brief Returns the default value of the __USER_LABEL_PREFIX__ macro,
  /// which is the prefix given to user symbols by default.
  ///
  /// On most platforms this is "", but it is "_" on some.
  const char *getUserLabelPrefix() const { return UserLabelPrefix; }

  /// Returns the name of the mcount instrumentation function.
  const char *getMCountName() const {
    return MCountName;
  }

  /// Check if the Objective-C built-in boolean type should be signed
  /// char.
  ///
  /// Otherwise, if this returns false, the normal built-in boolean type
  /// should also be used for Objective-C.
  bool useSignedCharForObjCBool() const {
    return UseSignedCharForObjCBool;
  }
  void noSignedCharForObjCBool() {
    UseSignedCharForObjCBool = false;
  }

  /// Check whether the alignment of bit-field types is respected
  /// when laying out structures.
  bool useBitFieldTypeAlignment() const {
    return UseBitFieldTypeAlignment;
  }

  /// Check whether zero length bitfields should force alignment of
  /// the next member.
  bool useZeroLengthBitfieldAlignment() const {
    return UseZeroLengthBitfieldAlignment;
  }

  /// Check whether zero length bitfield alignment is respected if they are
  /// leading members.
  bool useLeadingZeroLengthBitfield() const {
    return UseLeadingZeroLengthBitfield;
  }

  /// Get the fixed alignment value in bits for a member that follows
  /// a zero length bitfield.
  unsigned getZeroLengthBitfieldBoundary() const {
    return ZeroLengthBitfieldBoundary;
  }

  /// Get the maximum alignment in bits for a static variable with
  /// aligned attribute.
  unsigned getMaxAlignedAttribute() const { return MaxAlignedAttribute; }

  /// Check whether explicit bitfield alignment attributes should be
  //  honored, as in "__attribute__((aligned(2))) int b : 1;".
  bool useExplicitBitFieldAlignment() const {
    return UseExplicitBitFieldAlignment;
  }

  /// Check whether this target support '\#pragma options align=mac68k'.
  bool hasAlignMac68kSupport() const {
    return HasAlignMac68kSupport;
  }

  /// Return the user string for the specified integer type enum.
  ///
  /// For example, SignedShort -> "short".
  static const char *getTypeName(IntType T);

  /// Return the constant suffix for the specified integer type enum.
  ///
  /// For example, SignedLong -> "L".
  const char *getTypeConstantSuffix(IntType T) const;

  /// Return the printf format modifier for the specified
  /// integer type enum.
  ///
  /// For example, SignedLong -> "l".
  static const char *getTypeFormatModifier(IntType T);

  /// Check whether the given real type should use the "fpret" flavor of
  /// Objective-C message passing on this target.
  bool useObjCFPRetForRealType(FloatModeKind T) const {
    return RealTypeUsesObjCFPRet & (1 << (int)T);
  }

  /// Check whether _Complex long double should use the "fp2ret" flavor
  /// of Objective-C message passing on this target.
  bool useObjCFP2RetForComplexLongDouble() const {
    return ComplexLongDoubleUsesFP2Ret;
  }

  /// Check whether llvm intrinsics such as llvm.convert.to.fp16 should be used
  /// to convert to and from __fp16.
  /// FIXME: This function should be removed once all targets stop using the
  /// conversion intrinsics.
  virtual bool useFP16ConversionIntrinsics() const {
    return true;
  }

  /// Specify if mangling based on address space map should be used or
  /// not for language specific address spaces
  bool useAddressSpaceMapMangling() const {
    return UseAddrSpaceMapMangling;
  }

  ///===---- Other target property query methods --------------------------===//

  /// Appends the target-specific \#define values for this
  /// target set to the specified buffer.
  virtual void getTargetDefines(const LangOptions &Opts,
                                MacroBuilder &Builder) const = 0;


  /// Return information about target-specific builtins for
  /// the current primary target, and info about which builtins are non-portable
  /// across the current set of primary and secondary targets.
  virtual ArrayRef<Builtin::Info> getTargetBuiltins() const = 0;

  /// Returns target-specific min and max values VScale_Range.
  virtual Optional<std::pair<unsigned, unsigned>>
  getVScaleRange(const LangOptions &LangOpts) const {
    return None;
  }
  /// The __builtin_clz* and __builtin_ctz* built-in
  /// functions are specified to have undefined results for zero inputs, but
  /// on targets that support these operations in a way that provides
  /// well-defined results for zero without loss of performance, it is a good
  /// idea to avoid optimizing based on that undef behavior.
  virtual bool isCLZForZeroUndef() const { return true; }

  /// Returns the kind of __builtin_va_list type that should be used
  /// with this target.
  virtual BuiltinVaListKind getBuiltinVaListKind() const = 0;

  /// Returns whether or not type \c __builtin_ms_va_list type is
  /// available on this target.
  bool hasBuiltinMSVaList() const { return HasBuiltinMSVaList; }

  /// Returns true for RenderScript.
  bool isRenderScriptTarget() const { return IsRenderScriptTarget; }

  /// Returns whether or not the AArch64 SVE built-in types are
  /// available on this target.
  bool hasAArch64SVETypes() const { return HasAArch64SVETypes; }

  /// Returns whether or not the RISC-V V built-in types are
  /// available on this target.
  bool hasRISCVVTypes() const { return HasRISCVVTypes; }

  /// Returns whether or not the AMDGPU unsafe floating point atomics are
  /// allowed.
  bool allowAMDGPUUnsafeFPAtomics() const { return AllowAMDGPUUnsafeFPAtomics; }

  /// For ARM targets returns a mask defining which coprocessors are configured
  /// as Custom Datapath.
  uint32_t getARMCDECoprocMask() const { return ARMCDECoprocMask; }

  /// Returns whether the passed in string is a valid clobber in an
  /// inline asm statement.
  ///
  /// This is used by Sema.
  bool isValidClobber(StringRef Name) const;

  /// Returns whether the passed in string is a valid register name
  /// according to GCC.
  ///
  /// This is used by Sema for inline asm statements.
  virtual bool isValidGCCRegisterName(StringRef Name) const;

  /// Returns the "normalized" GCC register name.
  ///
  /// ReturnCannonical true will return the register name without any additions
  /// such as "{}" or "%" in it's canonical form, for example:
  /// ReturnCanonical = true and Name = "rax", will return "ax".
  StringRef getNormalizedGCCRegisterName(StringRef Name,
                                         bool ReturnCanonical = false) const;

  virtual bool isSPRegName(StringRef) const { return false; }

  /// Extracts a register from the passed constraint (if it is a
  /// single-register constraint) and the asm label expression related to a
  /// variable in the input or output list of an inline asm statement.
  ///
  /// This function is used by Sema in order to diagnose conflicts between
  /// the clobber list and the input/output lists.
  virtual StringRef getConstraintRegister(StringRef Constraint,
                                          StringRef Expression) const {
    return "";
  }

  struct ConstraintInfo {
    enum {
      CI_None = 0x00,
      CI_AllowsMemory = 0x01,
      CI_AllowsRegister = 0x02,
      CI_ReadWrite = 0x04,         // "+r" output constraint (read and write).
      CI_HasMatchingInput = 0x08,  // This output operand has a matching input.
      CI_ImmediateConstant = 0x10, // This operand must be an immediate constant
      CI_EarlyClobber = 0x20,      // "&" output constraint (early clobber).
    };
    unsigned Flags;
    int TiedOperand;
    struct {
      int Min;
      int Max;
      bool isConstrained;
    } ImmRange;
    llvm::SmallSet<int, 4> ImmSet;

    std::string ConstraintStr;  // constraint: "=rm"
    std::string Name;           // Operand name: [foo] with no []'s.
  public:
    ConstraintInfo(StringRef ConstraintStr, StringRef Name)
        : Flags(0), TiedOperand(-1), ConstraintStr(ConstraintStr.str()),
          Name(Name.str()) {
      ImmRange.Min = ImmRange.Max = 0;
      ImmRange.isConstrained = false;
    }

    const std::string &getConstraintStr() const { return ConstraintStr; }
    const std::string &getName() const { return Name; }
    bool isReadWrite() const { return (Flags & CI_ReadWrite) != 0; }
    bool earlyClobber() { return (Flags & CI_EarlyClobber) != 0; }
    bool allowsRegister() const { return (Flags & CI_AllowsRegister) != 0; }
    bool allowsMemory() const { return (Flags & CI_AllowsMemory) != 0; }

    /// Return true if this output operand has a matching
    /// (tied) input operand.
    bool hasMatchingInput() const { return (Flags & CI_HasMatchingInput) != 0; }

    /// Return true if this input operand is a matching
    /// constraint that ties it to an output operand.
    ///
    /// If this returns true then getTiedOperand will indicate which output
    /// operand this is tied to.
    bool hasTiedOperand() const { return TiedOperand != -1; }
    unsigned getTiedOperand() const {
      assert(hasTiedOperand() && "Has no tied operand!");
      return (unsigned)TiedOperand;
    }

    bool requiresImmediateConstant() const {
      return (Flags & CI_ImmediateConstant) != 0;
    }
    bool isValidAsmImmediate(const llvm::APInt &Value) const {
      if (!ImmSet.empty())
        return Value.isSignedIntN(32) && ImmSet.contains(Value.getZExtValue());
      return !ImmRange.isConstrained ||
             (Value.sge(ImmRange.Min) && Value.sle(ImmRange.Max));
    }

    void setIsReadWrite() { Flags |= CI_ReadWrite; }
    void setEarlyClobber() { Flags |= CI_EarlyClobber; }
    void setAllowsMemory() { Flags |= CI_AllowsMemory; }
    void setAllowsRegister() { Flags |= CI_AllowsRegister; }
    void setHasMatchingInput() { Flags |= CI_HasMatchingInput; }
    void setRequiresImmediate(int Min, int Max) {
      Flags |= CI_ImmediateConstant;
      ImmRange.Min = Min;
      ImmRange.Max = Max;
      ImmRange.isConstrained = true;
    }
    void setRequiresImmediate(llvm::ArrayRef<int> Exacts) {
      Flags |= CI_ImmediateConstant;
      for (int Exact : Exacts)
        ImmSet.insert(Exact);
    }
    void setRequiresImmediate(int Exact) {
      Flags |= CI_ImmediateConstant;
      ImmSet.insert(Exact);
    }
    void setRequiresImmediate() {
      Flags |= CI_ImmediateConstant;
    }

    /// Indicate that this is an input operand that is tied to
    /// the specified output operand.
    ///
    /// Copy over the various constraint information from the output.
    void setTiedOperand(unsigned N, ConstraintInfo &Output) {
      Output.setHasMatchingInput();
      Flags = Output.Flags;
      TiedOperand = N;
      // Don't copy Name or constraint string.
    }
  };

  /// Validate register name used for global register variables.
  ///
  /// This function returns true if the register passed in RegName can be used
  /// for global register variables on this target. In addition, it returns
  /// true in HasSizeMismatch if the size of the register doesn't match the
  /// variable size passed in RegSize.
  virtual bool validateGlobalRegisterVariable(StringRef RegName,
                                              unsigned RegSize,
                                              bool &HasSizeMismatch) const {
    HasSizeMismatch = false;
    return true;
  }

  // validateOutputConstraint, validateInputConstraint - Checks that
  // a constraint is valid and provides information about it.
  // FIXME: These should return a real error instead of just true/false.
  bool validateOutputConstraint(ConstraintInfo &Info) const;
  bool validateInputConstraint(MutableArrayRef<ConstraintInfo> OutputConstraints,
                               ConstraintInfo &info) const;

  virtual bool validateOutputSize(const llvm::StringMap<bool> &FeatureMap,
                                  StringRef /*Constraint*/,
                                  unsigned /*Size*/) const {
    return true;
  }

  virtual bool validateInputSize(const llvm::StringMap<bool> &FeatureMap,
                                 StringRef /*Constraint*/,
                                 unsigned /*Size*/) const {
    return true;
  }
  virtual bool
  validateConstraintModifier(StringRef /*Constraint*/,
                             char /*Modifier*/,
                             unsigned /*Size*/,
                             std::string &/*SuggestedModifier*/) const {
    return true;
  }
  virtual bool
  validateAsmConstraint(const char *&Name,
                        TargetInfo::ConstraintInfo &info) const = 0;

  bool resolveSymbolicName(const char *&Name,
                           ArrayRef<ConstraintInfo> OutputConstraints,
                           unsigned &Index) const;

  // Constraint parm will be left pointing at the last character of
  // the constraint.  In practice, it won't be changed unless the
  // constraint is longer than one character.
  virtual std::string convertConstraint(const char *&Constraint) const {
    // 'p' defaults to 'r', but can be overridden by targets.
    if (*Constraint == 'p')
      return std::string("r");
    return std::string(1, *Constraint);
  }

  /// Replace some escaped characters with another string based on
  /// target-specific rules
  virtual llvm::Optional<std::string> handleAsmEscapedChar(char C) const {
    return llvm::None;
  }

  /// Returns a string of target-specific clobbers, in LLVM format.
  virtual const char *getClobbers() const = 0;

  /// Returns true if NaN encoding is IEEE 754-2008.
  /// Only MIPS allows a different encoding.
  virtual bool isNan2008() const {
    return true;
  }

  /// Returns the target triple of the primary target.
  const llvm::Triple &getTriple() const {
    return Triple;
  }

  /// Returns the target ID if supported.
  virtual llvm::Optional<std::string> getTargetID() const { return llvm::None; }

  const char *getDataLayoutString() const {
    assert(!DataLayoutString.empty() && "Uninitialized DataLayout!");
    return DataLayoutString.c_str();
  }

  struct GCCRegAlias {
    const char * const Aliases[5];
    const char * const Register;
  };

  struct AddlRegName {
    const char * const Names[5];
    const unsigned RegNum;
  };

  /// Does this target support "protected" visibility?
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

  /// Does this target aim for semantic compatibility with
  /// Microsoft C++ code using dllimport/export attributes?
  virtual bool shouldDLLImportComdatSymbols() const {
    return getTriple().isWindowsMSVCEnvironment() ||
           getTriple().isWindowsItaniumEnvironment() || getTriple().isPS4CPU();
  }

  // Does this target have PS4 specific dllimport/export handling?
  virtual bool hasPS4DLLImportExport() const {
    return getTriple().isPS4CPU() ||
           // Windows Itanium support allows for testing the SCEI flavour of
           // dllimport/export handling on a Windows system.
           (getTriple().isWindowsItaniumEnvironment() &&
            getTriple().getVendor() == llvm::Triple::SCEI);
  }

  /// Set forced language options.
  ///
  /// Apply changes to the target information with respect to certain
  /// language options which change the target configuration and adjust
  /// the language based on the target options where applicable.
  virtual void adjust(DiagnosticsEngine &Diags, LangOptions &Opts);

  /// Adjust target options based on codegen options.
  virtual void adjustTargetOptions(const CodeGenOptions &CGOpts,
                                   TargetOptions &TargetOpts) const {}

  /// Initialize the map with the default set of target features for the
  /// CPU this should include all legal feature strings on the target.
  ///
  /// \return False on error (invalid features).
  virtual bool initFeatureMap(llvm::StringMap<bool> &Features,
                              DiagnosticsEngine &Diags, StringRef CPU,
                              const std::vector<std::string> &FeatureVec) const;

  /// Get the ABI currently in use.
  virtual StringRef getABI() const { return StringRef(); }

  /// Get the C++ ABI currently in use.
  TargetCXXABI getCXXABI() const {
    return TheCXXABI;
  }

  /// Target the specified CPU.
  ///
  /// \return  False on error (invalid CPU name).
  virtual bool setCPU(const std::string &Name) {
    return false;
  }

  /// Fill a SmallVectorImpl with the valid values to setCPU.
  virtual void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const {}

  /// Fill a SmallVectorImpl with the valid values for tuning CPU.
  virtual void fillValidTuneCPUList(SmallVectorImpl<StringRef> &Values) const {
    fillValidCPUList(Values);
  }

  /// brief Determine whether this TargetInfo supports the given CPU name.
  virtual bool isValidCPUName(StringRef Name) const {
    return true;
  }

  /// brief Determine whether this TargetInfo supports the given CPU name for
  // tuning.
  virtual bool isValidTuneCPUName(StringRef Name) const {
    return isValidCPUName(Name);
  }

  /// brief Determine whether this TargetInfo supports tune in target attribute.
  virtual bool supportsTargetAttributeTune() const {
    return false;
  }

  /// Use the specified ABI.
  ///
  /// \return False on error (invalid ABI name).
  virtual bool setABI(const std::string &Name) {
    return false;
  }

  /// Use the specified unit for FP math.
  ///
  /// \return False on error (invalid unit name).
  virtual bool setFPMath(StringRef Name) {
    return false;
  }

  /// Check if target has a given feature enabled
  virtual bool hasFeatureEnabled(const llvm::StringMap<bool> &Features,
                                 StringRef Name) const {
    return Features.lookup(Name);
  }

  /// Enable or disable a specific target feature;
  /// the feature name must be valid.
  virtual void setFeatureEnabled(llvm::StringMap<bool> &Features,
                                 StringRef Name,
                                 bool Enabled) const {
    Features[Name] = Enabled;
  }

  /// Determine whether this TargetInfo supports the given feature.
  virtual bool isValidFeatureName(StringRef Feature) const {
    return true;
  }

  struct BranchProtectionInfo {
    LangOptions::SignReturnAddressScopeKind SignReturnAddr =
        LangOptions::SignReturnAddressScopeKind::None;
    LangOptions::SignReturnAddressKeyKind SignKey =
        LangOptions::SignReturnAddressKeyKind::AKey;
    bool BranchTargetEnforcement = false;
  };

  /// Determine if this TargetInfo supports the given branch protection
  /// specification
  virtual bool validateBranchProtection(StringRef Spec,
                                        BranchProtectionInfo &BPI,
                                        StringRef &Err) const {
    Err = "";
    return false;
  }

  /// Perform initialization based on the user configured
  /// set of features (e.g., +sse4).
  ///
  /// The list is guaranteed to have at most one entry per feature.
  ///
  /// The target may modify the features list, to change which options are
  /// passed onwards to the backend.
  /// FIXME: This part should be fixed so that we can change handleTargetFeatures
  /// to merely a TargetInfo initialization routine.
  ///
  /// \return  False on error.
  virtual bool handleTargetFeatures(std::vector<std::string> &Features,
                                    DiagnosticsEngine &Diags) {
    return true;
  }

  /// Determine whether the given target has the given feature.
  virtual bool hasFeature(StringRef Feature) const {
    return false;
  }

  /// Identify whether this target supports multiversioning of functions,
  /// which requires support for cpu_supports and cpu_is functionality.
  bool supportsMultiVersioning() const { return getTriple().isX86(); }

  /// Identify whether this target supports IFuncs.
  bool supportsIFunc() const { return getTriple().isOSBinFormatELF(); }

  // Validate the contents of the __builtin_cpu_supports(const char*)
  // argument.
  virtual bool validateCpuSupports(StringRef Name) const { return false; }

  // Return the target-specific priority for features/cpus/vendors so
  // that they can be properly sorted for checking.
  virtual unsigned multiVersionSortPriority(StringRef Name) const {
    return 0;
  }

  // Validate the contents of the __builtin_cpu_is(const char*)
  // argument.
  virtual bool validateCpuIs(StringRef Name) const { return false; }

  // Validate a cpu_dispatch/cpu_specific CPU option, which is a different list
  // from cpu_is, since it checks via features rather than CPUs directly.
  virtual bool validateCPUSpecificCPUDispatch(StringRef Name) const {
    return false;
  }

  // Get the character to be added for mangling purposes for cpu_specific.
  virtual char CPUSpecificManglingCharacter(StringRef Name) const {
    llvm_unreachable(
        "cpu_specific Multiversioning not implemented on this target");
  }

  // Get a list of the features that make up the CPU option for
  // cpu_specific/cpu_dispatch so that it can be passed to llvm as optimization
  // options.
  virtual void getCPUSpecificCPUDispatchFeatures(
      StringRef Name, llvm::SmallVectorImpl<StringRef> &Features) const {
    llvm_unreachable(
        "cpu_specific Multiversioning not implemented on this target");
  }

  // Get the cache line size of a given cpu. This method switches over
  // the given cpu and returns "None" if the CPU is not found.
  virtual Optional<unsigned> getCPUCacheLineSize() const { return None; }

  // Returns maximal number of args passed in registers.
  unsigned getRegParmMax() const {
    assert(RegParmMax < 7 && "RegParmMax value is larger than AST can handle");
    return RegParmMax;
  }

  /// Whether the target supports thread-local storage.
  bool isTLSSupported() const {
    return TLSSupported;
  }

  /// Return the maximum alignment (in bits) of a TLS variable
  ///
  /// Gets the maximum alignment (in bits) of a TLS variable on this target.
  /// Returns zero if there is no such constraint.
  unsigned getMaxTLSAlign() const { return MaxTLSAlign; }

  /// Whether target supports variable-length arrays.
  bool isVLASupported() const { return VLASupported; }

  /// Whether the target supports SEH __try.
  bool isSEHTrySupported() const {
    return getTriple().isOSWindows() &&
           (getTriple().isX86() ||
            getTriple().getArch() == llvm::Triple::aarch64);
  }

  /// Return true if {|} are normal characters in the asm string.
  ///
  /// If this returns false (the default), then {abc|xyz} is syntax
  /// that says that when compiling for asm variant #0, "abc" should be
  /// generated, but when compiling for asm variant #1, "xyz" should be
  /// generated.
  bool hasNoAsmVariants() const {
    return NoAsmVariants;
  }

  /// Return the register number that __builtin_eh_return_regno would
  /// return with the specified argument.
  /// This corresponds with TargetLowering's getExceptionPointerRegister
  /// and getExceptionSelectorRegister in the backend.
  virtual int getEHDataRegisterNumber(unsigned RegNo) const {
    return -1;
  }

  /// Return the section to use for C++ static initialization functions.
  virtual const char *getStaticInitSectionSpecifier() const {
    return nullptr;
  }

  const LangASMap &getAddressSpaceMap() const { return *AddrSpaceMap; }

  /// Map from the address space field in builtin description strings to the
  /// language address space.
  virtual LangAS getOpenCLBuiltinAddressSpace(unsigned AS) const {
    return getLangASFromTargetAS(AS);
  }

  /// Map from the address space field in builtin description strings to the
  /// language address space.
  virtual LangAS getCUDABuiltinAddressSpace(unsigned AS) const {
    return getLangASFromTargetAS(AS);
  }

  /// Return an AST address space which can be used opportunistically
  /// for constant global memory. It must be possible to convert pointers into
  /// this address space to LangAS::Default. If no such address space exists,
  /// this may return None, and such optimizations will be disabled.
  virtual llvm::Optional<LangAS> getConstantAddressSpace() const {
    return LangAS::Default;
  }

  // access target-specific GPU grid values that must be consistent between
  // host RTL (plugin), deviceRTL and clang.
  virtual const llvm::omp::GV &getGridValue() const {
    llvm_unreachable("getGridValue not implemented on this target");
  }

  /// Retrieve the name of the platform as it is used in the
  /// availability attribute.
  StringRef getPlatformName() const { return PlatformName; }

  /// Retrieve the minimum desired version of the platform, to
  /// which the program should be compiled.
  VersionTuple getPlatformMinVersion() const { return PlatformMinVersion; }

  bool isBigEndian() const { return BigEndian; }
  bool isLittleEndian() const { return !BigEndian; }

  /// Whether the option -fextend-arguments={32,64} is supported on the target.
  virtual bool supportsExtendIntArgs() const { return false; }

  /// Controls if __arithmetic_fence is supported in the targeted backend.
  virtual bool checkArithmeticFenceSupported() const { return false; }

  /// Gets the default calling convention for the given target and
  /// declaration context.
  virtual CallingConv getDefaultCallingConv() const {
    // Not all targets will specify an explicit calling convention that we can
    // express.  This will always do the right thing, even though it's not
    // an explicit calling convention.
    return CC_C;
  }

  enum CallingConvCheckResult {
    CCCR_OK,
    CCCR_Warning,
    CCCR_Ignore,
    CCCR_Error,
  };

  /// Determines whether a given calling convention is valid for the
  /// target. A calling convention can either be accepted, produce a warning
  /// and be substituted with the default calling convention, or (someday)
  /// produce an error (such as using thiscall on a non-instance function).
  virtual CallingConvCheckResult checkCallingConvention(CallingConv CC) const {
    switch (CC) {
      default:
        return CCCR_Warning;
      case CC_C:
        return CCCR_OK;
    }
  }

  enum CallingConvKind {
    CCK_Default,
    CCK_ClangABI4OrPS4,
    CCK_MicrosoftWin64
  };

  virtual CallingConvKind getCallingConvKind(bool ClangABICompat4) const;

  /// Controls if __builtin_longjmp / __builtin_setjmp can be lowered to
  /// llvm.eh.sjlj.longjmp / llvm.eh.sjlj.setjmp.
  virtual bool hasSjLjLowering() const {
    return false;
  }

  /// Check if the target supports CFProtection branch.
  virtual bool
  checkCFProtectionBranchSupported(DiagnosticsEngine &Diags) const;

  /// Check if the target supports CFProtection branch.
  virtual bool
  checkCFProtectionReturnSupported(DiagnosticsEngine &Diags) const;

  /// Whether target allows to overalign ABI-specified preferred alignment
  virtual bool allowsLargerPreferedTypeAlignment() const { return true; }

  /// Whether target defaults to the `power` alignment rules of AIX.
  virtual bool defaultsToAIXPowerAlignment() const { return false; }

  /// Set supported OpenCL extensions and optional core features.
  virtual void setSupportedOpenCLOpts() {}

  virtual void supportAllOpenCLOpts(bool V = true) {
#define OPENCLEXTNAME(Ext)                                                     \
  setFeatureEnabled(getTargetOpts().OpenCLFeaturesMap, #Ext, V);
#include "clang/Basic/OpenCLExtensions.def"
  }

  /// Set supported OpenCL extensions as written on command line
  virtual void setCommandLineOpenCLOpts() {
    for (const auto &Ext : getTargetOpts().OpenCLExtensionsAsWritten) {
      bool IsPrefixed = (Ext[0] == '+' || Ext[0] == '-');
      std::string Name = IsPrefixed ? Ext.substr(1) : Ext;
      bool V = IsPrefixed ? Ext[0] == '+' : true;

      if (Name == "all") {
        supportAllOpenCLOpts(V);
        continue;
      }

      getTargetOpts().OpenCLFeaturesMap[Name] = V;
    }
  }

  /// Get supported OpenCL extensions and optional core features.
  llvm::StringMap<bool> &getSupportedOpenCLOpts() {
    return getTargetOpts().OpenCLFeaturesMap;
  }

  /// Get const supported OpenCL extensions and optional core features.
  const llvm::StringMap<bool> &getSupportedOpenCLOpts() const {
    return getTargetOpts().OpenCLFeaturesMap;
  }

  /// Get address space for OpenCL type.
  virtual LangAS getOpenCLTypeAddrSpace(OpenCLTypeKind TK) const;

  /// \returns Target specific vtbl ptr address space.
  virtual unsigned getVtblPtrAddressSpace() const {
    return 0;
  }

  /// \returns If a target requires an address within a target specific address
  /// space \p AddressSpace to be converted in order to be used, then return the
  /// corresponding target specific DWARF address space.
  ///
  /// \returns Otherwise return None and no conversion will be emitted in the
  /// DWARF.
  virtual Optional<unsigned> getDWARFAddressSpace(unsigned AddressSpace) const {
    return None;
  }

  /// \returns The version of the SDK which was used during the compilation if
  /// one was specified, or an empty version otherwise.
  const llvm::VersionTuple &getSDKVersion() const {
    return getTargetOpts().SDKVersion;
  }

  /// Check the target is valid after it is fully initialized.
  virtual bool validateTarget(DiagnosticsEngine &Diags) const {
    return true;
  }

  /// Check that OpenCL target has valid options setting based on OpenCL
  /// version.
  virtual bool validateOpenCLTarget(const LangOptions &Opts,
                                    DiagnosticsEngine &Diags) const;

  virtual void setAuxTarget(const TargetInfo *Aux) {}

  /// Whether target allows debuginfo types for decl only variables/functions.
  virtual bool allowDebugInfoForExternalRef() const { return false; }

protected:
  /// Copy type and layout related info.
  void copyAuxTarget(const TargetInfo *Aux);
  virtual uint64_t getPointerWidthV(unsigned AddrSpace) const {
    return PointerWidth;
  }
  virtual uint64_t getPointerAlignV(unsigned AddrSpace) const {
    return PointerAlign;
  }
  virtual enum IntType getPtrDiffTypeV(unsigned AddrSpace) const {
    return PtrDiffType;
  }
  virtual ArrayRef<const char *> getGCCRegNames() const = 0;
  virtual ArrayRef<GCCRegAlias> getGCCRegAliases() const = 0;
  virtual ArrayRef<AddlRegName> getGCCAddlRegNames() const {
    return None;
  }

 private:
  // Assert the values for the fractional and integral bits for each fixed point
  // type follow the restrictions given in clause 6.2.6.3 of N1169.
  void CheckFixedPointBits() const;
};

}  // end namespace clang

#endif
