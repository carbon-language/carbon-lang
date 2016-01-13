//===-- CodeView.h - On-disk record types for CodeView ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides data structures useful for consuming on-disk
/// CodeView. It is based on information published by Microsoft at
/// https://github.com/Microsoft/microsoft-pdb/.
///
//===----------------------------------------------------------------------===//

// FIXME: Find a home for this in include/llvm/DebugInfo/CodeView/.

#ifndef LLVM_READOBJ_CODEVIEW_H
#define LLVM_READOBJ_CODEVIEW_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

/// A Symbols subsection is a sequence of SymRecords. Advancing by 'len'
/// bytes will find the next SymRecord. These are the possible types of a
/// record. Equivalent to SYM_ENUM_e in cvinfo.h.
enum SymType : uint16_t {
#define SYMBOL_TYPE(ename, value) ename = value,
#include "CVSymbolTypes.def"
};

/// Generic record compatible with all symbol records.
struct SymRecord {
  ulittle16_t RecordLength; // Record length, starting from the next field
  ulittle16_t RecordType;   // Record type (SymType)
  // Symbol data follows.
};

/// Corresponds to the CV_PROCFLAGS bitfield.
enum ProcFlags : uint8_t {
  HasFP = 1 << 0,
  HasIRET = 1 << 1,
  HasFRET = 1 << 2,
  IsNoReturn = 1 << 3,
  IsUnreachable = 1 << 4,
  HasCustomCallingConv = 1 << 5,
  IsNoInline = 1 << 6,
  HasOptimizedDebugInfo = 1 << 7,
};

// S_GPROC32, S_LPROC32, S_GPROC32_ID, S_LPROC32_ID, S_LPROC32_DPC or
// S_LPROC32_DPC_ID
struct ProcSym {
  ulittle32_t PtrParent;
  ulittle32_t PtrEnd;
  ulittle32_t PtrNext;
  ulittle32_t CodeSize;
  ulittle32_t DbgStart;
  ulittle32_t DbgEnd;
  TypeIndex FunctionType;
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  uint8_t Flags; // CV_PROCFLAGS
  // Name: The null-terminated name follows.
};

// S_INLINESITE
struct InlineSiteSym {
  ulittle32_t PtrParent;
  ulittle32_t PtrEnd;
  TypeIndex Inlinee;
  // BinaryAnnotations
};

// S_LOCAL
struct LocalSym {
  TypeIndex Type;
  ulittle16_t Flags;
  enum : uint16_t {
    IsParameter = 1 << 0,
    IsAddressTaken = 1 << 1,
    IsCompilerGenerated = 1 << 2,
    IsAggregate = 1 << 3,
    IsAggregated = 1 << 4,
    IsAliased = 1 << 5,
    IsAlias = 1 << 6,
    IsReturnValue = 1 << 7,
    IsOptimizedOut = 1 << 8,
    IsEnregisteredGlobal = 1 << 9,
    IsEnregisteredStatic = 1 << 10,
  };
  // Name: The null-terminated name follows.
};

// S_BLOCK32
struct BlockSym {
  ulittle32_t PtrParent;
  ulittle32_t PtrEnd;
  ulittle32_t CodeSize;
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  // Name: The null-terminated name follows.
};

// S_LABEL32
struct LabelSym {
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  uint8_t Flags; // CV_PROCFLAGS
  // Name: The null-terminated name follows.
};

// S_OBJNAME
struct ObjNameSym {
  ulittle32_t Signature;
  // Name: The null-terminated name follows.
};

// S_COMPILE3
struct CompileSym3 {
  ulittle32_t flags;
  uint8_t getLanguage() const { return flags & 0xff; }
  enum Flags : uint32_t {
    EC = 1 << 8,
    NoDbgInfo = 1 << 9,
    LTCG = 1 << 10,
    NoDataAlign = 1 << 11,
    ManagedPresent = 1 << 12,
    SecurityChecks = 1 << 13,
    HotPatch = 1 << 14,
    CVTCIL = 1 << 15,
    MSILModule = 1 << 16,
    Sdl = 1 << 17,
    PGO = 1 << 18,
    Exp = 1 << 19,
  };
  ulittle16_t Machine; // CPUType
  ulittle16_t VersionFrontendMajor;
  ulittle16_t VersionFrontendMinor;
  ulittle16_t VersionFrontendBuild;
  ulittle16_t VersionFrontendQFE;
  ulittle16_t VersionBackendMajor;
  ulittle16_t VersionBackendMinor;
  ulittle16_t VersionBackendBuild;
  ulittle16_t VersionBackendQFE;
  // VersionString: The null-terminated version string follows.
};

// S_FRAMEPROC
struct FrameProcSym {
  ulittle32_t TotalFrameBytes;
  ulittle32_t PaddingFrameBytes;
  ulittle32_t OffsetToPadding;
  ulittle32_t BytesOfCalleeSavedRegisters;
  ulittle32_t OffsetOfExceptionHandler;
  ulittle16_t SectionIdOfExceptionHandler;
  ulittle32_t Flags;
};

// S_CALLSITEINFO
struct CallSiteInfoSym {
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  ulittle16_t Reserved;
  TypeIndex Type;
};

// S_HEAPALLOCSITE
struct HeapAllocationSiteSym {
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  ulittle16_t CallInstructionSize;
  TypeIndex Type;
};

// S_FRAMECOOKIE
struct FrameCookieSym {
  ulittle32_t CodeOffset;
  ulittle16_t Register;
  ulittle16_t CookieKind;

  enum : uint16_t {
    Copy,
    XorStackPointer,
    XorFramePointer,
    XorR13,
  };
};

// S_UDT, S_COBOLUDT
struct UDTSym {
  TypeIndex Type; // Type of the UDT
  // Name: The null-terminated name follows.
};

// S_BUILDINFO
struct BuildInfoSym {
  ulittle32_t BuildId;
};

// S_BPREL32
struct BPRelativeSym {
  ulittle32_t Offset; // Offset from the base pointer register
  TypeIndex Type;     // Type of the variable
  // Name: The null-terminated name follows.
};

// S_REGREL32
struct RegRelativeSym {
  ulittle32_t Offset;   // Offset from the register
  TypeIndex Type;       // Type of the variable
  ulittle16_t Register; // Register to which the variable is relative
  // Name: The null-terminated name follows.
};

// S_CONSTANT, S_MANCONSTANT
struct ConstantSym {
  TypeIndex Type;
  // Value: The value of the constant.
  // Name: The null-terminated name follows.
};

// S_LDATA32, S_GDATA32, S_LMANDATA, S_GMANDATA
struct DataSym {
  TypeIndex Type;
  ulittle32_t DataOffset;
  ulittle16_t Segment;
  // Name: The null-terminated name follows.
};

// S_LTHREAD32, S_GTHREAD32
struct ThreadLocalDataSym {
  TypeIndex Type;
  ulittle32_t DataOffset;
  ulittle16_t Segment;
  // Name: The null-terminated name follows.
};

/// Data in the the SUBSEC_FRAMEDATA subection.
struct FrameData {
  ulittle32_t RvaStart;
  ulittle32_t CodeSize;
  ulittle32_t LocalSize;
  ulittle32_t ParamsSize;
  ulittle32_t MaxStackSize;
  ulittle32_t FrameFunc;
  ulittle16_t PrologSize;
  ulittle16_t SavedRegsSize;
  ulittle32_t Flags;
  enum : uint32_t {
    HasSEH = 1 << 0,
    HasEH = 1 << 1,
    IsFunctionStart = 1 << 2,
  };
};

//===----------------------------------------------------------------------===//
// On-disk representation of type information

/// Indicates the kind of TypeRecord we're dealing with here. The documentation
/// and headers talk about this as the "leaf" type.
enum LeafType : uint16_t {
#define LEAF_TYPE(name, val) name = val,
#include "CVLeafTypes.def"
};

// A CodeView type stream is a sequence of TypeRecords. Records larger than
// 65536 must chain on to a second record. Each TypeRecord is followed by one of
// the leaf types described below.
struct TypeRecord {
  ulittle16_t Len;  // Type record length, starting from &Leaf.
  ulittle16_t Leaf; // Type record kind (LeafType)
};

// LF_TYPESERVER2
struct TypeServer2 {
  char Signature[16];  // GUID
  ulittle32_t Age;
  // Name: Name of the PDB as a null-terminated string
};

// LF_STRING_ID
struct StringId {
  TypeIndex id;
};

// LF_FUNC_ID
struct FuncId {
  TypeIndex ParentScope;
  TypeIndex FunctionType;
  // Name: The null-terminated name follows.
};

// LF_CLASS, LF_STRUCT, LF_INTERFACE
struct ClassType {
  ulittle16_t MemberCount; // Number of members in FieldList.
  ulittle16_t Properties;  // ClassOptions bitset
  TypeIndex FieldList;     // LF_FIELDLIST: List of all kinds of members
  TypeIndex DerivedFrom;   // LF_DERIVED: List of known derived classes
  TypeIndex VShape;        // LF_VTSHAPE: Shape of the vftable
  // SizeOf: The 'sizeof' the UDT in bytes is encoded as an LF_NUMERIC integer.
  // Name: The null-terminated name follows.
};

// LF_UNION
struct UnionType {
  ulittle16_t MemberCount; // Number of members in FieldList.
  ulittle16_t Properties;  // ClassOptions bitset
  TypeIndex FieldList;     // LF_FIELDLIST: List of all kinds of members
  // SizeOf: The 'sizeof' the UDT in bytes is encoded as an LF_NUMERIC integer.
  // Name: The null-terminated name follows.
};

// LF_POINTER
struct PointerType {
  TypeIndex PointeeType;
  ulittle32_t Attrs; // pointer attributes
  // if pointer to member:
  //   PointerToMemberTail

  PointerKind getPtrKind() const { return PointerKind(Attrs & 0x1f); }
  PointerMode getPtrMode() const { return PointerMode((Attrs >> 5) & 0x07); }
  bool isFlat() const { return Attrs & (1 << 8); }
  bool isVolatile() const { return Attrs & (1 << 9); }
  bool isConst() const { return Attrs & (1 << 10); }
  bool isUnaligned() const { return Attrs & (1 << 11); }

  bool isPointerToDataMember() const {
    return getPtrMode() == PointerMode::PointerToDataMember;
  }
  bool isPointerToMemberFunction() const {
    return getPtrMode() == PointerMode::PointerToMemberFunction;
  }
  bool isPointerToMember() const {
    return isPointerToMemberFunction() || isPointerToDataMember();
  }
};

struct PointerToMemberTail {
  TypeIndex ClassType;
  ulittle16_t Representation; // PointerToMemberRepresentation
};

/// In Clang parlance, these are "qualifiers".  LF_MODIFIER
struct TypeModifier {
  TypeIndex ModifiedType;
  ulittle16_t Modifiers; // ModifierOptions
};

// LF_VTSHAPE
struct VTableShape {
  // Number of vftable entries. Each method may have more than one entry due to
  // things like covariant return types.
  ulittle16_t VFEntryCount;
  // Descriptors[]: 4-bit virtual method descriptors of type CV_VTS_desc_e.
};

// LF_UDT_SRC_LINE
struct UDTSrcLine {
  TypeIndex UDT;        // The user-defined type
  TypeIndex SourceFile; // StringID containing the source filename
  ulittle32_t LineNumber;
};

// LF_ARGLIST, LF_SUBSTR_LIST
struct ArgList {
  ulittle32_t NumArgs; // Number of arguments
  // ArgTypes[]: Type indicies of arguments
};

// LF_BUILDINFO
struct BuildInfo {
  ulittle16_t NumArgs; // Number of arguments
  // ArgTypes[]: Type indicies of arguments
};

// LF_ENUM
struct EnumType {
  ulittle16_t NumEnumerators; // Number of enumerators
  ulittle16_t Properties;
  TypeIndex UnderlyingType;
  TypeIndex FieldListType;
  // Name: The null-terminated name follows.
};

// LF_ARRAY
struct ArrayType {
  TypeIndex ElementType;
  TypeIndex IndexType;
  // SizeOf: LF_NUMERIC encoded size in bytes. Not element count!
  // Name: The null-terminated name follows.
};

// LF_VFTABLE
struct VFTableType {
  TypeIndex CompleteClass;     // Class that owns this vftable.
  TypeIndex OverriddenVFTable; // VFTable that this overrides.
  ulittle32_t VFPtrOffset;     // VFPtr offset in CompleteClass
  ulittle32_t NamesLen;        // Length of subsequent names array in bytes.
  // Names: A sequence of null-terminated strings. First string is vftable
  // names.
};

// LF_MFUNC_ID
struct MemberFuncId {
  TypeIndex ClassType;
  TypeIndex FunctionType;
  // Name: The null-terminated name follows.
};

// LF_PROCEDURE
struct ProcedureType {
  TypeIndex ReturnType;
  CallingConvention CallConv;
  FunctionOptions Options;
  ulittle16_t NumParameters;
  TypeIndex ArgListType;
};

// LF_MFUNCTION
struct MemberFunctionType {
  TypeIndex ReturnType;
  TypeIndex ClassType;
  TypeIndex ThisType;
  CallingConvention CallConv;
  FunctionOptions Options;
  ulittle16_t NumParameters;
  TypeIndex ArgListType;
  little32_t ThisAdjustment;
};

//===----------------------------------------------------------------------===//
// Field list records, which do not include leafs or sizes

/// Equvalent to CV_fldattr_t in cvinfo.h.
struct MemberAttributes {
  ulittle16_t Attrs;

  /// Get the access specifier. Valid for any kind of member.
  MemberAccess getAccess() const {
    return MemberAccess(unsigned(Attrs) & unsigned(MethodOptions::AccessMask));
  }

  /// Indicates if a method is defined with friend, virtual, static, etc.
  MethodKind getMethodKind() const {
    return MethodKind(
        (unsigned(Attrs) & unsigned(MethodOptions::MethodKindMask)) >> 2);
  }

  /// Get the flags that are not included in access control or method
  /// properties.
  MethodOptions getFlags() const {
    return MethodOptions(
        unsigned(Attrs) &
        ~unsigned(MethodOptions::AccessMask | MethodOptions::MethodKindMask));
  }

  /// Is this method virtual.
  bool isVirtual() const {
    auto MP = getMethodKind();
    return MP != MethodKind::Vanilla && MP != MethodKind::Friend &&
           MP != MethodKind::Static;
  }

  /// Does this member introduce a new virtual method.
  bool isIntroducedVirtual() const {
    auto MP = getMethodKind();
    return MP == MethodKind::IntroducingVirtual ||
           MP == MethodKind::PureIntroducingVirtual;
  }
};

// LF_NESTTYPE
struct NestedType {
  ulittle16_t Pad0; // Should be zero
  TypeIndex Type;   // Type index of nested type
  // Name: Null-terminated string
};

// LF_ONEMETHOD
struct OneMethod {
  MemberAttributes Attrs;
  TypeIndex Type;
  // If is introduced virtual method:
  //   VFTableOffset: int32_t offset in vftable
  // Name: Null-terminated string

  MethodKind getMethodKind() const {
    return Attrs.getMethodKind();
  }

  bool isVirtual() const { return Attrs.isVirtual(); }
  bool isIntroducedVirtual() const { return Attrs.isIntroducedVirtual(); }
};

struct MethodListEntry {
  MemberAttributes Attrs;
  ulittle16_t Padding;

  TypeIndex Type;
  // If is introduced virtual method:
  //   VFTableOffset: int32_t offset in vftable

  MethodKind getMethodKind() const {
    return Attrs.getMethodKind();
  }

  bool isVirtual() const { return Attrs.isVirtual(); }
  bool isIntroducedVirtual() const { return Attrs.isIntroducedVirtual(); }
};

/// For method overload sets.  LF_METHOD
struct OverloadedMethod {
  ulittle16_t MethodCount; // Size of overload set
  TypeIndex MethList;      // Type index of methods in overload set
  // Name: Null-terminated string
};

// LF_VFUNCTAB
struct VirtualFunctionPointer {
  ulittle16_t Pad0;
  TypeIndex Type;   // Type of vfptr
};

// LF_MEMBER
struct DataMember {
  MemberAttributes Attrs; // Access control attributes, etc
  TypeIndex Type;
  // FieldOffset: LF_NUMERIC encoded byte offset
  // Name: Null-terminated string
};

// LF_STMEMBER
struct StaticDataMember {
  MemberAttributes Attrs; // Access control attributes, etc
  TypeIndex Type;
  // Name: Null-terminated string
};

// LF_ENUMERATE
struct Enumerator {
  MemberAttributes Attrs; // Access control attributes, etc
  // EnumValue: LF_NUMERIC encoded enumerator value
  // Name: Null-terminated string
};

// LF_BCLASS, LF_BINTERFACE
struct BaseClass {
  MemberAttributes Attrs; // Access control attributes, etc
  TypeIndex BaseType;     // Base class type
  // BaseOffset: LF_NUMERIC encoded byte offset of base from derived.
};

// LF_VBCLASS | LV_IVBCLASS
struct VirtualBaseClass {
  MemberAttributes Attrs; // Access control attributes, etc.
  TypeIndex BaseType;     // Base class type
  TypeIndex VBPtrType;    // Virtual base pointer type
  // VBPtrOffset: Offset of vbptr from vfptr encoded as LF_NUMERIC.
  // VBTableIndex: Index of vbase within vbtable encoded as LF_NUMERIC.
};

} // namespace codeview
} // namespace llvm

#endif // LLVM_READOBJ_CODEVIEW_H
