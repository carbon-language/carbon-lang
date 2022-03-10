//===- llvm/CodeGen/DwarfDebug.h - Dwarf Debug Framework --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFDEBUG_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFDEBUG_H

#include "AddressPool.h"
#include "DebugLocStream.h"
#include "DebugLocEntry.h"
#include "DwarfFile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/DbgEntityHistoryCalculator.h"
#include "llvm/CodeGen/DebugHandlerBase.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Metadata.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace llvm {

class AsmPrinter;
class ByteStreamer;
class DIE;
class DwarfCompileUnit;
class DwarfExpression;
class DwarfTypeUnit;
class DwarfUnit;
class LexicalScope;
class MachineFunction;
class MCSection;
class MCSymbol;
class Module;

//===----------------------------------------------------------------------===//
/// This class is defined as the common parent of DbgVariable and DbgLabel
/// such that it could levarage polymorphism to extract common code for
/// DbgVariable and DbgLabel.
class DbgEntity {
public:
  enum DbgEntityKind {
    DbgVariableKind,
    DbgLabelKind
  };

private:
  const DINode *Entity;
  const DILocation *InlinedAt;
  DIE *TheDIE = nullptr;
  const DbgEntityKind SubclassID;

public:
  DbgEntity(const DINode *N, const DILocation *IA, DbgEntityKind ID)
      : Entity(N), InlinedAt(IA), SubclassID(ID) {}
  virtual ~DbgEntity() = default;

  /// Accessors.
  /// @{
  const DINode *getEntity() const { return Entity; }
  const DILocation *getInlinedAt() const { return InlinedAt; }
  DIE *getDIE() const { return TheDIE; }
  DbgEntityKind getDbgEntityID() const { return SubclassID; }
  /// @}

  void setDIE(DIE &D) { TheDIE = &D; }

  static bool classof(const DbgEntity *N) {
    switch (N->getDbgEntityID()) {
    case DbgVariableKind:
    case DbgLabelKind:
      return true;
    }
    llvm_unreachable("Invalid DbgEntityKind");
  }
};

//===----------------------------------------------------------------------===//
/// This class is used to track local variable information.
///
/// Variables can be created from allocas, in which case they're generated from
/// the MMI table.  Such variables can have multiple expressions and frame
/// indices.
///
/// Variables can be created from \c DBG_VALUE instructions.  Those whose
/// location changes over time use \a DebugLocListIndex, while those with a
/// single location use \a ValueLoc and (optionally) a single entry of \a Expr.
///
/// Variables that have been optimized out use none of these fields.
class DbgVariable : public DbgEntity {
  /// Index of the entry list in DebugLocs.
  unsigned DebugLocListIndex = ~0u;
  /// DW_OP_LLVM_tag_offset value from DebugLocs.
  Optional<uint8_t> DebugLocListTagOffset;

  /// Single value location description.
  std::unique_ptr<DbgValueLoc> ValueLoc = nullptr;

  struct FrameIndexExpr {
    int FI;
    const DIExpression *Expr;
  };
  mutable SmallVector<FrameIndexExpr, 1>
      FrameIndexExprs; /// Frame index + expression.

public:
  /// Construct a DbgVariable.
  ///
  /// Creates a variable without any DW_AT_location.  Call \a initializeMMI()
  /// for MMI entries, or \a initializeDbgValue() for DBG_VALUE instructions.
  DbgVariable(const DILocalVariable *V, const DILocation *IA)
      : DbgEntity(V, IA, DbgVariableKind) {}

  /// Initialize from the MMI table.
  void initializeMMI(const DIExpression *E, int FI) {
    assert(FrameIndexExprs.empty() && "Already initialized?");
    assert(!ValueLoc.get() && "Already initialized?");

    assert((!E || E->isValid()) && "Expected valid expression");
    assert(FI != std::numeric_limits<int>::max() && "Expected valid index");

    FrameIndexExprs.push_back({FI, E});
  }

  // Initialize variable's location.
  void initializeDbgValue(DbgValueLoc Value) {
    assert(FrameIndexExprs.empty() && "Already initialized?");
    assert(!ValueLoc && "Already initialized?");
    assert(!Value.getExpression()->isFragment() && "Fragments not supported.");

    ValueLoc = std::make_unique<DbgValueLoc>(Value);
    if (auto *E = ValueLoc->getExpression())
      if (E->getNumElements())
        FrameIndexExprs.push_back({0, E});
  }

  /// Initialize from a DBG_VALUE instruction.
  void initializeDbgValue(const MachineInstr *DbgValue);

  // Accessors.
  const DILocalVariable *getVariable() const {
    return cast<DILocalVariable>(getEntity());
  }

  const DIExpression *getSingleExpression() const {
    assert(ValueLoc.get() && FrameIndexExprs.size() <= 1);
    return FrameIndexExprs.size() ? FrameIndexExprs[0].Expr : nullptr;
  }

  void setDebugLocListIndex(unsigned O) { DebugLocListIndex = O; }
  unsigned getDebugLocListIndex() const { return DebugLocListIndex; }
  void setDebugLocListTagOffset(uint8_t O) { DebugLocListTagOffset = O; }
  Optional<uint8_t> getDebugLocListTagOffset() const { return DebugLocListTagOffset; }
  StringRef getName() const { return getVariable()->getName(); }
  const DbgValueLoc *getValueLoc() const { return ValueLoc.get(); }
  /// Get the FI entries, sorted by fragment offset.
  ArrayRef<FrameIndexExpr> getFrameIndexExprs() const;
  bool hasFrameIndexExprs() const { return !FrameIndexExprs.empty(); }
  void addMMIEntry(const DbgVariable &V);

  // Translate tag to proper Dwarf tag.
  dwarf::Tag getTag() const {
    // FIXME: Why don't we just infer this tag and store it all along?
    if (getVariable()->isParameter())
      return dwarf::DW_TAG_formal_parameter;

    return dwarf::DW_TAG_variable;
  }

  /// Return true if DbgVariable is artificial.
  bool isArtificial() const {
    if (getVariable()->isArtificial())
      return true;
    if (getType()->isArtificial())
      return true;
    return false;
  }

  bool isObjectPointer() const {
    if (getVariable()->isObjectPointer())
      return true;
    if (getType()->isObjectPointer())
      return true;
    return false;
  }

  bool hasComplexAddress() const {
    assert(ValueLoc.get() && "Expected DBG_VALUE, not MMI variable");
    assert((FrameIndexExprs.empty() ||
            (FrameIndexExprs.size() == 1 &&
             FrameIndexExprs[0].Expr->getNumElements())) &&
           "Invalid Expr for DBG_VALUE");
    return !FrameIndexExprs.empty();
  }

  const DIType *getType() const;

  static bool classof(const DbgEntity *N) {
    return N->getDbgEntityID() == DbgVariableKind;
  }
};

//===----------------------------------------------------------------------===//
/// This class is used to track label information.
///
/// Labels are collected from \c DBG_LABEL instructions.
class DbgLabel : public DbgEntity {
  const MCSymbol *Sym;                  /// Symbol before DBG_LABEL instruction.

public:
  /// We need MCSymbol information to generate DW_AT_low_pc.
  DbgLabel(const DILabel *L, const DILocation *IA, const MCSymbol *Sym = nullptr)
      : DbgEntity(L, IA, DbgLabelKind), Sym(Sym) {}

  /// Accessors.
  /// @{
  const DILabel *getLabel() const { return cast<DILabel>(getEntity()); }
  const MCSymbol *getSymbol() const { return Sym; }

  StringRef getName() const { return getLabel()->getName(); }
  /// @}

  /// Translate tag to proper Dwarf tag.
  dwarf::Tag getTag() const {
    return dwarf::DW_TAG_label;
  }

  static bool classof(const DbgEntity *N) {
    return N->getDbgEntityID() == DbgLabelKind;
  }
};

/// Used for tracking debug info about call site parameters.
class DbgCallSiteParam {
private:
  unsigned Register; ///< Parameter register at the callee entry point.
  DbgValueLoc Value; ///< Corresponding location for the parameter value at
                     ///< the call site.
public:
  DbgCallSiteParam(unsigned Reg, DbgValueLoc Val)
      : Register(Reg), Value(Val) {
    assert(Reg && "Parameter register cannot be undef");
  }

  unsigned getRegister() const { return Register; }
  DbgValueLoc getValue() const { return Value; }
};

/// Collection used for storing debug call site parameters.
using ParamSet = SmallVector<DbgCallSiteParam, 4>;

/// Helper used to pair up a symbol and its DWARF compile unit.
struct SymbolCU {
  SymbolCU(DwarfCompileUnit *CU, const MCSymbol *Sym) : Sym(Sym), CU(CU) {}

  const MCSymbol *Sym;
  DwarfCompileUnit *CU;
};

/// The kind of accelerator tables we should emit.
enum class AccelTableKind {
  Default, ///< Platform default.
  None,    ///< None.
  Apple,   ///< .apple_names, .apple_namespaces, .apple_types, .apple_objc.
  Dwarf,   ///< DWARF v5 .debug_names.
};

/// Collects and handles dwarf debug information.
class DwarfDebug : public DebugHandlerBase {
  /// All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  /// Maps MDNode with its corresponding DwarfCompileUnit.
  MapVector<const MDNode *, DwarfCompileUnit *> CUMap;

  /// Maps a CU DIE with its corresponding DwarfCompileUnit.
  DenseMap<const DIE *, DwarfCompileUnit *> CUDieMap;

  /// List of all labels used in aranges generation.
  std::vector<SymbolCU> ArangeLabels;

  /// Size of each symbol emitted (for those symbols that have a specific size).
  DenseMap<const MCSymbol *, uint64_t> SymSize;

  /// Collection of abstract variables/labels.
  SmallVector<std::unique_ptr<DbgEntity>, 64> ConcreteEntities;

  /// Collection of DebugLocEntry. Stored in a linked list so that DIELocLists
  /// can refer to them in spite of insertions into this list.
  DebugLocStream DebugLocs;

  /// This is a collection of subprogram MDNodes that are processed to
  /// create DIEs.
  SetVector<const DISubprogram *, SmallVector<const DISubprogram *, 16>,
            SmallPtrSet<const DISubprogram *, 16>>
      ProcessedSPNodes;

  /// If nonnull, stores the current machine function we're processing.
  const MachineFunction *CurFn = nullptr;

  /// If nonnull, stores the CU in which the previous subprogram was contained.
  const DwarfCompileUnit *PrevCU = nullptr;

  /// As an optimization, there is no need to emit an entry in the directory
  /// table for the same directory as DW_AT_comp_dir.
  StringRef CompilationDir;

  /// Holder for the file specific debug information.
  DwarfFile InfoHolder;

  /// Holders for the various debug information flags that we might need to
  /// have exposed. See accessor functions below for description.

  /// Map from MDNodes for user-defined types to their type signatures. Also
  /// used to keep track of which types we have emitted type units for.
  DenseMap<const MDNode *, uint64_t> TypeSignatures;

  DenseMap<const MCSection *, const MCSymbol *> SectionLabels;

  SmallVector<
      std::pair<std::unique_ptr<DwarfTypeUnit>, const DICompositeType *>, 1>
      TypeUnitsUnderConstruction;

  /// Whether to use the GNU TLS opcode (instead of the standard opcode).
  bool UseGNUTLSOpcode;

  /// Whether to use DWARF 2 bitfields (instead of the DWARF 4 format).
  bool UseDWARF2Bitfields;

  /// Whether to emit all linkage names, or just abstract subprograms.
  bool UseAllLinkageNames;

  /// Use inlined strings.
  bool UseInlineStrings = false;

  /// Allow emission of .debug_ranges section.
  bool UseRangesSection = true;

  /// True if the sections itself must be used as references and don't create
  /// temp symbols inside DWARF sections.
  bool UseSectionsAsReferences = false;

  ///Allow emission of the .debug_loc section.
  bool UseLocSection = true;

  /// Generate DWARF v4 type units.
  bool GenerateTypeUnits;

  /// Emit a .debug_macro section instead of .debug_macinfo.
  bool UseDebugMacroSection;

  /// Avoid using DW_OP_convert due to consumer incompatibilities.
  bool EnableOpConvert;

public:
  enum class MinimizeAddrInV5 {
    Default,
    Disabled,
    Ranges,
    Expressions,
    Form,
  };

private:
  /// Force the use of DW_AT_ranges even for single-entry range lists.
  MinimizeAddrInV5 MinimizeAddr = MinimizeAddrInV5::Disabled;

  /// DWARF5 Experimental Options
  /// @{
  AccelTableKind TheAccelTableKind;
  bool HasAppleExtensionAttributes;
  bool HasSplitDwarf;

  /// Whether to generate the DWARF v5 string offsets table.
  /// It consists of a series of contributions, each preceded by a header.
  /// The pre-DWARF v5 string offsets table for split dwarf is, in contrast,
  /// a monolithic sequence of string offsets.
  bool UseSegmentedStringOffsetsTable;

  /// Enable production of call site parameters needed to print the debug entry
  /// values. Useful for testing purposes when a debugger does not support the
  /// feature yet.
  bool EmitDebugEntryValues;

  /// Separated Dwarf Variables
  /// In general these will all be for bits that are left in the
  /// original object file, rather than things that are meant
  /// to be in the .dwo sections.

  /// Holder for the skeleton information.
  DwarfFile SkeletonHolder;

  /// Store file names for type units under fission in a line table
  /// header that will be emitted into debug_line.dwo.
  // FIXME: replace this with a map from comp_dir to table so that we
  // can emit multiple tables during LTO each of which uses directory
  // 0, referencing the comp_dir of all the type units that use it.
  MCDwarfDwoLineTable SplitTypeUnitFileTable;
  /// @}

  /// True iff there are multiple CUs in this module.
  bool SingleCU;
  bool IsDarwin;

  /// Map for tracking Fortran deferred CHARACTER lengths.
  DenseMap<const DIStringType *, unsigned> StringTypeLocMap;

  AddressPool AddrPool;

  /// Accelerator tables.
  AccelTable<DWARF5AccelTableData> AccelDebugNames;
  AccelTable<AppleAccelTableOffsetData> AccelNames;
  AccelTable<AppleAccelTableOffsetData> AccelObjC;
  AccelTable<AppleAccelTableOffsetData> AccelNamespace;
  AccelTable<AppleAccelTableTypeData> AccelTypes;

  /// Identify a debugger for "tuning" the debug info.
  ///
  /// The "tuning" should be used to set defaults for individual feature flags
  /// in DwarfDebug; if a given feature has a more specific command-line option,
  /// that option should take precedence over the tuning.
  DebuggerKind DebuggerTuning = DebuggerKind::Default;

  MCDwarfDwoLineTable *getDwoLineTable(const DwarfCompileUnit &);

  const SmallVectorImpl<std::unique_ptr<DwarfCompileUnit>> &getUnits() {
    return InfoHolder.getUnits();
  }

  using InlinedEntity = DbgValueHistoryMap::InlinedEntity;

  void ensureAbstractEntityIsCreated(DwarfCompileUnit &CU,
                                     const DINode *Node,
                                     const MDNode *Scope);
  void ensureAbstractEntityIsCreatedIfScoped(DwarfCompileUnit &CU,
                                             const DINode *Node,
                                             const MDNode *Scope);

  DbgEntity *createConcreteEntity(DwarfCompileUnit &TheCU,
                                  LexicalScope &Scope,
                                  const DINode *Node,
                                  const DILocation *Location,
                                  const MCSymbol *Sym = nullptr);

  /// Construct a DIE for this abstract scope.
  void constructAbstractSubprogramScopeDIE(DwarfCompileUnit &SrcCU, LexicalScope *Scope);

  /// Construct DIEs for call site entries describing the calls in \p MF.
  void constructCallSiteEntryDIEs(const DISubprogram &SP, DwarfCompileUnit &CU,
                                  DIE &ScopeDIE, const MachineFunction &MF);

  template <typename DataT>
  void addAccelNameImpl(const DICompileUnit &CU, AccelTable<DataT> &AppleAccel,
                        StringRef Name, const DIE &Die);

  void finishEntityDefinitions();

  void finishSubprogramDefinitions();

  /// Finish off debug information after all functions have been
  /// processed.
  void finalizeModuleInfo();

  /// Emit the debug info section.
  void emitDebugInfo();

  /// Emit the abbreviation section.
  void emitAbbreviations();

  /// Emit the string offsets table header.
  void emitStringOffsetsTableHeader();

  /// Emit a specified accelerator table.
  template <typename AccelTableT>
  void emitAccel(AccelTableT &Accel, MCSection *Section, StringRef TableName);

  /// Emit DWARF v5 accelerator table.
  void emitAccelDebugNames();

  /// Emit visible names into a hashed accelerator table section.
  void emitAccelNames();

  /// Emit objective C classes and categories into a hashed
  /// accelerator table section.
  void emitAccelObjC();

  /// Emit namespace dies into a hashed accelerator table.
  void emitAccelNamespaces();

  /// Emit type dies into a hashed accelerator table.
  void emitAccelTypes();

  /// Emit visible names and types into debug pubnames and pubtypes sections.
  void emitDebugPubSections();

  void emitDebugPubSection(bool GnuStyle, StringRef Name,
                           DwarfCompileUnit *TheU,
                           const StringMap<const DIE *> &Globals);

  /// Emit null-terminated strings into a debug str section.
  void emitDebugStr();

  /// Emit variable locations into a debug loc section.
  void emitDebugLoc();

  /// Emit variable locations into a debug loc dwo section.
  void emitDebugLocDWO();

  void emitDebugLocImpl(MCSection *Sec);

  /// Emit address ranges into a debug aranges section.
  void emitDebugARanges();

  /// Emit address ranges into a debug ranges section.
  void emitDebugRanges();
  void emitDebugRangesDWO();
  void emitDebugRangesImpl(const DwarfFile &Holder, MCSection *Section);

  /// Emit macros into a debug macinfo section.
  void emitDebugMacinfo();
  /// Emit macros into a debug macinfo.dwo section.
  void emitDebugMacinfoDWO();
  void emitDebugMacinfoImpl(MCSection *Section);
  void emitMacro(DIMacro &M);
  void emitMacroFile(DIMacroFile &F, DwarfCompileUnit &U);
  void emitMacroFileImpl(DIMacroFile &F, DwarfCompileUnit &U,
                         unsigned StartFile, unsigned EndFile,
                         StringRef (*MacroFormToString)(unsigned Form));
  void handleMacroNodes(DIMacroNodeArray Nodes, DwarfCompileUnit &U);

  /// DWARF 5 Experimental Split Dwarf Emitters

  /// Initialize common features of skeleton units.
  void initSkeletonUnit(const DwarfUnit &U, DIE &Die,
                        std::unique_ptr<DwarfCompileUnit> NewU);

  /// Construct the split debug info compile unit for the debug info section.
  /// In DWARF v5, the skeleton unit DIE may have the following attributes:
  /// DW_AT_addr_base, DW_AT_comp_dir, DW_AT_dwo_name, DW_AT_high_pc,
  /// DW_AT_low_pc, DW_AT_ranges, DW_AT_stmt_list, and DW_AT_str_offsets_base.
  /// Prior to DWARF v5 it may also have DW_AT_GNU_dwo_id. DW_AT_GNU_dwo_name
  /// is used instead of DW_AT_dwo_name, Dw_AT_GNU_addr_base instead of
  /// DW_AT_addr_base, and DW_AT_GNU_ranges_base instead of DW_AT_rnglists_base.
  DwarfCompileUnit &constructSkeletonCU(const DwarfCompileUnit &CU);

  /// Emit the debug info dwo section.
  void emitDebugInfoDWO();

  /// Emit the debug abbrev dwo section.
  void emitDebugAbbrevDWO();

  /// Emit the debug line dwo section.
  void emitDebugLineDWO();

  /// Emit the dwo stringoffsets table header.
  void emitStringOffsetsTableHeaderDWO();

  /// Emit the debug str dwo section.
  void emitDebugStrDWO();

  /// Emit DWO addresses.
  void emitDebugAddr();

  /// Flags to let the linker know we have emitted new style pubnames. Only
  /// emit it here if we don't have a skeleton CU for split dwarf.
  void addGnuPubAttributes(DwarfCompileUnit &U, DIE &D) const;

  /// Create new DwarfCompileUnit for the given metadata node with tag
  /// DW_TAG_compile_unit.
  DwarfCompileUnit &getOrCreateDwarfCompileUnit(const DICompileUnit *DIUnit);
  void finishUnitAttributes(const DICompileUnit *DIUnit,
                            DwarfCompileUnit &NewCU);

  /// Construct imported_module or imported_declaration DIE.
  void constructAndAddImportedEntityDIE(DwarfCompileUnit &TheCU,
                                        const DIImportedEntity *N);

  /// Register a source line with debug info. Returns the unique
  /// label that was emitted and which provides correspondence to the
  /// source line list.
  void recordSourceLine(unsigned Line, unsigned Col, const MDNode *Scope,
                        unsigned Flags);

  /// Populate LexicalScope entries with variables' info.
  void collectEntityInfo(DwarfCompileUnit &TheCU, const DISubprogram *SP,
                         DenseSet<InlinedEntity> &ProcessedVars);

  /// Build the location list for all DBG_VALUEs in the
  /// function that describe the same variable. If the resulting
  /// list has only one entry that is valid for entire variable's
  /// scope return true.
  bool buildLocationList(SmallVectorImpl<DebugLocEntry> &DebugLoc,
                         const DbgValueHistoryMap::Entries &Entries);

  /// Collect variable information from the side table maintained by MF.
  void collectVariableInfoFromMFTable(DwarfCompileUnit &TheCU,
                                      DenseSet<InlinedEntity> &P);

  /// Emit the reference to the section.
  void emitSectionReference(const DwarfCompileUnit &CU);

protected:
  /// Gather pre-function debug information.
  void beginFunctionImpl(const MachineFunction *MF) override;

  /// Gather and emit post-function debug information.
  void endFunctionImpl(const MachineFunction *MF) override;

  /// Get Dwarf compile unit ID for line table.
  unsigned getDwarfCompileUnitIDForLineTable(const DwarfCompileUnit &CU);

  void skippedNonDebugFunction() override;

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A);

  ~DwarfDebug() override;

  /// Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule(Module *M) override;

  /// Emit all Dwarf sections that should come after the content.
  void endModule() override;

  /// Emits inital debug location directive.
  DebugLoc emitInitialLocDirective(const MachineFunction &MF, unsigned CUID);

  /// Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI) override;

  /// Perform an MD5 checksum of \p Identifier and return the lower 64 bits.
  static uint64_t makeTypeSignature(StringRef Identifier);

  /// Add a DIE to the set of types that we're going to pull into
  /// type units.
  void addDwarfTypeUnitType(DwarfCompileUnit &CU, StringRef Identifier,
                            DIE &Die, const DICompositeType *CTy);

  class NonTypeUnitContext {
    DwarfDebug *DD;
    decltype(DwarfDebug::TypeUnitsUnderConstruction) TypeUnitsUnderConstruction;
    bool AddrPoolUsed;
    friend class DwarfDebug;
    NonTypeUnitContext(DwarfDebug *DD);
  public:
    NonTypeUnitContext(NonTypeUnitContext&&) = default;
    ~NonTypeUnitContext();
  };

  NonTypeUnitContext enterNonTypeUnitContext();

  /// Add a label so that arange data can be generated for it.
  void addArangeLabel(SymbolCU SCU) { ArangeLabels.push_back(SCU); }

  /// For symbols that have a size designated (e.g. common symbols),
  /// this tracks that size.
  void setSymbolSize(const MCSymbol *Sym, uint64_t Size) override {
    SymSize[Sym] = Size;
  }

  /// Returns whether we should emit all DW_AT_[MIPS_]linkage_name.
  /// If not, we still might emit certain cases.
  bool useAllLinkageNames() const { return UseAllLinkageNames; }

  /// Returns whether to use DW_OP_GNU_push_tls_address, instead of the
  /// standard DW_OP_form_tls_address opcode
  bool useGNUTLSOpcode() const { return UseGNUTLSOpcode; }

  /// Returns whether to use the DWARF2 format for bitfields instyead of the
  /// DWARF4 format.
  bool useDWARF2Bitfields() const { return UseDWARF2Bitfields; }

  /// Returns whether to use inline strings.
  bool useInlineStrings() const { return UseInlineStrings; }

  /// Returns whether ranges section should be emitted.
  bool useRangesSection() const { return UseRangesSection; }

  /// Returns whether range encodings should be used for single entry range
  /// lists.
  bool alwaysUseRanges() const {
    return MinimizeAddr == MinimizeAddrInV5::Ranges;
  }

  // Returns whether novel exprloc addrx+offset encodings should be used to
  // reduce debug_addr size.
  bool useAddrOffsetExpressions() const {
    return MinimizeAddr == MinimizeAddrInV5::Expressions;
  }

  // Returns whether addrx+offset LLVM extension form should be used to reduce
  // debug_addr size.
  bool useAddrOffsetForm() const {
    return MinimizeAddr == MinimizeAddrInV5::Form;
  }

  /// Returns whether to use sections as labels rather than temp symbols.
  bool useSectionsAsReferences() const {
    return UseSectionsAsReferences;
  }

  /// Returns whether .debug_loc section should be emitted.
  bool useLocSection() const { return UseLocSection; }

  /// Returns whether to generate DWARF v4 type units.
  bool generateTypeUnits() const { return GenerateTypeUnits; }

  // Experimental DWARF5 features.

  /// Returns what kind (if any) of accelerator tables to emit.
  AccelTableKind getAccelTableKind() const { return TheAccelTableKind; }

  bool useAppleExtensionAttributes() const {
    return HasAppleExtensionAttributes;
  }

  /// Returns whether or not to change the current debug info for the
  /// split dwarf proposal support.
  bool useSplitDwarf() const { return HasSplitDwarf; }

  /// Returns whether to generate a string offsets table with (possibly shared)
  /// contributions from each CU and type unit. This implies the use of
  /// DW_FORM_strx* indirect references with DWARF v5 and beyond. Note that
  /// DW_FORM_GNU_str_index is also an indirect reference, but it is used with
  /// a pre-DWARF v5 implementation of split DWARF sections, which uses a
  /// monolithic string offsets table.
  bool useSegmentedStringOffsetsTable() const {
    return UseSegmentedStringOffsetsTable;
  }

  bool emitDebugEntryValues() const {
    return EmitDebugEntryValues;
  }

  bool useOpConvert() const {
    return EnableOpConvert;
  }

  bool shareAcrossDWOCUs() const;

  /// Returns the Dwarf Version.
  uint16_t getDwarfVersion() const;

  /// Returns a suitable DWARF form to represent a section offset, i.e.
  /// * DW_FORM_sec_offset for DWARF version >= 4;
  /// * DW_FORM_data8 for 64-bit DWARFv3;
  /// * DW_FORM_data4 for 32-bit DWARFv3 and DWARFv2.
  dwarf::Form getDwarfSectionOffsetForm() const;

  /// Returns the previous CU that was being updated
  const DwarfCompileUnit *getPrevCU() const { return PrevCU; }
  void setPrevCU(const DwarfCompileUnit *PrevCU) { this->PrevCU = PrevCU; }

  /// Terminate the line table by adding the last range label.
  void terminateLineTable(const DwarfCompileUnit *CU);

  /// Returns the entries for the .debug_loc section.
  const DebugLocStream &getDebugLocs() const { return DebugLocs; }

  /// Emit an entry for the debug loc section. This can be used to
  /// handle an entry that's going to be emitted into the debug loc section.
  void emitDebugLocEntry(ByteStreamer &Streamer,
                         const DebugLocStream::Entry &Entry,
                         const DwarfCompileUnit *CU);

  /// Emit the location for a debug loc entry, including the size header.
  void emitDebugLocEntryLocation(const DebugLocStream::Entry &Entry,
                                 const DwarfCompileUnit *CU);

  void addSubprogramNames(const DICompileUnit &CU, const DISubprogram *SP,
                          DIE &Die);

  AddressPool &getAddressPool() { return AddrPool; }

  void addAccelName(const DICompileUnit &CU, StringRef Name, const DIE &Die);

  void addAccelObjC(const DICompileUnit &CU, StringRef Name, const DIE &Die);

  void addAccelNamespace(const DICompileUnit &CU, StringRef Name,
                         const DIE &Die);

  void addAccelType(const DICompileUnit &CU, StringRef Name, const DIE &Die,
                    char Flags);

  const MachineFunction *getCurrentFunction() const { return CurFn; }

  /// A helper function to check whether the DIE for a given Scope is
  /// going to be null.
  bool isLexicalScopeDIENull(LexicalScope *Scope);

  /// Find the matching DwarfCompileUnit for the given CU DIE.
  DwarfCompileUnit *lookupCU(const DIE *Die) { return CUDieMap.lookup(Die); }
  const DwarfCompileUnit *lookupCU(const DIE *Die) const {
    return CUDieMap.lookup(Die);
  }

  unsigned getStringTypeLoc(const DIStringType *ST) const {
    return StringTypeLocMap.lookup(ST);
  }

  void addStringTypeLoc(const DIStringType *ST, unsigned Loc) {
    assert(ST);
    if (Loc)
      StringTypeLocMap[ST] = Loc;
  }

  /// \defgroup DebuggerTuning Predicates to tune DWARF for a given debugger.
  ///
  /// Returns whether we are "tuning" for a given debugger.
  /// @{
  bool tuneForGDB() const { return DebuggerTuning == DebuggerKind::GDB; }
  bool tuneForLLDB() const { return DebuggerTuning == DebuggerKind::LLDB; }
  bool tuneForSCE() const { return DebuggerTuning == DebuggerKind::SCE; }
  bool tuneForDBX() const { return DebuggerTuning == DebuggerKind::DBX; }
  /// @}

  const MCSymbol *getSectionLabel(const MCSection *S);
  void insertSectionLabel(const MCSymbol *S);

  static void emitDebugLocValue(const AsmPrinter &AP, const DIBasicType *BT,
                                const DbgValueLoc &Value,
                                DwarfExpression &DwarfExpr);

  /// If the \p File has an MD5 checksum, return it as an MD5Result
  /// allocated in the MCContext.
  Optional<MD5::MD5Result> getMD5AsBytes(const DIFile *File) const;
};

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DWARFDEBUG_H
