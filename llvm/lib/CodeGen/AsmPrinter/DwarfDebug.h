//===-- llvm/CodeGen/DwarfDebug.h - Dwarf Debug Framework ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFDEBUG_H__
#define CODEGEN_ASMPRINTER_DWARFDEBUG_H__

#include "AsmPrinterHandler.h"
#include "DIE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/DebugInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLoc.h"

namespace llvm {

class DwarfUnit;
class DwarfCompileUnit;
class ConstantInt;
class ConstantFP;
class DbgVariable;
class MachineFrameInfo;
class MachineModuleInfo;
class MachineOperand;
class MCAsmInfo;
class MCObjectFileInfo;
class DIEAbbrev;
class DIE;
class DIEBlock;
class DIEEntry;

//===----------------------------------------------------------------------===//
/// \brief This class is used to record source line correspondence.
class SrcLineInfo {
  unsigned Line;     // Source line number.
  unsigned Column;   // Source column.
  unsigned SourceID; // Source ID number.
  MCSymbol *Label;   // Label in code ID number.
public:
  SrcLineInfo(unsigned L, unsigned C, unsigned S, MCSymbol *label)
      : Line(L), Column(C), SourceID(S), Label(label) {}

  // Accessors
  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  MCSymbol *getLabel() const { return Label; }
};

/// \brief This struct describes location entries emitted in the .debug_loc
/// section.
class DotDebugLocEntry {
  // Begin and end symbols for the address range that this location is valid.
  const MCSymbol *Begin;
  const MCSymbol *End;

  // Type of entry that this represents.
  enum EntryType {
    E_Location,
    E_Integer,
    E_ConstantFP,
    E_ConstantInt
  };
  enum EntryType EntryKind;

  union {
    int64_t Int;
    const ConstantFP *CFP;
    const ConstantInt *CIP;
  } Constants;

  // The location in the machine frame.
  MachineLocation Loc;

  // The variable to which this location entry corresponds.
  const MDNode *Variable;

  // Whether this location has been merged.
  bool Merged;

public:
  DotDebugLocEntry() : Begin(0), End(0), Variable(0), Merged(false) {
    Constants.Int = 0;
  }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, MachineLocation &L,
                   const MDNode *V)
      : Begin(B), End(E), Loc(L), Variable(V), Merged(false) {
    Constants.Int = 0;
    EntryKind = E_Location;
  }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, int64_t i)
      : Begin(B), End(E), Variable(0), Merged(false) {
    Constants.Int = i;
    EntryKind = E_Integer;
  }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, const ConstantFP *FPtr)
      : Begin(B), End(E), Variable(0), Merged(false) {
    Constants.CFP = FPtr;
    EntryKind = E_ConstantFP;
  }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E,
                   const ConstantInt *IPtr)
      : Begin(B), End(E), Variable(0), Merged(false) {
    Constants.CIP = IPtr;
    EntryKind = E_ConstantInt;
  }

  /// \brief Empty entries are also used as a trigger to emit temp label. Such
  /// labels are referenced is used to find debug_loc offset for a given DIE.
  bool isEmpty() { return Begin == 0 && End == 0; }
  bool isMerged() { return Merged; }
  void Merge(DotDebugLocEntry *Next) {
    if (!(Begin && Loc == Next->Loc && End == Next->Begin))
      return;
    Next->Begin = Begin;
    Merged = true;
  }
  bool isLocation() const { return EntryKind == E_Location; }
  bool isInt() const { return EntryKind == E_Integer; }
  bool isConstantFP() const { return EntryKind == E_ConstantFP; }
  bool isConstantInt() const { return EntryKind == E_ConstantInt; }
  int64_t getInt() const { return Constants.Int; }
  const ConstantFP *getConstantFP() const { return Constants.CFP; }
  const ConstantInt *getConstantInt() const { return Constants.CIP; }
  const MDNode *getVariable() const { return Variable; }
  const MCSymbol *getBeginSym() const { return Begin; }
  const MCSymbol *getEndSym() const { return End; }
  MachineLocation getLoc() const { return Loc; }
};

//===----------------------------------------------------------------------===//
/// \brief This class is used to track local variable information.
class DbgVariable {
  DIVariable Var;             // Variable Descriptor.
  DIE *TheDIE;                // Variable DIE.
  unsigned DotDebugLocOffset; // Offset in DotDebugLocEntries.
  DbgVariable *AbsVar;        // Corresponding Abstract variable, if any.
  const MachineInstr *MInsn;  // DBG_VALUE instruction of the variable.
  int FrameIndex;
  DwarfDebug *DD;

public:
  // AbsVar may be NULL.
  DbgVariable(DIVariable V, DbgVariable *AV, DwarfDebug *DD)
      : Var(V), TheDIE(0), DotDebugLocOffset(~0U), AbsVar(AV), MInsn(0),
        FrameIndex(~0), DD(DD) {}

  // Accessors.
  DIVariable getVariable() const { return Var; }
  void setDIE(DIE *D) { TheDIE = D; }
  DIE *getDIE() const { return TheDIE; }
  void setDotDebugLocOffset(unsigned O) { DotDebugLocOffset = O; }
  unsigned getDotDebugLocOffset() const { return DotDebugLocOffset; }
  StringRef getName() const { return Var.getName(); }
  DbgVariable *getAbstractVariable() const { return AbsVar; }
  const MachineInstr *getMInsn() const { return MInsn; }
  void setMInsn(const MachineInstr *M) { MInsn = M; }
  int getFrameIndex() const { return FrameIndex; }
  void setFrameIndex(int FI) { FrameIndex = FI; }
  // Translate tag to proper Dwarf tag.
  uint16_t getTag() const {
    if (Var.getTag() == dwarf::DW_TAG_arg_variable)
      return dwarf::DW_TAG_formal_parameter;

    return dwarf::DW_TAG_variable;
  }
  /// \brief Return true if DbgVariable is artificial.
  bool isArtificial() const {
    if (Var.isArtificial())
      return true;
    if (getType().isArtificial())
      return true;
    return false;
  }

  bool isObjectPointer() const {
    if (Var.isObjectPointer())
      return true;
    if (getType().isObjectPointer())
      return true;
    return false;
  }

  bool variableHasComplexAddress() const {
    assert(Var.isVariable() && "Invalid complex DbgVariable!");
    return Var.hasComplexAddress();
  }
  bool isBlockByrefVariable() const {
    assert(Var.isVariable() && "Invalid complex DbgVariable!");
    return Var.isBlockByrefVariable();
  }
  unsigned getNumAddrElements() const {
    assert(Var.isVariable() && "Invalid complex DbgVariable!");
    return Var.getNumAddrElements();
  }
  uint64_t getAddrElement(unsigned i) const { return Var.getAddrElement(i); }
  DIType getType() const;

private:
  /// resolve - Look in the DwarfDebug map for the MDNode that
  /// corresponds to the reference.
  template <typename T> T resolve(DIRef<T> Ref) const;
};

/// \brief Collects and handles information specific to a particular
/// collection of units. This collection represents all of the units
/// that will be ultimately output into a single object file.
class DwarfFile {
  // Target of Dwarf emission, used for sizing of abbreviations.
  AsmPrinter *Asm;

  // Used to uniquely define abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  // A list of all the unique abbreviations in use.
  std::vector<DIEAbbrev *> Abbreviations;

  // A pointer to all units in the section.
  SmallVector<DwarfUnit *, 1> CUs;

  // Collection of strings for this unit and assorted symbols.
  // A String->Symbol mapping of strings used by indirect
  // references.
  typedef StringMap<std::pair<MCSymbol *, unsigned>, BumpPtrAllocator &>
  StrPool;
  StrPool StringPool;
  unsigned NextStringPoolNumber;
  std::string StringPref;

  // Collection of addresses for this unit and assorted labels.
  // A Symbol->unsigned mapping of addresses used by indirect
  // references.
  typedef DenseMap<const MCExpr *, unsigned> AddrPool;
  AddrPool AddressPool;
  unsigned NextAddrPoolNumber;

public:
  DwarfFile(AsmPrinter *AP, const char *Pref, BumpPtrAllocator &DA)
      : Asm(AP), StringPool(DA), NextStringPoolNumber(0), StringPref(Pref),
        AddressPool(), NextAddrPoolNumber(0) {}

  ~DwarfFile();

  const SmallVectorImpl<DwarfUnit *> &getUnits() { return CUs; }

  /// \brief Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset);

  /// \brief Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// \brief Define a unique number for the abbreviation.
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  /// \brief Add a unit to the list of CUs.
  void addUnit(DwarfUnit *CU) { CUs.push_back(CU); }

  /// \brief Emit all of the units to the section listed with the given
  /// abbreviation section.
  void emitUnits(DwarfDebug *DD, const MCSection *ASection,
                 const MCSymbol *ASectionSym);

  /// \brief Emit a set of abbreviations to the specific section.
  void emitAbbrevs(const MCSection *);

  /// \brief Emit all of the strings to the section given.
  void emitStrings(const MCSection *StrSection, const MCSection *OffsetSection,
                   const MCSymbol *StrSecSym);

  /// \brief Emit all of the addresses to the section given.
  void emitAddresses(const MCSection *AddrSection);

  /// \brief Returns the entry into the start of the pool.
  MCSymbol *getStringPoolSym();

  /// \brief Returns an entry into the string pool with the given
  /// string text.
  MCSymbol *getStringPoolEntry(StringRef Str);

  /// \brief Returns the index into the string pool with the given
  /// string text.
  unsigned getStringPoolIndex(StringRef Str);

  /// \brief Returns the string pool.
  StrPool *getStringPool() { return &StringPool; }

  /// \brief Returns the index into the address pool with the given
  /// label/symbol.
  unsigned getAddrPoolIndex(const MCExpr *Sym);
  unsigned getAddrPoolIndex(const MCSymbol *Sym);

  /// \brief Returns the address pool.
  AddrPool *getAddrPool() { return &AddressPool; }
};

/// \brief Helper used to pair up a symbol and its DWARF compile unit.
struct SymbolCU {
  SymbolCU(DwarfCompileUnit *CU, const MCSymbol *Sym) : Sym(Sym), CU(CU) {}
  const MCSymbol *Sym;
  DwarfCompileUnit *CU;
};

/// \brief Collects and handles dwarf debug information.
class DwarfDebug : public AsmPrinterHandler {
  // Target of Dwarf emission.
  AsmPrinter *Asm;

  // Collected machine module information.
  MachineModuleInfo *MMI;

  // All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  // Handle to the compile unit used for the inline extension handling,
  // this is just so that the DIEValue allocator has a place to store
  // the particular elements.
  // FIXME: Store these off of DwarfDebug instead?
  DwarfCompileUnit *FirstCU;

  // Maps MDNode with its corresponding DwarfCompileUnit.
  DenseMap<const MDNode *, DwarfCompileUnit *> CUMap;

  // Maps subprogram MDNode with its corresponding DwarfCompileUnit.
  DenseMap<const MDNode *, DwarfCompileUnit *> SPMap;

  // Maps a CU DIE with its corresponding DwarfCompileUnit.
  DenseMap<const DIE *, DwarfCompileUnit *> CUDieMap;

  /// Maps MDNodes for type sysstem with the corresponding DIEs. These DIEs can
  /// be shared across CUs, that is why we keep the map here instead
  /// of in DwarfCompileUnit.
  DenseMap<const MDNode *, DIE *> MDTypeNodeToDieMap;

  // Stores the current file ID for a given compile unit.
  DenseMap<unsigned, unsigned> FileIDCUMap;
  // Source id map, i.e. CUID, source filename and directory,
  // separated by a zero byte, mapped to a unique id.
  StringMap<unsigned, BumpPtrAllocator &> SourceIdMap;

  // List of all labels used in aranges generation.
  std::vector<SymbolCU> ArangeLabels;

  // Size of each symbol emitted (for those symbols that have a specific size).
  DenseMap<const MCSymbol *, uint64_t> SymSize;

  // Provides a unique id per text section.
  typedef DenseMap<const MCSection *, SmallVector<SymbolCU, 8> > SectionMapType;
  SectionMapType SectionMap;

  // List of arguments for current function.
  SmallVector<DbgVariable *, 8> CurrentFnArguments;

  LexicalScopes LScopes;

  // Collection of abstract subprogram DIEs.
  DenseMap<const MDNode *, DIE *> AbstractSPDies;

  // Collection of dbg variables of a scope.
  typedef DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8> >
  ScopeVariablesMap;
  ScopeVariablesMap ScopeVariables;

  // Collection of abstract variables.
  DenseMap<const MDNode *, DbgVariable *> AbstractVariables;

  // Collection of DotDebugLocEntry.
  SmallVector<DotDebugLocEntry, 4> DotDebugLocEntries;

  // Collection of subprogram DIEs that are marked (at the end of the module)
  // as DW_AT_inline.
  SmallPtrSet<DIE *, 4> InlinedSubprogramDIEs;

  // This is a collection of subprogram MDNodes that are processed to
  // create DIEs.
  SmallPtrSet<const MDNode *, 16> ProcessedSPNodes;

  // Maps instruction with label emitted before instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsBeforeInsn;

  // Maps instruction with label emitted after instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsAfterInsn;

  // Every user variable mentioned by a DBG_VALUE instruction in order of
  // appearance.
  SmallVector<const MDNode *, 8> UserVariables;

  // For each user variable, keep a list of DBG_VALUE instructions in order.
  // The list can also contain normal instructions that clobber the previous
  // DBG_VALUE.
  typedef DenseMap<const MDNode *, SmallVector<const MachineInstr *, 4> >
  DbgValueHistoryMap;
  DbgValueHistoryMap DbgValues;

  // Previous instruction's location information. This is used to determine
  // label location to indicate scope boundries in dwarf debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel;

  // This location indicates end of function prologue and beginning of function
  // body.
  DebugLoc PrologEndLoc;

  // If nonnull, stores the current machine function we're processing.
  const MachineFunction *CurFn;

  // If nonnull, stores the current machine instruction we're processing.
  const MachineInstr *CurMI;

  // Section Symbols: these are assembler temporary labels that are emitted at
  // the beginning of each supported dwarf section.  These are used to form
  // section offsets and are created by EmitSectionLabels.
  MCSymbol *DwarfInfoSectionSym, *DwarfAbbrevSectionSym;
  MCSymbol *DwarfStrSectionSym, *TextSectionSym, *DwarfDebugRangeSectionSym;
  MCSymbol *DwarfDebugLocSectionSym, *DwarfLineSectionSym, *DwarfAddrSectionSym;
  MCSymbol *FunctionBeginSym, *FunctionEndSym;
  MCSymbol *DwarfAbbrevDWOSectionSym, *DwarfStrDWOSectionSym;
  MCSymbol *DwarfGnuPubNamesSectionSym, *DwarfGnuPubTypesSectionSym;

  // As an optimization, there is no need to emit an entry in the directory
  // table for the same directory as DW_AT_comp_dir.
  StringRef CompilationDir;

  // Counter for assigning globally unique IDs for ranges.
  unsigned GlobalRangeCount;

  // Holder for the file specific debug information.
  DwarfFile InfoHolder;

  // Holders for the various debug information flags that we might need to
  // have exposed. See accessor functions below for description.

  // Holder for imported entities.
  typedef SmallVector<std::pair<const MDNode *, const MDNode *>, 32>
  ImportedEntityMap;
  ImportedEntityMap ScopesWithImportedEntities;

  // Map from MDNodes for user-defined types to the type units that describe
  // them.
  DenseMap<const MDNode *, const DwarfTypeUnit *> DwarfTypeUnits;

  // Whether to emit the pubnames/pubtypes sections.
  bool HasDwarfPubSections;

  // Version of dwarf we're emitting.
  unsigned DwarfVersion;

  // Maps from a type identifier to the actual MDNode.
  DITypeIdentifierMap TypeIdentifierMap;

  // DWARF5 Experimental Options
  bool HasDwarfAccelTables;
  bool HasSplitDwarf;

  // Separated Dwarf Variables
  // In general these will all be for bits that are left in the
  // original object file, rather than things that are meant
  // to be in the .dwo sections.

  // Holder for the skeleton information.
  DwarfFile SkeletonHolder;

  void addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  const SmallVectorImpl<DwarfUnit *> &getUnits() {
    return InfoHolder.getUnits();
  }

  /// \brief Find abstract variable associated with Var.
  DbgVariable *findAbstractVariable(DIVariable &Var, DebugLoc Loc);

  /// \brief Find DIE for the given subprogram and attach appropriate
  /// DW_AT_low_pc and DW_AT_high_pc attributes. If there are global
  /// variables in this scope then create and insert DIEs for these
  /// variables.
  DIE *updateSubprogramScopeDIE(DwarfCompileUnit *SPCU, DISubprogram SP);

  /// \brief A helper function to check whether the DIE for a given Scope is
  /// going to be null.
  bool isLexicalScopeDIENull(LexicalScope *Scope);

  /// \brief A helper function to construct a RangeSpanList for a given
  /// lexical scope.
  void addScopeRangeList(DwarfCompileUnit *TheCU, DIE *ScopeDIE,
                         const SmallVectorImpl<InsnRange> &Range);

  /// \brief Construct new DW_TAG_lexical_block for this scope and
  /// attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(DwarfCompileUnit *TheCU, LexicalScope *Scope);

  /// \brief This scope represents inlined body of a function. Construct
  /// DIE to represent this concrete inlined copy of the function.
  DIE *constructInlinedScopeDIE(DwarfCompileUnit *TheCU, LexicalScope *Scope);

  /// \brief Construct a DIE for this scope.
  DIE *constructScopeDIE(DwarfCompileUnit *TheCU, LexicalScope *Scope);
  /// A helper function to create children of a Scope DIE.
  DIE *createScopeChildrenDIE(DwarfCompileUnit *TheCU, LexicalScope *Scope,
                              SmallVectorImpl<DIE *> &Children);

  /// \brief Emit initial Dwarf sections with a label at the start of each one.
  void emitSectionLabels();

  /// \brief Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset);

  /// \brief Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// \brief Attach DW_AT_inline attribute with inlined subprogram DIEs.
  void computeInlinedDIEs();

  /// \brief Collect info for variables that were optimized out.
  void collectDeadVariables();

  /// \brief Finish off debug information after all functions have been
  /// processed.
  void finalizeModuleInfo();

  /// \brief Emit labels to close any remaining sections that have been left
  /// open.
  void endSections();

  /// \brief Emit the debug info section.
  void emitDebugInfo();

  /// \brief Emit the abbreviation section.
  void emitAbbreviations();

  /// \brief Emit the last address of the section and the end of
  /// the line matrix.
  void emitEndOfLineMatrix(unsigned SectionEnd);

  /// \brief Emit visible names into a hashed accelerator table section.
  void emitAccelNames();

  /// \brief Emit objective C classes and categories into a hashed
  /// accelerator table section.
  void emitAccelObjC();

  /// \brief Emit namespace dies into a hashed accelerator table.
  void emitAccelNamespaces();

  /// \brief Emit type dies into a hashed accelerator table.
  void emitAccelTypes();

  /// \brief Emit visible names into a debug pubnames section.
  /// \param GnuStyle determines whether or not we want to emit
  /// additional information into the table ala newer gcc for gdb
  /// index.
  void emitDebugPubNames(bool GnuStyle = false);

  /// \brief Emit visible types into a debug pubtypes section.
  /// \param GnuStyle determines whether or not we want to emit
  /// additional information into the table ala newer gcc for gdb
  /// index.
  void emitDebugPubTypes(bool GnuStyle = false);

  /// \brief Emit visible names into a debug str section.
  void emitDebugStr();

  /// \brief Emit visible names into a debug loc section.
  void emitDebugLoc();

  /// \brief Emit visible names into a debug aranges section.
  void emitDebugARanges();

  /// \brief Emit visible names into a debug ranges section.
  void emitDebugRanges();

  /// \brief Emit inline info using custom format.
  void emitDebugInlineInfo();

  /// DWARF 5 Experimental Split Dwarf Emitters

  /// \brief Construct the split debug info compile unit for the debug info
  /// section.
  DwarfCompileUnit *constructSkeletonCU(const DwarfCompileUnit *CU);

  /// \brief Emit the debug info dwo section.
  void emitDebugInfoDWO();

  /// \brief Emit the debug abbrev dwo section.
  void emitDebugAbbrevDWO();

  /// \brief Emit the debug str dwo section.
  void emitDebugStrDWO();

  /// Flags to let the linker know we have emitted new style pubnames. Only
  /// emit it here if we don't have a skeleton CU for split dwarf.
  void addGnuPubAttributes(DwarfUnit *U, DIE *D) const;

  /// \brief Create new DwarfCompileUnit for the given metadata node with tag
  /// DW_TAG_compile_unit.
  DwarfCompileUnit *constructDwarfCompileUnit(DICompileUnit DIUnit);

  /// \brief Construct subprogram DIE.
  void constructSubprogramDIE(DwarfCompileUnit *TheCU, const MDNode *N);

  /// \brief Construct imported_module or imported_declaration DIE.
  void constructImportedEntityDIE(DwarfCompileUnit *TheCU, const MDNode *N);

  /// \brief Construct import_module DIE.
  void constructImportedEntityDIE(DwarfCompileUnit *TheCU, const MDNode *N,
                                  DIE *Context);

  /// \brief Construct import_module DIE.
  void constructImportedEntityDIE(DwarfCompileUnit *TheCU,
                                  const DIImportedEntity &Module, DIE *Context);

  /// \brief Register a source line with debug info. Returns the unique
  /// label that was emitted and which provides correspondence to the
  /// source line list.
  void recordSourceLine(unsigned Line, unsigned Col, const MDNode *Scope,
                        unsigned Flags);

  /// \brief Indentify instructions that are marking the beginning of or
  /// ending of a scope.
  void identifyScopeMarkers();

  /// \brief If Var is an current function argument that add it in
  /// CurrentFnArguments list.
  bool addCurrentFnArgument(DbgVariable *Var, LexicalScope *Scope);

  /// \brief Populate LexicalScope entries with variables' info.
  void collectVariableInfo(SmallPtrSet<const MDNode *, 16> &ProcessedVars);

  /// \brief Collect variable information from the side table maintained
  /// by MMI.
  void collectVariableInfoFromMMITable(SmallPtrSet<const MDNode *, 16> &P);

  /// \brief Ensure that a label will be emitted before MI.
  void requestLabelBeforeInsn(const MachineInstr *MI) {
    LabelsBeforeInsn.insert(std::make_pair(MI, (MCSymbol *)0));
  }

  /// \brief Return Label preceding the instruction.
  MCSymbol *getLabelBeforeInsn(const MachineInstr *MI);

  /// \brief Ensure that a label will be emitted after MI.
  void requestLabelAfterInsn(const MachineInstr *MI) {
    LabelsAfterInsn.insert(std::make_pair(MI, (MCSymbol *)0));
  }

  /// \brief Return Label immediately following the instruction.
  MCSymbol *getLabelAfterInsn(const MachineInstr *MI);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A, Module *M);

  void insertDIE(const MDNode *TypeMD, DIE *Die) {
    MDTypeNodeToDieMap.insert(std::make_pair(TypeMD, Die));
  }
  DIE *getDIE(const MDNode *TypeMD) {
    return MDTypeNodeToDieMap.lookup(TypeMD);
  }

  /// \brief Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule();

  /// \brief Emit all Dwarf sections that should come after the content.
  void endModule();

  /// \brief Gather pre-function debug information.
  void beginFunction(const MachineFunction *MF);

  /// \brief Gather and emit post-function debug information.
  void endFunction(const MachineFunction *MF);

  /// \brief Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI);

  /// \brief Process end of an instruction.
  void endInstruction();

  /// \brief Add a DIE to the set of types that we're going to pull into
  /// type units.
  void addDwarfTypeUnitType(uint16_t Language, DIE *Die, DICompositeType CTy);

  /// \brief Add a label so that arange data can be generated for it.
  void addArangeLabel(SymbolCU SCU) { ArangeLabels.push_back(SCU); }

  /// \brief For symbols that have a size designated (e.g. common symbols),
  /// this tracks that size.
  void setSymbolSize(const MCSymbol *Sym, uint64_t Size) {
    SymSize[Sym] = Size;
  }

  /// \brief Look up the source id with the given directory and source file
  /// names. If none currently exists, create a new id and insert it in the
  /// SourceIds map.
  unsigned getOrCreateSourceID(StringRef DirName, StringRef FullName,
                               unsigned CUID);

  /// \brief Recursively Emits a debug information entry.
  void emitDIE(DIE *Die);

  // Experimental DWARF5 features.

  /// \brief Returns whether or not to emit tables that dwarf consumers can
  /// use to accelerate lookup.
  bool useDwarfAccelTables() { return HasDwarfAccelTables; }

  /// \brief Returns whether or not to change the current debug info for the
  /// split dwarf proposal support.
  bool useSplitDwarf() { return HasSplitDwarf; }

  /// Returns the Dwarf Version.
  unsigned getDwarfVersion() const { return DwarfVersion; }

  /// Find the MDNode for the given reference.
  template <typename T> T resolve(DIRef<T> Ref) const {
    return Ref.resolve(TypeIdentifierMap);
  }

  /// isSubprogramContext - Return true if Context is either a subprogram
  /// or another context nested inside a subprogram.
  bool isSubprogramContext(const MDNode *Context);
};
} // End of namespace llvm

#endif
