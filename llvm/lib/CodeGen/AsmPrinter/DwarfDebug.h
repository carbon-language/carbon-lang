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

class CompileUnit;
class ConstantInt;
class ConstantFP;
class DbgVariable;
class MachineFrameInfo;
class MachineModuleInfo;
class MachineOperand;
class MCAsmInfo;
class DIEAbbrev;
class DIE;
class DIEBlock;
class DIEEntry;

//===----------------------------------------------------------------------===//
/// \brief This class is used to record source line correspondence.
class SrcLineInfo {
  unsigned Line;                     // Source line number.
  unsigned Column;                   // Source column.
  unsigned SourceID;                 // Source ID number.
  MCSymbol *Label;                   // Label in code ID number.
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
typedef struct DotDebugLocEntry {
  const MCSymbol *Begin;
  const MCSymbol *End;
  MachineLocation Loc;
  const MDNode *Variable;
  bool Merged;
  bool Constant;
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
  DotDebugLocEntry()
    : Begin(0), End(0), Variable(0), Merged(false),
      Constant(false) { Constants.Int = 0;}
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, MachineLocation &L,
                   const MDNode *V)
    : Begin(B), End(E), Loc(L), Variable(V), Merged(false),
      Constant(false) { Constants.Int = 0; EntryKind = E_Location; }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, int64_t i)
    : Begin(B), End(E), Variable(0), Merged(false),
      Constant(true) { Constants.Int = i; EntryKind = E_Integer; }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, const ConstantFP *FPtr)
    : Begin(B), End(E), Variable(0), Merged(false),
      Constant(true) { Constants.CFP = FPtr; EntryKind = E_ConstantFP; }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E,
                   const ConstantInt *IPtr)
    : Begin(B), End(E), Variable(0), Merged(false),
      Constant(true) { Constants.CIP = IPtr; EntryKind = E_ConstantInt; }

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
  bool isLocation() const    { return EntryKind == E_Location; }
  bool isInt() const         { return EntryKind == E_Integer; }
  bool isConstantFP() const  { return EntryKind == E_ConstantFP; }
  bool isConstantInt() const { return EntryKind == E_ConstantInt; }
  int64_t getInt()                    { return Constants.Int; }
  const ConstantFP *getConstantFP()   { return Constants.CFP; }
  const ConstantInt *getConstantInt() { return Constants.CIP; }
} DotDebugLocEntry;

//===----------------------------------------------------------------------===//
/// \brief This class is used to track local variable information.
class DbgVariable {
  DIVariable Var;                    // Variable Descriptor.
  DIE *TheDIE;                       // Variable DIE.
  unsigned DotDebugLocOffset;        // Offset in DotDebugLocEntries.
  DbgVariable *AbsVar;               // Corresponding Abstract variable, if any.
  const MachineInstr *MInsn;         // DBG_VALUE instruction of the variable.
  int FrameIndex;
public:
  // AbsVar may be NULL.
  DbgVariable(DIVariable V, DbgVariable *AV)
    : Var(V), TheDIE(0), DotDebugLocOffset(~0U), AbsVar(AV), MInsn(0),
      FrameIndex(~0) {}

  // Accessors.
  DIVariable getVariable()           const { return Var; }
  void setDIE(DIE *D)                      { TheDIE = D; }
  DIE *getDIE()                      const { return TheDIE; }
  void setDotDebugLocOffset(unsigned O)    { DotDebugLocOffset = O; }
  unsigned getDotDebugLocOffset()    const { return DotDebugLocOffset; }
  StringRef getName()                const { return Var.getName(); }
  DbgVariable *getAbstractVariable() const { return AbsVar; }
  const MachineInstr *getMInsn()     const { return MInsn; }
  void setMInsn(const MachineInstr *M)     { MInsn = M; }
  int getFrameIndex()                const { return FrameIndex; }
  void setFrameIndex(int FI)               { FrameIndex = FI; }
  // Translate tag to proper Dwarf tag.
  unsigned getTag()                  const {
    if (Var.getTag() == dwarf::DW_TAG_arg_variable)
      return dwarf::DW_TAG_formal_parameter;

    return dwarf::DW_TAG_variable;
  }
  /// \brief Return true if DbgVariable is artificial.
  bool isArtificial()                const {
    if (Var.isArtificial())
      return true;
    if (getType().isArtificial())
      return true;
    return false;
  }

  bool isObjectPointer()             const {
    if (Var.isObjectPointer())
      return true;
    if (getType().isObjectPointer())
      return true;
    return false;
  }

  bool variableHasComplexAddress()   const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.hasComplexAddress();
  }
  bool isBlockByrefVariable()        const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.isBlockByrefVariable();
  }
  unsigned getNumAddrElements()      const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.getNumAddrElements();
  }
  uint64_t getAddrElement(unsigned i) const {
    return Var.getAddrElement(i);
  }
  DIType getType() const;
};

/// \brief Collects and handles information specific to a particular
/// collection of units.
class DwarfUnits {
  // Target of Dwarf emission, used for sizing of abbreviations.
  AsmPrinter *Asm;

  // Used to uniquely define abbreviations.
  FoldingSet<DIEAbbrev> *AbbreviationsSet;

  // A list of all the unique abbreviations in use.
  std::vector<DIEAbbrev *> *Abbreviations;

  // A pointer to all units in the section.
  SmallVector<CompileUnit *, 1> CUs;

public:
  DwarfUnits(AsmPrinter *AP, FoldingSet<DIEAbbrev> *AS,
             std::vector<DIEAbbrev *> *A) :
    Asm(AP), AbbreviationsSet(AS), Abbreviations(A) {}

  /// \brief Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset);

  /// \brief Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// \brief Define a unique number for the abbreviation.
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  /// \brief Add a unit to the list of CUs.
  void addUnit(CompileUnit *CU) { CUs.push_back(CU); }
};

/// \brief Collects and handles dwarf debug information.
class DwarfDebug {
  // Target of Dwarf emission.
  AsmPrinter *Asm;

  // Collected machine module information.
  MachineModuleInfo *MMI;

  // All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //

  CompileUnit *FirstCU;

  // Maps MDNode with its corresponding CompileUnit.
  DenseMap <const MDNode *, CompileUnit *> CUMap;

  // Maps subprogram MDNode with its corresponding CompileUnit.
  DenseMap <const MDNode *, CompileUnit *> SPMap;

  // Used to uniquely define abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  // A list of all the unique abbreviations in use.
  std::vector<DIEAbbrev *> Abbreviations;

  // Source id map, i.e. pair of source filename and directory,
  // separated by a zero byte, mapped to a unique id.
  StringMap<unsigned, BumpPtrAllocator&> SourceIdMap;

  // A String->Symbol mapping of strings used by indirect
  // references.
  StringMap<std::pair<MCSymbol*, unsigned>, BumpPtrAllocator&> StringPool;
  unsigned NextStringPoolNumber;

  // Provides a unique id per text section.
  SetVector<const MCSection*> SectionMap;

  // List of Arguments (DbgValues) for current function.
  SmallVector<DbgVariable *, 8> CurrentFnArguments;

  LexicalScopes LScopes;

  // Collection of abstract subprogram DIEs.
  DenseMap<const MDNode *, DIE *> AbstractSPDies;

  // Collection of dbg variables of a scope.
  DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8> > ScopeVariables;

  // Collection of abstract variables.
  DenseMap<const MDNode *, DbgVariable *> AbstractVariables;

  // Collection of DotDebugLocEntry.
  SmallVector<DotDebugLocEntry, 4> DotDebugLocEntries;

  // Collection of subprogram DIEs that are marked (at the end of the module)
  // as DW_AT_inline.
  SmallPtrSet<DIE *, 4> InlinedSubprogramDIEs;

  // Keep track of inlined functions and their location.  This
  // information is used to populate the debug_inlined section.
  typedef std::pair<const MCSymbol *, DIE *> InlineInfoLabels;
  DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> > InlineInfo;
  SmallVector<const MDNode *, 4> InlinedSPNodes;

  // This is a collection of subprogram MDNodes that are processed to create DIEs.
  SmallPtrSet<const MDNode *, 16> ProcessedSPNodes;

  // Maps instruction with label emitted before instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsBeforeInsn;

  // Maps instruction with label emitted after instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsAfterInsn;

  // Every user variable mentioned by a DBG_VALUE instruction in order of
  // appearance.
  SmallVector<const MDNode*, 8> UserVariables;

  // For each user variable, keep a list of DBG_VALUE instructions in order.
  // The list can also contain normal instructions that clobber the previous
  // DBG_VALUE.
  typedef DenseMap<const MDNode*, SmallVector<const MachineInstr*, 4> >
    DbgValueHistoryMap;
  DbgValueHistoryMap DbgValues;

  SmallVector<const MCSymbol *, 8> DebugRangeSymbols;

  // Previous instruction's location information. This is used to determine
  // label location to indicate scope boundries in dwarf debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel;

  // This location indicates end of function prologue and beginning of function
  // body.
  DebugLoc PrologEndLoc;

  struct FunctionDebugFrameInfo {
    unsigned Number;
    std::vector<MachineMove> Moves;

    FunctionDebugFrameInfo(unsigned Num, const std::vector<MachineMove> &M)
      : Number(Num), Moves(M) {}
  };

  std::vector<FunctionDebugFrameInfo> DebugFrames;

  // Section Symbols: these are assembler temporary labels that are emitted at
  // the beginning of each supported dwarf section.  These are used to form
  // section offsets and are created by EmitSectionLabels.
  MCSymbol *DwarfInfoSectionSym, *DwarfAbbrevSectionSym;
  MCSymbol *DwarfStrSectionSym, *TextSectionSym, *DwarfDebugRangeSectionSym;
  MCSymbol *DwarfDebugLocSectionSym;
  MCSymbol *FunctionBeginSym, *FunctionEndSym;

  // As an optimization, there is no need to emit an entry in the directory
  // table for the same directory as DW_at_comp_dir.
  StringRef CompilationDir;

  // Counter for assigning globally unique IDs for CUs.
  unsigned GlobalCUIndexCount;

  // Holder for the file specific debug information.
  DwarfUnits InfoHolder;

  // Holders for the various debug information flags that we might need to
  // have exposed. See accessor functions below for description.

  // Whether or not we're emitting info for older versions of gdb on darwin.
  bool IsDarwinGDBCompat;

  // DWARF5 Experimental Options
  bool HasDwarfAccelTables;
  bool HasSplitDwarf;

  // Separated Dwarf Variables
  // In general these will all be for bits that are left in the
  // original object file, rather than things that are meant
  // to be in the .dwo sections.

  // The CU left in the original object file for separated debug info.
  CompileUnit *SkeletonCU;
  DwarfUnits SkeletonHolder;

private:

  void addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  /// \brief Find abstract variable associated with Var.
  DbgVariable *findAbstractVariable(DIVariable &Var, DebugLoc Loc);

  /// \brief Find DIE for the given subprogram and attach appropriate
  /// DW_AT_low_pc and DW_AT_high_pc attributes. If there are global
  /// variables in this scope then create and insert DIEs for these
  /// variables.
  DIE *updateSubprogramScopeDIE(CompileUnit *SPCU, const MDNode *SPNode);

  /// \brief Construct new DW_TAG_lexical_block for this scope and
  /// attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(CompileUnit *TheCU, LexicalScope *Scope);

  /// \brief This scope represents inlined body of a function. Construct
  /// DIE to represent this concrete inlined copy of the function.
  DIE *constructInlinedScopeDIE(CompileUnit *TheCU, LexicalScope *Scope);

  /// \brief Construct a DIE for this scope.
  DIE *constructScopeDIE(CompileUnit *TheCU, LexicalScope *Scope);

  /// \brief Emit initial Dwarf sections with a label at the start of each one.
  void emitSectionLabels();

  /// \brief Recursively Emits a debug information entry.
  void emitDIE(DIE *Die);

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

  /// \brief Emit all of the compile units to the target section.
  void emitCompileUnits(const MCSection *);

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

  /// \brief Emit visible types into a debug pubtypes section.
  void emitDebugPubTypes();

  /// \brief Emit visible names into a debug str section.
  void emitDebugStr();

  /// \brief Emit visible names into a debug loc section.
  void emitDebugLoc();

  /// \brief Emit visible names into a debug aranges section.
  void emitDebugARanges();

  /// \brief Emit visible names into a debug ranges section.
  void emitDebugRanges();

  /// \brief Emit visible names into a debug macinfo section.
  void emitDebugMacInfo();

  /// \brief Emit inline info using custom format.
  void emitDebugInlineInfo();

  /// DWARF 5 Experimental Split Dwarf Emitters

  /// \brief Construct the split debug info compile unit for the debug info
  /// section.
  CompileUnit *constructSkeletonCU(const MDNode *);

  /// \brief Emit the local split debug info section.
  void emitSkeletonCU(const MCSection *);

  /// \brief Emit the debug info dwo section.
  void emitDebugInfoDWO();

  /// \brief Create new CompileUnit for the given metadata node with tag
  /// DW_TAG_compile_unit.
  CompileUnit *constructCompileUnit(const MDNode *N);

  /// \brief Construct subprogram DIE.
  void constructSubprogramDIE(CompileUnit *TheCU, const MDNode *N);

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
  bool addCurrentFnArgument(const MachineFunction *MF,
                            DbgVariable *Var, LexicalScope *Scope);

  /// \brief Populate LexicalScope entries with variables' info.
  void collectVariableInfo(const MachineFunction *,
                           SmallPtrSet<const MDNode *, 16> &ProcessedVars);

  /// \brief Collect variable information from the side table maintained
  /// by MMI.
  void collectVariableInfoFromMMITable(const MachineFunction * MF,
                                       SmallPtrSet<const MDNode *, 16> &P);

  /// \brief Ensure that a label will be emitted before MI.
  void requestLabelBeforeInsn(const MachineInstr *MI) {
    LabelsBeforeInsn.insert(std::make_pair(MI, (MCSymbol*)0));
  }

  /// \brief Return Label preceding the instruction.
  const MCSymbol *getLabelBeforeInsn(const MachineInstr *MI);

  /// \brief Ensure that a label will be emitted after MI.
  void requestLabelAfterInsn(const MachineInstr *MI) {
    LabelsAfterInsn.insert(std::make_pair(MI, (MCSymbol*)0));
  }

  /// \brief Return Label immediately following the instruction.
  const MCSymbol *getLabelAfterInsn(const MachineInstr *MI);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A, Module *M);
  ~DwarfDebug();

  /// \brief Collect debug info from named mdnodes such as llvm.dbg.enum
  /// and llvm.dbg.ty
  void collectInfoFromNamedMDNodes(const Module *M);

  /// \brief Collect debug info using DebugInfoFinder.
  /// FIXME - Remove this when DragonEgg switches to DIBuilder.
  bool collectLegacyDebugInfo(const Module *M);

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
  void endInstruction(const MachineInstr *MI);

  /// \brief Look up the source id with the given directory and source file
  /// names. If none currently exists, create a new id and insert it in the
  /// SourceIds map.
  unsigned getOrCreateSourceID(StringRef DirName, StringRef FullName);

  /// \brief Returns the entry into the start of the pool.
  MCSymbol *getStringPool();

  /// \brief Returns an entry into the string pool with the given
  /// string text.
  MCSymbol *getStringPoolEntry(StringRef Str);

  /// \brief Returns whether or not to limit some of our debug
  /// output to the limitations of darwin gdb.
  bool useDarwinGDBCompat() { return IsDarwinGDBCompat; }

  // Experimental DWARF5 features.

  /// \brief Returns whether or not to emit tables that dwarf consumers can
  /// use to accelerate lookup.
  bool useDwarfAccelTables() { return HasDwarfAccelTables; }

  /// \brief Returns whether or not to change the current debug info for the
  /// split dwarf proposal support.
  bool useSplitDwarf() { return HasSplitDwarf; }
};
} // End of namespace llvm

#endif
