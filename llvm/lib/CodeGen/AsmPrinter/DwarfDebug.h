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

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Analysis/DebugInfo.h"
#include "DIE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLoc.h"

namespace llvm {

class CompileUnit;
class DbgConcreteScope;
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
/// SrcLineInfo - This class is used to record source line correspondence.
///
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

/// DotDebugLocEntry - This struct describes location entries emitted in
/// .debug_loc section.
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
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, const ConstantInt *IPtr)
    : Begin(B), End(E), Variable(0), Merged(false), 
      Constant(true) { Constants.CIP = IPtr; EntryKind = E_ConstantInt; }

  /// Empty entries are also used as a trigger to emit temp label. Such
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
/// DbgVariable - This class is used to track local variable information.
///
class DbgVariable {
  DIVariable Var;                    // Variable Descriptor.
  DIE *TheDIE;                       // Variable DIE.
  unsigned DotDebugLocOffset;        // Offset in DotDebugLocEntries.
public:
  // AbsVar may be NULL.
  DbgVariable(DIVariable V) : Var(V), TheDIE(0), DotDebugLocOffset(~0U) {}

  // Accessors.
  DIVariable getVariable()           const { return Var; }
  void setDIE(DIE *D)                      { TheDIE = D; }
  DIE *getDIE()                      const { return TheDIE; }
  void setDotDebugLocOffset(unsigned O)    { DotDebugLocOffset = O; }
  unsigned getDotDebugLocOffset()    const { return DotDebugLocOffset; }
  StringRef getName()                const { return Var.getName(); }
  unsigned getTag()                  const { return Var.getTag(); }
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

class DwarfDebug {
  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //

  CompileUnit *FirstCU;
  DenseMap <const MDNode *, CompileUnit *> CUMap;

  /// AbbreviationsSet - Used to uniquely define abbreviations.
  ///
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Abbreviations - A list of all the unique abbreviations in use.
  ///
  std::vector<DIEAbbrev *> Abbreviations;

  /// SourceIdMap - Source id map, i.e. pair of directory id and source file
  /// id mapped to a unique id.
  StringMap<unsigned> SourceIdMap;

  /// StringPool - A String->Symbol mapping of strings used by indirect
  /// references.
  StringMap<std::pair<MCSymbol*, unsigned> > StringPool;
  unsigned NextStringPoolNumber;
  
  MCSymbol *getStringPoolEntry(StringRef Str);

  /// SectionMap - Provides a unique id per text section.
  ///
  UniqueVector<const MCSection*> SectionMap;

  /// CurrentFnArguments - List of Arguments (DbgValues) for current function.
  SmallVector<DbgVariable *, 8> CurrentFnArguments;

  LexicalScopes LScopes;

  /// AbstractSPDies - Collection of abstract subprogram DIEs.
  DenseMap<const MDNode *, DIE *> AbstractSPDies;

  /// ScopeVariables - Collection of dbg variables of a scope.
  DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8> > ScopeVariables;

  /// AbstractVariables - Collection on abstract variables.
  DenseMap<const MDNode *, DbgVariable *> AbstractVariables;

  /// DbgVariableToFrameIndexMap - Tracks frame index used to find 
  /// variable's value.
  DenseMap<const DbgVariable *, int> DbgVariableToFrameIndexMap;

  /// DbgVariableToDbgInstMap - Maps DbgVariable to corresponding DBG_VALUE
  /// machine instruction.
  DenseMap<const DbgVariable *, const MachineInstr *> DbgVariableToDbgInstMap;

  /// DotDebugLocEntries - Collection of DotDebugLocEntry.
  SmallVector<DotDebugLocEntry, 4> DotDebugLocEntries;

  /// UseDotDebugLocEntry - DW_AT_location attributes for the DIEs in this set
  /// idetifies corresponding .debug_loc entry offset.
  SmallPtrSet<const DIE *, 4> UseDotDebugLocEntry;

  /// VarToAbstractVarMap - Maps DbgVariable with corresponding Abstract
  /// DbgVariable, if any.
  DenseMap<const DbgVariable *, const DbgVariable *> VarToAbstractVarMap;

  /// InliendSubprogramDIEs - Collection of subprgram DIEs that are marked
  /// (at the end of the module) as DW_AT_inline.
  SmallPtrSet<DIE *, 4> InlinedSubprogramDIEs;

  /// ContainingTypeMap - This map is used to keep track of subprogram DIEs that
  /// need DW_AT_containing_type attribute. This attribute points to a DIE that
  /// corresponds to the MDNode mapped with the subprogram DIE.
  DenseMap<DIE *, const MDNode *> ContainingTypeMap;

  /// InlineInfo - Keep track of inlined functions and their location.  This
  /// information is used to populate debug_inlined section.
  typedef std::pair<const MCSymbol *, DIE *> InlineInfoLabels;
  DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> > InlineInfo;
  SmallVector<const MDNode *, 4> InlinedSPNodes;

  // ProcessedSPNodes - This is a collection of subprogram MDNodes that
  // are processed to create DIEs.
  SmallPtrSet<const MDNode *, 16> ProcessedSPNodes;

  /// LabelsBeforeInsn - Maps instruction with label emitted before 
  /// instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsBeforeInsn;

  /// LabelsAfterInsn - Maps instruction with label emitted after
  /// instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsAfterInsn;

  /// UserVariables - Every user variable mentioned by a DBG_VALUE instruction
  /// in order of appearance.
  SmallVector<const MDNode*, 8> UserVariables;

  /// DbgValues - For each user variable, keep a list of DBG_VALUE
  /// instructions in order. The list can also contain normal instructions that
  /// clobber the previous DBG_VALUE.
  typedef DenseMap<const MDNode*, SmallVector<const MachineInstr*, 4> >
    DbgValueHistoryMap;
  DbgValueHistoryMap DbgValues;

  SmallVector<const MCSymbol *, 8> DebugRangeSymbols;

  /// Previous instruction's location information. This is used to determine
  /// label location to indicate scope boundries in dwarf debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel;

  /// PrologEndLoc - This location indicates end of function prologue and
  /// beginning of function body.
  DebugLoc PrologEndLoc;

  struct FunctionDebugFrameInfo {
    unsigned Number;
    std::vector<MachineMove> Moves;

    FunctionDebugFrameInfo(unsigned Num, const std::vector<MachineMove> &M)
      : Number(Num), Moves(M) {}
  };

  std::vector<FunctionDebugFrameInfo> DebugFrames;

  // DIEValueAllocator - All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  // Section Symbols: these are assembler temporary labels that are emitted at
  // the beginning of each supported dwarf section.  These are used to form
  // section offsets and are created by EmitSectionLabels.
  MCSymbol *DwarfInfoSectionSym, *DwarfAbbrevSectionSym;
  MCSymbol *DwarfStrSectionSym, *TextSectionSym, *DwarfDebugRangeSectionSym;
  MCSymbol *DwarfDebugLocSectionSym;
  MCSymbol *FunctionBeginSym, *FunctionEndSym;

private:

  /// assignAbbrevNumber - Define a unique number for the abbreviation.
  ///
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  void addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  /// findAbstractVariable - Find abstract variable associated with Var.
  DbgVariable *findAbstractVariable(DIVariable &Var, DebugLoc Loc);

  /// updateSubprogramScopeDIE - Find DIE for the given subprogram and 
  /// attach appropriate DW_AT_low_pc and DW_AT_high_pc attributes.
  /// If there are global variables in this scope then create and insert
  /// DIEs for these variables.
  DIE *updateSubprogramScopeDIE(const MDNode *SPNode);

  /// constructLexicalScope - Construct new DW_TAG_lexical_block 
  /// for this scope and attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(LexicalScope *Scope);

  /// constructInlinedScopeDIE - This scope represents inlined body of
  /// a function. Construct DIE to represent this concrete inlined copy
  /// of the function.
  DIE *constructInlinedScopeDIE(LexicalScope *Scope);

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  DIE *constructVariableDIE(DbgVariable *DV, LexicalScope *S);

  /// constructScopeDIE - Construct a DIE for this scope.
  DIE *constructScopeDIE(LexicalScope *Scope);

  /// EmitSectionLabels - Emit initial Dwarf sections with a label at
  /// the start of each one.
  void EmitSectionLabels();

  /// emitDIE - Recusively Emits a debug information entry.
  ///
  void emitDIE(DIE *Die);

  /// computeSizeAndOffset - Compute the size and offset of a DIE.
  ///
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset, bool Last);

  /// computeSizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void computeSizeAndOffsets();

  /// EmitDebugInfo - Emit the debug info section.
  ///
  void emitDebugInfo();

  /// emitAbbreviations - Emit the abbreviation section.
  ///
  void emitAbbreviations() const;

  /// emitEndOfLineMatrix - Emit the last address of the section and the end of
  /// the line matrix.
  ///
  void emitEndOfLineMatrix(unsigned SectionEnd);

  /// emitDebugPubNames - Emit visible names into a debug pubnames section.
  ///
  void emitDebugPubNames();

  /// emitDebugPubTypes - Emit visible types into a debug pubtypes section.
  ///
  void emitDebugPubTypes();

  /// emitDebugStr - Emit visible names into a debug str section.
  ///
  void emitDebugStr();

  /// emitDebugLoc - Emit visible names into a debug loc section.
  ///
  void emitDebugLoc();

  /// EmitDebugARanges - Emit visible names into a debug aranges section.
  ///
  void EmitDebugARanges();

  /// emitDebugRanges - Emit visible names into a debug ranges section.
  ///
  void emitDebugRanges();

  /// emitDebugMacInfo - Emit visible names into a debug macinfo section.
  ///
  void emitDebugMacInfo();

  /// emitDebugInlineInfo - Emit inline info using following format.
  /// Section Header:
  /// 1. length of section
  /// 2. Dwarf version number
  /// 3. address size.
  ///
  /// Entries (one "entry" for each function that was inlined):
  ///
  /// 1. offset into __debug_str section for MIPS linkage name, if exists; 
  ///   otherwise offset into __debug_str for regular function name.
  /// 2. offset into __debug_str section for regular function name.
  /// 3. an unsigned LEB128 number indicating the number of distinct inlining 
  /// instances for the function.
  /// 
  /// The rest of the entry consists of a {die_offset, low_pc}  pair for each 
  /// inlined instance; the die_offset points to the inlined_subroutine die in
  /// the __debug_info section, and the low_pc is the starting address  for the
  ///  inlining instance.
  void emitDebugInlineInfo();

  /// constructCompileUnit - Create new CompileUnit for the given 
  /// metadata node with tag DW_TAG_compile_unit.
  void constructCompileUnit(const MDNode *N);

  /// getCompielUnit - Get CompileUnit DIE.
  CompileUnit *getCompileUnit(const MDNode *N) const;

  /// constructGlobalVariableDIE - Construct global variable DIE.
  void constructGlobalVariableDIE(const MDNode *N);

  /// construct SubprogramDIE - Construct subprogram DIE.
  void constructSubprogramDIE(const MDNode *N);

  /// recordSourceLine - Register a source line with debug info. Returns the
  /// unique label that was emitted and which provides correspondence to
  /// the source line list.
  void recordSourceLine(unsigned Line, unsigned Col, const MDNode *Scope,
                        unsigned Flags);
  
  /// recordVariableFrameIndex - Record a variable's index.
  void recordVariableFrameIndex(const DbgVariable *V, int Index);

  /// findVariableFrameIndex - Return true if frame index for the variable
  /// is found. Update FI to hold value of the index.
  bool findVariableFrameIndex(const DbgVariable *V, int *FI);

  /// identifyScopeMarkers() - Indentify instructions that are marking
  /// beginning of or end of a scope.
  void identifyScopeMarkers();

  /// addCurrentFnArgument - If Var is an current function argument that add
  /// it in CurrentFnArguments list.
  bool addCurrentFnArgument(const MachineFunction *MF,
                            DbgVariable *Var, LexicalScope *Scope);

  /// collectVariableInfo - Populate LexicalScope entries with variables' info.
  void collectVariableInfo(const MachineFunction *,
                           SmallPtrSet<const MDNode *, 16> &ProcessedVars);
  
  /// collectVariableInfoFromMMITable - Collect variable information from
  /// side table maintained by MMI.
  void collectVariableInfoFromMMITable(const MachineFunction * MF,
                                       SmallPtrSet<const MDNode *, 16> &P);

  /// requestLabelBeforeInsn - Ensure that a label will be emitted before MI.
  void requestLabelBeforeInsn(const MachineInstr *MI) {
    LabelsBeforeInsn.insert(std::make_pair(MI, (MCSymbol*)0));
  }

  /// getLabelBeforeInsn - Return Label preceding the instruction.
  const MCSymbol *getLabelBeforeInsn(const MachineInstr *MI);

  /// requestLabelAfterInsn - Ensure that a label will be emitted after MI.
  void requestLabelAfterInsn(const MachineInstr *MI) {
    LabelsAfterInsn.insert(std::make_pair(MI, (MCSymbol*)0));
  }

  /// getLabelAfterInsn - Return Label immediately following the instruction.
  const MCSymbol *getLabelAfterInsn(const MachineInstr *MI);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A, Module *M);
  ~DwarfDebug();

  /// beginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule(Module *M);

  /// endModule - Emit all Dwarf sections that should come after the content.
  ///
  void endModule();

  /// beginFunction - Gather pre-function debug information.  Assumes being
  /// emitted immediately after the function entry point.
  void beginFunction(const MachineFunction *MF);

  /// endFunction - Gather and emit post-function debug information.
  ///
  void endFunction(const MachineFunction *MF);

  /// beginInstruction - Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI);

  /// endInstruction - Prcess end of an instruction.
  void endInstruction(const MachineInstr *MI);

  /// GetOrCreateSourceID - Look up the source id with the given directory and
  /// source file names. If none currently exists, create a new id and insert it
  /// in the SourceIds map.
  unsigned GetOrCreateSourceID(StringRef DirName, StringRef FullName);

  /// createSubprogramDIE - Create new DIE using SP.
  DIE *createSubprogramDIE(DISubprogram SP);
};
} // End of namespace llvm

#endif
