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
#include "DwarfPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/UniqueVector.h"
#include <string>

namespace llvm {

class CompileUnit;
class DbgConcreteScope;
class DbgScope;
class DbgVariable;
class MachineFrameInfo;
class MachineModuleInfo;
class MCAsmInfo;
class Timer;

//===----------------------------------------------------------------------===//
/// SrcLineInfo - This class is used to record source line correspondence.
///
class SrcLineInfo {
  unsigned Line;                     // Source line number.
  unsigned Column;                   // Source column.
  unsigned SourceID;                 // Source ID number.
  unsigned LabelID;                  // Label in code ID number.
public:
  SrcLineInfo(unsigned L, unsigned C, unsigned S, unsigned I)
    : Line(L), Column(C), SourceID(S), LabelID(I) {}

  // Accessors
  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  unsigned getLabelID() const { return LabelID; }
};

class DwarfDebug : public DwarfPrinter {
  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //

  /// CompileUnitMap - A map of global variables representing compile units to
  /// compile units.
  DenseMap<Value *, CompileUnit *> CompileUnitMap;

  /// CompileUnits - All the compile units in this module.
  ///
  SmallVector<CompileUnit *, 8> CompileUnits;

  /// ModuleCU - All DIEs are inserted in ModuleCU.
  CompileUnit *ModuleCU;

  /// AbbreviationsSet - Used to uniquely define abbreviations.
  ///
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Abbreviations - A list of all the unique abbreviations in use.
  ///
  std::vector<DIEAbbrev *> Abbreviations;

  /// DirectoryIdMap - Directory name to directory id map.
  ///
  StringMap<unsigned> DirectoryIdMap;

  /// DirectoryNames - A list of directory names.
  SmallVector<std::string, 8> DirectoryNames;

  /// SourceFileIdMap - Source file name to source file id map.
  ///
  StringMap<unsigned> SourceFileIdMap;

  /// SourceFileNames - A list of source file names.
  SmallVector<std::string, 8> SourceFileNames;

  /// SourceIdMap - Source id map, i.e. pair of directory id and source file
  /// id mapped to a unique id.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> SourceIdMap;

  /// SourceIds - Reverse map from source id to directory id + file id pair.
  ///
  SmallVector<std::pair<unsigned, unsigned>, 8> SourceIds;

  /// Lines - List of of source line correspondence.
  std::vector<SrcLineInfo> Lines;

  /// DIEValues - A list of all the unique values in use.
  ///
  std::vector<DIEValue *> DIEValues;

  /// StringPool - A UniqueVector of strings used by indirect references.
  ///
  UniqueVector<std::string> StringPool;

  /// SectionMap - Provides a unique id per text section.
  ///
  UniqueVector<const MCSection*> SectionMap;

  /// SectionSourceLines - Tracks line numbers per text section.
  ///
  std::vector<std::vector<SrcLineInfo> > SectionSourceLines;

  /// didInitial - Flag to indicate if initial emission has been done.
  ///
  bool didInitial;

  /// shouldEmit - Flag to indicate if debug information should be emitted.
  ///
  bool shouldEmit;

  // CurrentFnDbgScope - Top level scope for the current function.
  //
  DbgScope *CurrentFnDbgScope;
  
  /// DbgScopeMap - Tracks the scopes in the current function.
  ///
  DenseMap<MDNode *, DbgScope *> DbgScopeMap;

  /// ConcreteScopes - Tracks the concrete scopees in the current function.
  /// These scopes are also included in DbgScopeMap.
  DenseMap<MDNode *, DbgScope *> ConcreteScopes;

  /// AbstractScopes - Tracks the abstract scopes a module. These scopes are
  /// not included DbgScopeMap.
  DenseMap<MDNode *, DbgScope *> AbstractScopes;
  SmallVector<DbgScope *, 4>AbstractScopesList;

  /// AbstractVariables - Collection on abstract variables.
  DenseMap<MDNode *, DbgVariable *> AbstractVariables;

  /// InliendSubprogramDIEs - Collection of subprgram DIEs that are marked
  /// (at the end of the module) as DW_AT_inline.
  SmallPtrSet<DIE *, 4> InlinedSubprogramDIEs;

  DenseMap<DIE *, MDNode *> ContainingTypeMap;

  /// AbstractSubprogramDIEs - Collection of abstruct subprogram DIEs.
  SmallPtrSet<DIE *, 4> AbstractSubprogramDIEs;

  /// TopLevelDIEs - Collection of top level DIEs. 
  SmallPtrSet<DIE *, 4> TopLevelDIEs;
  SmallVector<DIE *, 4> TopLevelDIEsVector;

  typedef SmallVector<DbgScope *, 2> ScopeVector;
  typedef DenseMap<const MachineInstr *, ScopeVector>
    InsnToDbgScopeMapTy;

  /// DbgScopeBeginMap - Maps instruction with a list of DbgScopes it starts.
  InsnToDbgScopeMapTy DbgScopeBeginMap;

  /// DbgScopeEndMap - Maps instruction with a list DbgScopes it ends.
  InsnToDbgScopeMapTy DbgScopeEndMap;

  /// InlineInfo - Keep track of inlined functions and their location.  This
  /// information is used to populate debug_inlined section.
  typedef std::pair<unsigned, DIE *> InlineInfoLabels;
  DenseMap<MDNode *, SmallVector<InlineInfoLabels, 4> > InlineInfo;
  SmallVector<MDNode *, 4> InlinedSPNodes;

  /// CompileUnitOffsets - A vector of the offsets of the compile units. This is
  /// used when calculating the "origin" of a concrete instance of an inlined
  /// function.
  DenseMap<CompileUnit *, unsigned> CompileUnitOffsets;

  /// DebugTimer - Timer for the Dwarf debug writer.
  Timer *DebugTimer;
  
  struct FunctionDebugFrameInfo {
    unsigned Number;
    std::vector<MachineMove> Moves;

    FunctionDebugFrameInfo(unsigned Num, const std::vector<MachineMove> &M)
      : Number(Num), Moves(M) {}
  };

  std::vector<FunctionDebugFrameInfo> DebugFrames;

  /// getSourceDirectoryAndFileIds - Return the directory and file ids that
  /// maps to the source id. Source id starts at 1.
  std::pair<unsigned, unsigned>
  getSourceDirectoryAndFileIds(unsigned SId) const {
    return SourceIds[SId-1];
  }

  /// getNumSourceDirectories - Return the number of source directories in the
  /// debug info.
  unsigned getNumSourceDirectories() const {
    return DirectoryNames.size();
  }

  /// getSourceDirectoryName - Return the name of the directory corresponding
  /// to the id.
  const std::string &getSourceDirectoryName(unsigned Id) const {
    return DirectoryNames[Id - 1];
  }

  /// getSourceFileName - Return the name of the source file corresponding
  /// to the id.
  const std::string &getSourceFileName(unsigned Id) const {
    return SourceFileNames[Id - 1];
  }

  /// getNumSourceIds - Return the number of unique source ids.
  unsigned getNumSourceIds() const {
    return SourceIds.size();
  }

  /// assignAbbrevNumber - Define a unique number for the abbreviation.
  ///
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  /// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
  /// information entry.
  DIEEntry *createDIEEntry(DIE *Entry = NULL);

  /// addUInt - Add an unsigned integer attribute data and value.
  ///
  void addUInt(DIE *Die, unsigned Attribute, unsigned Form, uint64_t Integer);

  /// addSInt - Add an signed integer attribute data and value.
  ///
  void addSInt(DIE *Die, unsigned Attribute, unsigned Form, int64_t Integer);

  /// addString - Add a string attribute data and value.
  ///
  void addString(DIE *Die, unsigned Attribute, unsigned Form,
                 const StringRef Str);

  /// addLabel - Add a Dwarf label attribute data and value.
  ///
  void addLabel(DIE *Die, unsigned Attribute, unsigned Form,
                const DWLabel &Label);

  /// addObjectLabel - Add an non-Dwarf label attribute data and value.
  ///
  void addObjectLabel(DIE *Die, unsigned Attribute, unsigned Form,
                      const MCSymbol *Sym);

  /// addSectionOffset - Add a section offset label attribute data and value.
  ///
  void addSectionOffset(DIE *Die, unsigned Attribute, unsigned Form,
                        const DWLabel &Label, const DWLabel &Section,
                        bool isEH = false, bool useSet = true);

  /// addDelta - Add a label delta attribute data and value.
  ///
  void addDelta(DIE *Die, unsigned Attribute, unsigned Form,
                const DWLabel &Hi, const DWLabel &Lo);

  /// addDIEEntry - Add a DIE attribute data and value.
  ///
  void addDIEEntry(DIE *Die, unsigned Attribute, unsigned Form, DIE *Entry) {
    Die->addValue(Attribute, Form, createDIEEntry(Entry));
  }

  /// addBlock - Add block data.
  ///
  void addBlock(DIE *Die, unsigned Attribute, unsigned Form, DIEBlock *Block);

  /// addSourceLine - Add location information to specified debug information
  /// entry.
  void addSourceLine(DIE *Die, const DIVariable *V);
  void addSourceLine(DIE *Die, const DIGlobal *G);
  void addSourceLine(DIE *Die, const DISubprogram *SP);
  void addSourceLine(DIE *Die, const DIType *Ty);
  void addSourceLine(DIE *Die, const DINameSpace *NS);

  /// addAddress - Add an address attribute to a die based on the location
  /// provided.
  void addAddress(DIE *Die, unsigned Attribute,
                  const MachineLocation &Location);

  /// addComplexAddress - Start with the address based on the location provided,
  /// and generate the DWARF information necessary to find the actual variable
  /// (navigating the extra location information encoded in the type) based on
  /// the starting location.  Add the DWARF information to the die.
  ///
  void addComplexAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                         const MachineLocation &Location);

  // FIXME: Should be reformulated in terms of addComplexAddress.
  /// addBlockByrefAddress - Start with the address based on the location
  /// provided, and generate the DWARF information necessary to find the
  /// actual Block variable (navigating the Block struct) based on the
  /// starting location.  Add the DWARF information to the die.  Obsolete,
  /// please use addComplexAddress instead.
  ///
  void addBlockByrefAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                            const MachineLocation &Location);

  /// addToContextOwner - Add Die into the list of its context owner's children.
  void addToContextOwner(DIE *Die, DIDescriptor Context);

  /// addType - Add a new type attribute to the specified entity.
  void addType(DIE *Entity, DIType Ty);

 
  /// getOrCreateNameSpace - Create a DIE for DINameSpace.
  DIE *getOrCreateNameSpace(DINameSpace NS);

  /// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
  /// given DIType.
  DIE *getOrCreateTypeDIE(DIType Ty);

  void addPubTypes(DISubprogram SP);

  /// constructTypeDIE - Construct basic type die from DIBasicType.
  void constructTypeDIE(DIE &Buffer,
                        DIBasicType BTy);

  /// constructTypeDIE - Construct derived type die from DIDerivedType.
  void constructTypeDIE(DIE &Buffer,
                        DIDerivedType DTy);

  /// constructTypeDIE - Construct type DIE from DICompositeType.
  void constructTypeDIE(DIE &Buffer,
                        DICompositeType CTy);

  /// constructSubrangeDIE - Construct subrange DIE from DISubrange.
  void constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy);

  /// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
  void constructArrayTypeDIE(DIE &Buffer, 
                             DICompositeType *CTy);

  /// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
  DIE *constructEnumTypeDIE(DIEnumerator *ETy);

  /// createGlobalVariableDIE - Create new DIE using GV.
  DIE *createGlobalVariableDIE(const DIGlobalVariable &GV);

  /// createMemberDIE - Create new member DIE.
  DIE *createMemberDIE(const DIDerivedType &DT);

  /// createSubprogramDIE - Create new DIE using SP.
  DIE *createSubprogramDIE(const DISubprogram &SP, bool MakeDecl = false);

  /// findCompileUnit - Get the compile unit for the given descriptor. 
  ///
  CompileUnit *findCompileUnit(DICompileUnit Unit);

  /// getUpdatedDbgScope - Find or create DbgScope assicated with 
  /// the instruction. Initialize scope and update scope hierarchy.
  DbgScope *getUpdatedDbgScope(MDNode *N, const MachineInstr *MI, MDNode *InlinedAt);

  /// createDbgScope - Create DbgScope for the scope.
  void createDbgScope(MDNode *Scope, MDNode *InlinedAt);

  DbgScope *getOrCreateAbstractScope(MDNode *N);

  /// findAbstractVariable - Find abstract variable associated with Var.
  DbgVariable *findAbstractVariable(DIVariable &Var, unsigned FrameIdx, 
                                    DILocation &Loc);

  /// updateSubprogramScopeDIE - Find DIE for the given subprogram and 
  /// attach appropriate DW_AT_low_pc and DW_AT_high_pc attributes.
  /// If there are global variables in this scope then create and insert
  /// DIEs for these variables.
  DIE *updateSubprogramScopeDIE(MDNode *SPNode);

  /// constructLexicalScope - Construct new DW_TAG_lexical_block 
  /// for this scope and attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(DbgScope *Scope);

  /// constructInlinedScopeDIE - This scope represents inlined body of
  /// a function. Construct DIE to represent this concrete inlined copy
  /// of the function.
  DIE *constructInlinedScopeDIE(DbgScope *Scope);

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  DIE *constructVariableDIE(DbgVariable *DV, DbgScope *S);

  /// constructScopeDIE - Construct a DIE for this scope.
  DIE *constructScopeDIE(DbgScope *Scope);

  /// emitInitial - Emit initial Dwarf declarations.  This is necessary for cc
  /// tools to recognize the object file contains Dwarf information.
  void emitInitial();

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

  /// emitDebugLines - Emit source line information.
  ///
  void emitDebugLines();

  /// emitCommonDebugFrame - Emit common frame info into a debug frame section.
  ///
  void emitCommonDebugFrame();

  /// emitFunctionDebugFrame - Emit per function frame info into a debug frame
  /// section.
  void emitFunctionDebugFrame(const FunctionDebugFrameInfo &DebugFrameInfo);

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

  /// GetOrCreateSourceID - Look up the source id with the given directory and
  /// source file names. If none currently exists, create a new id and insert it
  /// in the SourceIds map. This can update DirectoryNames and SourceFileNames maps
  /// as well.
  unsigned GetOrCreateSourceID(StringRef DirName, StringRef FileName);

  CompileUnit *constructCompileUnit(MDNode *N);

  void constructGlobalVariableDIE(MDNode *N);

  void constructSubprogramDIE(MDNode *N);

  // FIXME: This should go away in favor of complex addresses.
  /// Find the type the programmer originally declared the variable to be
  /// and return that type.  Obsolete, use GetComplexAddrType instead.
  ///
  DIType getBlockByrefType(DIType Ty, std::string Name);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T);
  virtual ~DwarfDebug();

  /// ShouldEmitDwarfDebug - Returns true if Dwarf debugging declarations should
  /// be emitted.
  bool ShouldEmitDwarfDebug() const { return shouldEmit; }

  /// beginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule(Module *M, MachineModuleInfo *MMI);

  /// endModule - Emit all Dwarf sections that should come after the content.
  ///
  void endModule();

  /// beginFunction - Gather pre-function debug information.  Assumes being
  /// emitted immediately after the function entry point.
  void beginFunction(MachineFunction *MF);

  /// endFunction - Gather and emit post-function debug information.
  ///
  void endFunction(MachineFunction *MF);

  /// recordSourceLine - Records location information and associates it with a 
  /// label. Returns a unique label ID used to generate a label and provide
  /// correspondence to the source line list.
  unsigned recordSourceLine(unsigned Line, unsigned Col, MDNode *Scope);

  /// getSourceLineCount - Return the number of source lines in the debug
  /// info.
  unsigned getSourceLineCount() const {
    return Lines.size();
  }
                            
  /// getOrCreateSourceID - Public version of GetOrCreateSourceID. This can be
  /// timed. Look up the source id with the given directory and source file
  /// names. If none currently exists, create a new id and insert it in the
  /// SourceIds map. This can update DirectoryNames and SourceFileNames maps as
  /// well.
  unsigned getOrCreateSourceID(const std::string &DirName,
                               const std::string &FileName);

  /// extractScopeInformation - Scan machine instructions in this function
  /// and collect DbgScopes. Return true, if atleast one scope was found.
  bool extractScopeInformation(MachineFunction *MF);

  /// collectVariableInfo - Populate DbgScope entries with variables' info.
  void collectVariableInfo();

  /// beginScope - Process beginning of a scope starting at Label.
  void beginScope(const MachineInstr *MI, unsigned Label);

  /// endScope - Prcess end of a scope.
  void endScope(const MachineInstr *MI);
};
} // End of namespace llvm

#endif
