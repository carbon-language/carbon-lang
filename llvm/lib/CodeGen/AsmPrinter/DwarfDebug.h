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
class DbgVariable;
class DbgScope;
class DbgConcreteScope;
class MachineFrameInfo;
class MachineModuleInfo;
class MCAsmInfo;
class Timer;

//===----------------------------------------------------------------------===//
/// SrcLineInfo - This class is used to record source line correspondence.
///
class VISIBILITY_HIDDEN SrcLineInfo {
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

class VISIBILITY_HIDDEN DwarfDebug : public Dwarf {
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

  /// ValuesSet - Used to uniquely define values.
  ///
  FoldingSet<DIEValue> ValuesSet;

  /// Values - A list of all the unique values in use.
  ///
  std::vector<DIEValue *> Values;

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

  // FunctionDbgScope - Top level scope for the current function.
  //
  DbgScope *FunctionDbgScope;
  
  /// DbgScopeMap - Tracks the scopes in the current function.
  DenseMap<MDNode *, DbgScope *> DbgScopeMap;

  /// ScopedGVs - Tracks global variables that are not at file scope.
  /// For example void f() { static int b = 42; }
  SmallVector<WeakVH, 4> ScopedGVs;

  typedef DenseMap<const MachineInstr *, SmallVector<DbgScope *, 2> > 
    InsnToDbgScopeMapTy;

  /// DbgScopeBeginMap - Maps instruction with a list DbgScopes it starts.
  InsnToDbgScopeMapTy DbgScopeBeginMap;

  /// DbgScopeEndMap - Maps instruction with a list DbgScopes it ends.
  InsnToDbgScopeMapTy DbgScopeEndMap;

  /// DbgAbstractScopeMap - Tracks abstract instance scopes in the current
  /// function.
  DenseMap<MDNode *, DbgScope *> DbgAbstractScopeMap;

  /// DbgConcreteScopeMap - Tracks concrete instance scopes in the current
  /// function.
  DenseMap<MDNode *,
           SmallVector<DbgScope *, 8> > DbgConcreteScopeMap;

  /// InlineInfo - Keep track of inlined functions and their location.  This
  /// information is used to populate debug_inlined section.
  DenseMap<MDNode *, SmallVector<unsigned, 4> > InlineInfo;

  /// AbstractInstanceRootMap - Map of abstract instance roots of inlined
  /// functions. These are subroutine entries that contain a DW_AT_inline
  /// attribute.
  DenseMap<const MDNode *, DbgScope *> AbstractInstanceRootMap;

  /// AbstractInstanceRootList - List of abstract instance roots of inlined
  /// functions. These are subroutine entries that contain a DW_AT_inline
  /// attribute.
  SmallVector<DbgScope *, 32> AbstractInstanceRootList;

  /// LexicalScopeStack - A stack of lexical scopes. The top one is the current
  /// scope.
  SmallVector<DbgScope *, 16> LexicalScopeStack;

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

  /// AssignAbbrevNumber - Define a unique number for the abbreviation.
  ///
  void AssignAbbrevNumber(DIEAbbrev &Abbrev);

  /// CreateDIEEntry - Creates a new DIEEntry to be a proxy for a debug
  /// information entry.
  DIEEntry *CreateDIEEntry(DIE *Entry = NULL);

  /// SetDIEEntry - Set a DIEEntry once the debug information entry is defined.
  ///
  void SetDIEEntry(DIEEntry *Value, DIE *Entry);

  /// AddUInt - Add an unsigned integer attribute data and value.
  ///
  void AddUInt(DIE *Die, unsigned Attribute, unsigned Form, uint64_t Integer);

  /// AddSInt - Add an signed integer attribute data and value.
  ///
  void AddSInt(DIE *Die, unsigned Attribute, unsigned Form, int64_t Integer);

  /// AddString - Add a string attribute data and value.
  ///
  void AddString(DIE *Die, unsigned Attribute, unsigned Form,
                 const std::string &String);

  /// AddLabel - Add a Dwarf label attribute data and value.
  ///
  void AddLabel(DIE *Die, unsigned Attribute, unsigned Form,
                const DWLabel &Label);

  /// AddObjectLabel - Add an non-Dwarf label attribute data and value.
  ///
  void AddObjectLabel(DIE *Die, unsigned Attribute, unsigned Form,
                      const std::string &Label);

  /// AddSectionOffset - Add a section offset label attribute data and value.
  ///
  void AddSectionOffset(DIE *Die, unsigned Attribute, unsigned Form,
                        const DWLabel &Label, const DWLabel &Section,
                        bool isEH = false, bool useSet = true);

  /// AddDelta - Add a label delta attribute data and value.
  ///
  void AddDelta(DIE *Die, unsigned Attribute, unsigned Form,
                const DWLabel &Hi, const DWLabel &Lo);

  /// AddDIEEntry - Add a DIE attribute data and value.
  ///
  void AddDIEEntry(DIE *Die, unsigned Attribute, unsigned Form, DIE *Entry) {
    Die->AddValue(Attribute, Form, CreateDIEEntry(Entry));
  }

  /// AddBlock - Add block data.
  ///
  void AddBlock(DIE *Die, unsigned Attribute, unsigned Form, DIEBlock *Block);

  /// AddSourceLine - Add location information to specified debug information
  /// entry.
  void AddSourceLine(DIE *Die, const DIVariable *V);
  void AddSourceLine(DIE *Die, const DIGlobal *G);
  void AddSourceLine(DIE *Die, const DISubprogram *SP);
  void AddSourceLine(DIE *Die, const DIType *Ty);

  /// AddAddress - Add an address attribute to a die based on the location
  /// provided.
  void AddAddress(DIE *Die, unsigned Attribute,
                  const MachineLocation &Location);

  /// AddComplexAddress - Start with the address based on the location provided,
  /// and generate the DWARF information necessary to find the actual variable
  /// (navigating the extra location information encoded in the type) based on
  /// the starting location.  Add the DWARF information to the die.
  ///
  void AddComplexAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                         const MachineLocation &Location);

  // FIXME: Should be reformulated in terms of AddComplexAddress.
  /// AddBlockByrefAddress - Start with the address based on the location
  /// provided, and generate the DWARF information necessary to find the
  /// actual Block variable (navigating the Block struct) based on the
  /// starting location.  Add the DWARF information to the die.  Obsolete,
  /// please use AddComplexAddress instead.
  ///
  void AddBlockByrefAddress(DbgVariable *&DV, DIE *Die, unsigned Attribute,
                            const MachineLocation &Location);

  /// AddType - Add a new type attribute to the specified entity.
  void AddType(CompileUnit *DW_Unit, DIE *Entity, DIType Ty);

  /// ConstructTypeDIE - Construct basic type die from DIBasicType.
  void ConstructTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                        DIBasicType BTy);

  /// ConstructTypeDIE - Construct derived type die from DIDerivedType.
  void ConstructTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                        DIDerivedType DTy);

  /// ConstructTypeDIE - Construct type DIE from DICompositeType.
  void ConstructTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                        DICompositeType CTy);

  /// ConstructSubrangeDIE - Construct subrange DIE from DISubrange.
  void ConstructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy);

  /// ConstructArrayTypeDIE - Construct array type DIE from DICompositeType.
  void ConstructArrayTypeDIE(CompileUnit *DW_Unit, DIE &Buffer, 
                             DICompositeType *CTy);

  /// ConstructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
  DIE *ConstructEnumTypeDIE(CompileUnit *DW_Unit, DIEnumerator *ETy);

  /// CreateGlobalVariableDIE - Create new DIE using GV.
  DIE *CreateGlobalVariableDIE(CompileUnit *DW_Unit,
                               const DIGlobalVariable &GV);

  /// CreateMemberDIE - Create new member DIE.
  DIE *CreateMemberDIE(CompileUnit *DW_Unit, const DIDerivedType &DT);

  /// CreateSubprogramDIE - Create new DIE using SP.
  DIE *CreateSubprogramDIE(CompileUnit *DW_Unit,
                           const DISubprogram &SP,
                           bool IsConstructor = false,
                           bool IsInlined = false);

  /// FindCompileUnit - Get the compile unit for the given descriptor. 
  ///
  CompileUnit &FindCompileUnit(DICompileUnit Unit) const;

  /// CreateDbgScopeVariable - Create a new scope variable.
  ///
  DIE *CreateDbgScopeVariable(DbgVariable *DV, CompileUnit *Unit);

  /// getDbgScope - Returns the scope associated with the given descriptor.
  ///
  DbgScope *getOrCreateScope(MDNode *N);
  DbgScope *getDbgScope(MDNode *N, const MachineInstr *MI, MDNode *InlinedAt);

  /// ConstructDbgScope - Construct the components of a scope.
  ///
  void ConstructDbgScope(DbgScope *ParentScope,
                         unsigned ParentStartID, unsigned ParentEndID,
                         DIE *ParentDie, CompileUnit *Unit);

  /// ConstructFunctionDbgScope - Construct the scope for the subprogram.
  ///
  void ConstructFunctionDbgScope(DbgScope *RootScope,
                                 bool AbstractScope = false);

  /// ConstructDefaultDbgScope - Construct a default scope for the subprogram.
  ///
  void ConstructDefaultDbgScope(MachineFunction *MF);

  /// EmitInitial - Emit initial Dwarf declarations.  This is necessary for cc
  /// tools to recognize the object file contains Dwarf information.
  void EmitInitial();

  /// EmitDIE - Recusively Emits a debug information entry.
  ///
  void EmitDIE(DIE *Die);

  /// SizeAndOffsetDie - Compute the size and offset of a DIE.
  ///
  unsigned SizeAndOffsetDie(DIE *Die, unsigned Offset, bool Last);

  /// SizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void SizeAndOffsets();

  /// EmitDebugInfo / EmitDebugInfoPerCU - Emit the debug info section.
  ///
  void EmitDebugInfoPerCU(CompileUnit *Unit);

  void EmitDebugInfo();

  /// EmitAbbreviations - Emit the abbreviation section.
  ///
  void EmitAbbreviations() const;

  /// EmitEndOfLineMatrix - Emit the last address of the section and the end of
  /// the line matrix.
  ///
  void EmitEndOfLineMatrix(unsigned SectionEnd);

  /// EmitDebugLines - Emit source line information.
  ///
  void EmitDebugLines();

  /// EmitCommonDebugFrame - Emit common frame info into a debug frame section.
  ///
  void EmitCommonDebugFrame();

  /// EmitFunctionDebugFrame - Emit per function frame info into a debug frame
  /// section.
  void EmitFunctionDebugFrame(const FunctionDebugFrameInfo &DebugFrameInfo);

  void EmitDebugPubNamesPerCU(CompileUnit *Unit);

  /// EmitDebugPubNames - Emit visible names into a debug pubnames section.
  ///
  void EmitDebugPubNames();

  /// EmitDebugStr - Emit visible names into a debug str section.
  ///
  void EmitDebugStr();

  /// EmitDebugLoc - Emit visible names into a debug loc section.
  ///
  void EmitDebugLoc();

  /// EmitDebugARanges - Emit visible names into a debug aranges section.
  ///
  void EmitDebugARanges();

  /// EmitDebugRanges - Emit visible names into a debug ranges section.
  ///
  void EmitDebugRanges();

  /// EmitDebugMacInfo - Emit visible names into a debug macinfo section.
  ///
  void EmitDebugMacInfo();

  /// EmitDebugInlineInfo - Emit inline info using following format.
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
  void EmitDebugInlineInfo();

  /// GetOrCreateSourceID - Look up the source id with the given directory and
  /// source file names. If none currently exists, create a new id and insert it
  /// in the SourceIds map. This can update DirectoryNames and SourceFileNames maps
  /// as well.
  unsigned GetOrCreateSourceID(const char *DirName,
                               const char *FileName);

  void ConstructCompileUnit(MDNode *N);

  void ConstructGlobalVariableDIE(MDNode *N);

  void ConstructSubprogram(MDNode *N);

  // FIXME: This should go away in favor of complex addresses.
  /// Find the type the programmer originally declared the variable to be
  /// and return that type.  Obsolete, use GetComplexAddrType instead.
  ///
  DIType GetBlockByrefType(DIType Ty, std::string Name);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T);
  virtual ~DwarfDebug();

  /// ShouldEmitDwarfDebug - Returns true if Dwarf debugging declarations should
  /// be emitted.
  bool ShouldEmitDwarfDebug() const { return shouldEmit; }

  /// BeginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void BeginModule(Module *M, MachineModuleInfo *MMI);

  /// EndModule - Emit all Dwarf sections that should come after the content.
  ///
  void EndModule();

  /// BeginFunction - Gather pre-function debug information.  Assumes being
  /// emitted immediately after the function entry point.
  void BeginFunction(MachineFunction *MF);

  /// EndFunction - Gather and emit post-function debug information.
  ///
  void EndFunction(MachineFunction *MF);

  /// RecordSourceLine - Records location information and associates it with a 
  /// label. Returns a unique label ID used to generate a label and provide
  /// correspondence to the source line list.
  unsigned RecordSourceLine(unsigned Line, unsigned Col, MDNode *Scope);

  /// getRecordSourceLineCount - Return the number of source lines in the debug
  /// info.
  unsigned getRecordSourceLineCount() const {
    return Lines.size();
  }
                            
  /// getOrCreateSourceID - Public version of GetOrCreateSourceID. This can be
  /// timed. Look up the source id with the given directory and source file
  /// names. If none currently exists, create a new id and insert it in the
  /// SourceIds map. This can update DirectoryNames and SourceFileNames maps as
  /// well.
  unsigned getOrCreateSourceID(const std::string &DirName,
                               const std::string &FileName);

  /// RecordRegionStart - Indicate the start of a region.
  unsigned RecordRegionStart(MDNode *N);

  /// RecordRegionEnd - Indicate the end of a region.
  unsigned RecordRegionEnd(MDNode *N);

  /// RecordVariable - Indicate the declaration of  a local variable.
  void RecordVariable(MDNode *N, unsigned FrameIndex);

  //// RecordInlinedFnStart - Indicate the start of inlined subroutine.
  unsigned RecordInlinedFnStart(DISubprogram &SP, DICompileUnit CU,
                                unsigned Line, unsigned Col);

  /// RecordInlinedFnEnd - Indicate the end of inlined subroutine.
  unsigned RecordInlinedFnEnd(DISubprogram &SP);

  /// ExtractScopeInformation - Scan machine instructions in this function
  /// and collect DbgScopes. Return true, if atleast one scope was found.
  bool ExtractScopeInformation(MachineFunction *MF);

  /// CollectVariableInfo - Populate DbgScope entries with variables' info.
  void CollectVariableInfo();

  /// SetDbgScopeBeginLabels - Update DbgScope begin labels for the scopes that
  /// start with this machine instruction.
  void SetDbgScopeBeginLabels(const MachineInstr *MI, unsigned Label);

  /// SetDbgScopeEndLabels - Update DbgScope end labels for the scopes that
  /// end with this machine instruction.
  void SetDbgScopeEndLabels(const MachineInstr *MI, unsigned Label);
};

} // End of namespace llvm

#endif
