//===-- llvm/CodeGen/DwarfWriter.h - Dwarf Framework ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Dwarf debug info into asm files.  For
// Details on the Dwarf 3 specfication see DWARF Debugging Information Format
// V.3 reference manual http://dwarf.freestandards.org ,
//
// The role of the Dwarf Writer class is to extract debug information from the
// MachineDebugInfo object, organize it in Dwarf form and then emit it into asm
// the current asm file using data and high level Dwarf directives.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFWRITER_H
#define LLVM_CODEGEN_DWARFWRITER_H

#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/DataTypes.h"

#include <iosfwd>
#include <string>


namespace llvm {

// Forward declarations.

class AsmPrinter;
class CompileUnit;
class CompileUnitDesc;
class DebugInfoDesc;
class DebugVariable;
class DebugScope;
class DIE;
class DIEAbbrev;
class GlobalVariableDesc;
class MachineDebugInfo;
class MachineFunction;
class MachineLocation;
class MachineMove;
class Module;
class MRegisterInfo;
class SubprogramDesc;
class SourceLineInfo;
class TargetData;
class Type;
class TypeDesc;
  
//===----------------------------------------------------------------------===//
// DWLabel - Labels are used to track locations in the assembler file.
// Labels appear in the form <prefix>debug_<Tag><Number>, where the tag is a
// category of label (Ex. location) and number is a value unique in that
// category.
class DWLabel {
public:
  const char *Tag;                    // Label category tag. Should always be
                                      // a staticly declared C string.
  unsigned    Number;                 // Unique number.

  DWLabel(const char *T, unsigned N) : Tag(T), Number(N) {}
};

//===----------------------------------------------------------------------===//
// DwarfWriter - Emits Dwarf debug and exception handling directives.
//
class DwarfWriter {
protected:

  //===--------------------------------------------------------------------===//
  // Core attributes used by the Dwarf  writer.
  //
  
  //
  /// O - Stream to .s file.
  ///
  std::ostream &O;

  /// Asm - Target of Dwarf emission.
  ///
  AsmPrinter *Asm;
  
  /// TD - Target data.
  const TargetData *TD;
  
  /// RI - Register Information.
  const MRegisterInfo *RI;
  
  /// M - Current module.
  ///
  Module *M;
  
  /// MF - Current machine function.
  ///
  MachineFunction *MF;
  
  /// DebugInfo - Collected debug information.
  ///
  MachineDebugInfo *DebugInfo;
  
  /// didInitial - Flag to indicate if initial emission has been done.
  ///
  bool didInitial;
  
  /// shouldEmit - Flag to indicate if debug information should be emitted.
  ///
  bool shouldEmit;
  
  /// SubprogramCount - The running count of functions being compiled.
  ///
  unsigned SubprogramCount;
  
  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //
  
  /// CompileUnits - All the compile units involved in this build.  The index
  /// of each entry in this vector corresponds to the sources in DebugInfo.
  std::vector<CompileUnit *> CompileUnits;

  /// Abbreviations - A UniqueVector of TAG structure abbreviations.
  ///
  UniqueVector<DIEAbbrev> Abbreviations;
  
  /// StringPool - A UniqueVector of strings used by indirect references.
  /// UnitMap - Map debug information descriptor to compile unit.
   ///
  UniqueVector<std::string> StringPool;

  /// UnitMap - Map debug information descriptor to compile unit.
  ///
  std::map<DebugInfoDesc *, CompileUnit *> DescToUnitMap;
  
  /// DescToDieMap - Tracks the mapping of top level debug informaton
  /// descriptors to debug information entries.
  std::map<DebugInfoDesc *, DIE *> DescToDieMap;
  
  /// SectionMap - Provides a unique id per text section.
  ///
  UniqueVector<std::string> SectionMap;
  
  /// SectionSourceLines - Tracks line numbers per text section.
  ///
  std::vector<std::vector<SourceLineInfo *> > SectionSourceLines;
  
  //===--------------------------------------------------------------------===//
  // Properties to be set by the derived class ctor, used to configure the
  // Dwarf writer.
  //
  
  /// AddressSize - Size of addresses used in file.
  ///
  unsigned AddressSize;

  /// hasLEB128 - True if target asm supports leb128 directives.
  ///
  bool hasLEB128; /// Defaults to false.
  
  /// hasDotLoc - True if target asm supports .loc directives.
  ///
  bool hasDotLoc; /// Defaults to false.
  
  /// hasDotFile - True if target asm supports .file directives.
  ///
  bool hasDotFile; /// Defaults to false.
  
  /// needsSet - True if target asm can't compute addresses on data
  /// directives.
  bool needsSet; /// Defaults to false.
  
  /// DwarfAbbrevSection - Section directive for Dwarf abbrev.
  ///
  const char *DwarfAbbrevSection; /// Defaults to ".debug_abbrev".

  /// DwarfInfoSection - Section directive for Dwarf info.
  ///
  const char *DwarfInfoSection; /// Defaults to ".debug_info".

  /// DwarfLineSection - Section directive for Dwarf info.
  ///
  const char *DwarfLineSection; /// Defaults to ".debug_line".
  
  /// DwarfFrameSection - Section directive for Dwarf info.
  ///
  const char *DwarfFrameSection; /// Defaults to ".debug_frame".
  
  /// DwarfPubNamesSection - Section directive for Dwarf info.
  ///
  const char *DwarfPubNamesSection; /// Defaults to ".debug_pubnames".
  
  /// DwarfPubTypesSection - Section directive for Dwarf info.
  ///
  const char *DwarfPubTypesSection; /// Defaults to ".debug_pubtypes".
  
  /// DwarfStrSection - Section directive for Dwarf info.
  ///
  const char *DwarfStrSection; /// Defaults to ".debug_str".

  /// DwarfLocSection - Section directive for Dwarf info.
  ///
  const char *DwarfLocSection; /// Defaults to ".debug_loc".

  /// DwarfARangesSection - Section directive for Dwarf info.
  ///
  const char *DwarfARangesSection; /// Defaults to ".debug_aranges".

  /// DwarfRangesSection - Section directive for Dwarf info.
  ///
  const char *DwarfRangesSection; /// Defaults to ".debug_ranges".

  /// DwarfMacInfoSection - Section directive for Dwarf info.
  ///
  const char *DwarfMacInfoSection; /// Defaults to ".debug_macinfo".

  /// TextSection - Section directive for standard text.
  ///
  const char *TextSection; /// Defaults to ".text".
  
  /// DataSection - Section directive for standard data.
  ///
  const char *DataSection; /// Defaults to ".data".

  //===--------------------------------------------------------------------===//
  // Emission and print routines
  //

public:
  /// getAddressSize - Return the size of a target address in bytes.
  ///
  unsigned getAddressSize() const { return AddressSize; }

  /// PrintHex - Print a value as a hexidecimal value.
  ///
  void PrintHex(int Value) const;

  /// EOL - Print a newline character to asm stream.  If a comment is present
  /// then it will be printed first.  Comments should not contain '\n'.
  void EOL(const std::string &Comment) const;
  
  /// EmitAlign - Print a align directive.
  ///
  void EmitAlign(unsigned Alignment) const;
                                        
  /// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
  /// unsigned leb128 value.
  void EmitULEB128Bytes(unsigned Value) const;
  
  /// EmitSLEB128Bytes - print an assembler byte data directive to compose a
  /// signed leb128 value.
  void EmitSLEB128Bytes(int Value) const;
  
  /// PrintULEB128 - Print a series of hexidecimal values (separated by
  /// commas) representing an unsigned leb128 value.
  void PrintULEB128(unsigned Value) const;

  /// SizeULEB128 - Compute the number of bytes required for an unsigned
  /// leb128 value.
  static unsigned SizeULEB128(unsigned Value);
  
  /// PrintSLEB128 - Print a series of hexidecimal values (separated by
  /// commas) representing a signed leb128 value.
  void PrintSLEB128(int Value) const;
  
  /// SizeSLEB128 - Compute the number of bytes required for a signed leb128
  /// value.
  static unsigned SizeSLEB128(int Value);
  
  /// EmitInt8 - Emit a byte directive and value.
  ///
  void EmitInt8(int Value) const;

  /// EmitInt16 - Emit a short directive and value.
  ///
  void EmitInt16(int Value) const;

  /// EmitInt32 - Emit a long directive and value.
  ///
  void EmitInt32(int Value) const;
  
  /// EmitInt64 - Emit a long long directive and value.
  ///
  void EmitInt64(uint64_t Value) const;
  
  /// EmitString - Emit a string with quotes and a null terminator.
  /// Special characters are emitted properly. 
  /// \literal (Eg. '\t') \endliteral
  void EmitString(const std::string &String) const;

  /// PrintLabelName - Print label name in form used by Dwarf writer.
  ///
  void PrintLabelName(DWLabel Label) const {
    PrintLabelName(Label.Tag, Label.Number);
  }
  void PrintLabelName(const char *Tag, unsigned Number) const;
  
  /// EmitLabel - Emit location label for internal use by Dwarf.
  ///
  void EmitLabel(DWLabel Label) const {
    EmitLabel(Label.Tag, Label.Number);
  }
  void EmitLabel(const char *Tag, unsigned Number) const;
  
  /// EmitReference - Emit a reference to a label.
  ///
  void EmitReference(DWLabel Label) const {
    EmitReference(Label.Tag, Label.Number);
  }
  void EmitReference(const char *Tag, unsigned Number) const;
  void EmitReference(const std::string &Name) const;

  /// EmitDifference - Emit the difference between two labels.  Some
  /// assemblers do not behave with absolute expressions with data directives,
  /// so there is an option (needsSet) to use an intermediary set expression.
  void EmitDifference(DWLabel LabelHi, DWLabel LabelLo) const {
    EmitDifference(LabelHi.Tag, LabelHi.Number, LabelLo.Tag, LabelLo.Number);
  }
  void EmitDifference(const char *TagHi, unsigned NumberHi,
                      const char *TagLo, unsigned NumberLo) const;
                      
  /// NewAbbreviation - Add the abbreviation to the Abbreviation vector.
  ///  
  unsigned NewAbbreviation(DIEAbbrev *Abbrev);
  
  /// NewString - Add a string to the constant pool and returns a label.
  ///
  DWLabel NewString(const std::string &String);
  
  /// getDieMapSlotFor - Returns the debug information entry map slot for the
  /// specified debug descriptor.
  DIE *&getDieMapSlotFor(DebugInfoDesc *DD);
                                 
private:

  /// AddSourceLine - Add location information to specified debug information
  /// entry. 
  void AddSourceLine(DIE *Die, CompileUnitDesc *File, unsigned Line);

  /// AddAddress - Add an address attribute to a die based on the location
  /// provided.
  void AddAddress(DIE *Die, unsigned Attribute,
                  const MachineLocation &Location);

  /// NewType - Create a new type DIE.
  ///
  DIE *NewType(DIE *Context, TypeDesc *TyDesc, CompileUnit *Unit);
  
  /// NewCompileUnit - Create new compile unit and it's die.
  ///
  CompileUnit *NewCompileUnit(CompileUnitDesc *UnitDesc, unsigned ID);
  
  /// FindCompileUnit - Get the compile unit for the given descriptor.
  ///
  CompileUnit *FindCompileUnit(CompileUnitDesc *UnitDesc);
  
  /// NewGlobalVariable - Make a new global variable DIE.
  ///
  DIE *NewGlobalVariable(GlobalVariableDesc *GVD);

  /// NewSubprogram - Add a new subprogram DIE.
  ///
  DIE *NewSubprogram(SubprogramDesc *SPD);

  /// NewScopeVariable - Create a new scope variable.
  ///
  DIE *NewScopeVariable(DebugVariable *DV, CompileUnit *Unit);

  /// ConstructScope - Construct the components of a scope.
  ///
  void ConstructScope(DebugScope *ParentScope, DIE *ParentDie,
                      CompileUnit *Unit);

  /// ConstructRootScope - Construct the scope for the subprogram.
  ///
  void ConstructRootScope(DebugScope *RootScope);

  /// EmitInitial - Emit initial Dwarf declarations.
  ///
  void EmitInitial();
  
  /// EmitDIE - Recusively Emits a debug information entry.
  ///
  void EmitDIE(DIE *Die) const;
  
  /// SizeAndOffsetDie - Compute the size and offset of a DIE.
  ///
  unsigned SizeAndOffsetDie(DIE *Die, unsigned Offset, bool Last);

  /// SizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void SizeAndOffsets();
  
  /// EmitFrameMoves - Emit frame instructions to describe the layout of the
  /// frame.
  void EmitFrameMoves(const char *BaseLabel, unsigned BaseLabelID,
                      std::vector<MachineMove *> &Moves);

  /// EmitDebugInfo - Emit the debug info section.
  ///
  void EmitDebugInfo() const;
  
  /// EmitAbbreviations - Emit the abbreviation section.
  ///
  void EmitAbbreviations() const;
  
  /// EmitDebugLines - Emit source line information.
  ///
  void EmitDebugLines() const;

  /// EmitInitialDebugFrame - Emit common frame info into a debug frame section.
  ///
  void EmitInitialDebugFrame();
    
  /// EmitFunctionDebugFrame - Emit per function frame info into a debug frame
  /// section.
  void EmitFunctionDebugFrame();

  /// EmitDebugPubNames - Emit info into a debug pubnames section.
  ///
  void EmitDebugPubNames();
  
  /// EmitDebugStr - Emit info into a debug str section.
  ///
  void EmitDebugStr();
  
  /// EmitDebugLoc - Emit info into a debug loc section.
  ///
  void EmitDebugLoc();
  
  /// EmitDebugARanges - Emit info into a debug aranges section.
  ///
  void EmitDebugARanges();
  
  /// EmitDebugRanges - Emit info into a debug ranges section.
  ///
  void EmitDebugRanges();
  
  /// EmitDebugMacInfo - Emit info into a debug macinfo section.
  ///
  void EmitDebugMacInfo();
  
  /// ConstructCompileUnitDIEs - Create a compile unit DIE for each source and
  /// header file.
  void ConstructCompileUnitDIEs();
  
  /// ConstructGlobalDIEs - Create DIEs for each of the externally visible
  /// global variables.
  void ConstructGlobalDIEs();

  /// ConstructSubprogramDIEs - Create DIEs for each of the externally visible
  /// subprograms.
  void ConstructSubprogramDIEs();

  /// ShouldEmitDwarf - Returns true if Dwarf declarations should be made.
  ///
  bool ShouldEmitDwarf() const { return shouldEmit; }

public:
  
  DwarfWriter(std::ostream &OS, AsmPrinter *A);
  virtual ~DwarfWriter();
  
  /// SetDebugInfo - Set DebugInfo when it's known that pass manager has
  /// created it.  Set by the target AsmPrinter.
  void SetDebugInfo(MachineDebugInfo *DI);

  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  
  /// BeginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void BeginModule(Module *M);
  
  /// EndModule - Emit all Dwarf sections that should come after the content.
  ///
  void EndModule();
  
  /// BeginFunction - Gather pre-function debug information.  Assumes being 
  /// emitted immediately after the function entry point.
  void BeginFunction(MachineFunction *MF);
  
  /// EndFunction - Gather and emit post-function debug information.
  ///
  void EndFunction();
  
  /// NonFunction - Function does not have a true body.
  ///
  void NonFunction();
};

} // end llvm namespace

#endif
