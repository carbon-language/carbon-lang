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
class CompileUnitDesc;
class DebugInfoDesc;
class DIE;
class DIEAbbrev;
class GlobalVariableDesc;
class MachineDebugInfo;
class MachineFunction;
class Module;
class SubprogramDesc;
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
  
  /// DebugInfo - Collected debug information.
  ///
  MachineDebugInfo *DebugInfo;
  
  /// didInitial - Flag to indicate if initial emission has been done.
  ///
  bool didInitial;
  
  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //
  
  /// CompileUnits - All the compile units involved in this build.  The index
  /// of each entry in this vector corresponds to the sources in DebugInfo.
  std::vector<DIE *> CompileUnits;

  /// Abbreviations - A UniqueVector of TAG structure abbreviations.
  ///
  UniqueVector<DIEAbbrev> Abbreviations;
  
  /// GlobalTypes - A map of globally visible named types.
  ///
  std::map<std::string, DIE *> GlobalTypes;
  
  /// GlobalEntities - A map of globally visible named entities.
  ///
  std::map<std::string, DIE *> GlobalEntities;
   
  /// StringPool - A UniqueVector of strings used by indirect references.
  ///
  UniqueVector<std::string> StringPool;
  
  /// DescToDieMap - Tracks the mapping of debug informaton descriptors to
  /// DIES.
  std::map<DebugInfoDesc *, DIE *> DescToDieMap;
  
  /// TypeToDieMap - Type to DIEType map.
  ///
  // FIXME - Should not be needed.
  std::map<Type *, DIE *> TypeToDieMap;
  
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
  /// Special characters are emitted properly. (Eg. '\t')
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
  
  /// NewBasicType - Creates a new basic type if necessary, then adds to the
  /// owner.
  /// FIXME - Should never be needed.
  DIE *NewBasicType(DIE *Owner, Type *Ty);

  /// NewGlobalType - Make the type visible globally using the given name.
  ///
  void NewGlobalType(const std::string &Name, DIE *Type);
  
  /// NewGlobalEntity - Make the entity visible globally using the given name.
  ///
  void NewGlobalEntity(const std::string &Name, DIE *Entity);

private:

  /// NewType - Create a new type DIE.
  ///
  DIE *NewType(DIE *Unit, TypeDesc *TyDesc);
  
  /// NewCompileUnit - Create new compile unit DIE.
  ///
  DIE *NewCompileUnit(CompileUnitDesc *CompileUnit);
  
  /// NewGlobalVariable - Make a new global variable DIE.
  ///
  DIE *NewGlobalVariable(GlobalVariableDesc *GVD);

  /// NewSubprogram - Add a new subprogram DIE.
  ///
  DIE *NewSubprogram(SubprogramDesc *SPD);

  /// EmitInitial - Emit initial Dwarf declarations.
  ///
  void EmitInitial() const;
  
  /// EmitDIE - Recusively Emits a debug information entry.
  ///
  void EmitDIE(DIE *Die) const;
  
  /// SizeAndOffsetDie - Compute the size and offset of a DIE.
  ///
  unsigned SizeAndOffsetDie(DIE *Die, unsigned Offset);

  /// SizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void SizeAndOffsets();
  
  /// EmitDebugInfo - Emit the debug info section.
  ///
  void EmitDebugInfo() const;
  
  /// EmitAbbreviations - Emit the abbreviation section.
  ///
  void EmitAbbreviations() const;
  
  /// EmitDebugLines - Emit source line information.
  ///
  void EmitDebugLines() const;

  /// EmitDebugFrame - Emit info into a debug frame section.
  ///
  void EmitDebugFrame();
  
  /// EmitDebugPubNames - Emit info into a debug pubnames section.
  ///
  void EmitDebugPubNames();
  
  /// EmitDebugPubTypes - Emit info into a debug pubtypes section.
  ///
  void EmitDebugPubTypes();
  
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
  void ConstructGlobalDIEs(Module &M);

  /// ConstructSubprogramDIEs - Create DIEs for each of the externally visible
  /// subprograms.
  void ConstructSubprogramDIEs(Module &M);

  /// ShouldEmitDwarf - Returns true if Dwarf declarations should be made.
  /// When called it also checks to see if debug info is newly available.  if
  /// so the initial Dwarf headers are emitted.
  bool ShouldEmitDwarf();

public:
  
  DwarfWriter(std::ostream &OS, AsmPrinter *A);
  virtual ~DwarfWriter();
  
  /// SetDebugInfo - Set DebugInfo when it's known that pass manager has
  /// created it.  Set by the target AsmPrinter.
  void SetDebugInfo(MachineDebugInfo *DI) { DebugInfo = DI; }

  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  
  /// BeginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void BeginModule(Module &M);
  
  /// EndModule - Emit all Dwarf sections that should come after the content.
  ///
  void EndModule(Module &M);
  
  /// BeginFunction - Gather pre-function debug information.
  ///
  void BeginFunction(MachineFunction &MF);
  
  /// EndFunction - Gather and emit post-function debug information.
  ///
  void EndFunction(MachineFunction &MF);
};

} // end llvm namespace

#endif
