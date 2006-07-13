//===-- llvm/CodeGen/DwarfWriter.cpp - Dwarf Framework ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DwarfWriter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"

#include <iostream>

using namespace llvm;
using namespace llvm::dwarf;

static cl::opt<bool>
DwarfVerbose("dwarf-verbose", cl::Hidden,
                                cl::desc("Add comments to Dwarf directives."));

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
//
class DIE;

//===----------------------------------------------------------------------===//
// CompileUnit - This dwarf writer support class manages information associate
// with a source file.
class CompileUnit {
private:
  CompileUnitDesc *Desc;                // Compile unit debug descriptor.
  unsigned ID;                          // File ID for source.
  DIE *Die;                             // Compile unit debug information entry.
  std::map<std::string, DIE *> Globals; // A map of globally visible named
                                        // entities for this unit.
  std::map<DebugInfoDesc *, DIE *> DescToDieMap;
                                        // Tracks the mapping of unit level
                                        // debug informaton descriptors to debug
                                        // information entries.

public:
  CompileUnit(CompileUnitDesc *CUD, unsigned I, DIE *D)
  : Desc(CUD)
  , ID(I)
  , Die(D)
  , Globals()
  , DescToDieMap()
  {}
  
  ~CompileUnit();
  
  // Accessors.
  CompileUnitDesc *getDesc() const { return Desc; }
  unsigned getID()           const { return ID; }
  DIE* getDie()              const { return Die; }
  std::map<std::string, DIE *> &getGlobals() { return Globals; }
  
  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const;
  
  /// AddGlobal - Add a new global entity to the compile unit.
  ///
  void AddGlobal(const std::string &Name, DIE *Die);
  
  /// getDieMapSlotFor - Returns the debug information entry map slot for the
  /// specified debug descriptor.
  DIE *&getDieMapSlotFor(DebugInfoDesc *DD) {
    return DescToDieMap[DD];
  }
};

//===----------------------------------------------------------------------===//
// DIEAbbrevData - Dwarf abbreviation data, describes the one attribute of a
// Dwarf abbreviation.
class DIEAbbrevData {
private:
  unsigned Attribute;                 // Dwarf attribute code.
  unsigned Form;                      // Dwarf form code.
  
public:
  DIEAbbrevData(unsigned A, unsigned F)
  : Attribute(A)
  , Form(F)
  {}
  
  // Accessors.
  unsigned getAttribute() const { return Attribute; }
  unsigned getForm()      const { return Form; }
  
  /// operator== - Used by DIEAbbrev to locate entry.
  ///
  bool operator==(const DIEAbbrevData &DAD) const {
    return Attribute == DAD.Attribute && Form == DAD.Form;
  }

  /// operator!= - Used by DIEAbbrev to locate entry.
  ///
  bool operator!=(const DIEAbbrevData &DAD) const {
    return Attribute != DAD.Attribute || Form != DAD.Form;
  }
  
  /// operator< - Used by DIEAbbrev to locate entry.
  ///
  bool operator<(const DIEAbbrevData &DAD) const {
    return Attribute < DAD.Attribute ||
          (Attribute == DAD.Attribute && Form < DAD.Form);
  }
};

//===----------------------------------------------------------------------===//
// DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
// information object.
class DIEAbbrev {
private:
  unsigned Tag;                       // Dwarf tag code.
  unsigned ChildrenFlag;              // Dwarf children flag.
  std::vector<DIEAbbrevData> Data;    // Raw data bytes for abbreviation.

public:

  DIEAbbrev(unsigned T, unsigned C)
  : Tag(T)
  , ChildrenFlag(C)
  , Data()
  {}
  ~DIEAbbrev() {}
  
  // Accessors.
  unsigned getTag()                           const { return Tag; }
  unsigned getChildrenFlag()                  const { return ChildrenFlag; }
  const std::vector<DIEAbbrevData> &getData() const { return Data; }
  void setChildrenFlag(unsigned CF)                 { ChildrenFlag = CF; }

  /// operator== - Used by UniqueVector to locate entry.
  ///
  bool operator==(const DIEAbbrev &DA) const;

  /// operator< - Used by UniqueVector to locate entry.
  ///
  bool operator<(const DIEAbbrev &DA) const;

  /// AddAttribute - Adds another set of attribute information to the
  /// abbreviation.
  void AddAttribute(unsigned Attribute, unsigned Form) {
    Data.push_back(DIEAbbrevData(Attribute, Form));
  }
  
  /// AddFirstAttribute - Adds a set of attribute information to the front
  /// of the abbreviation.
  void AddFirstAttribute(unsigned Attribute, unsigned Form) {
    Data.insert(Data.begin(), DIEAbbrevData(Attribute, Form));
  }
  
  /// Emit - Print the abbreviation using the specified Dwarf writer.
  ///
  void Emit(const DwarfWriter &DW) const; 
      
#ifndef NDEBUG
  void print(std::ostream &O);
  void dump();
#endif
};

//===----------------------------------------------------------------------===//
// DIEValue - A debug information entry value.
//
class DIEValue {
public:
  enum {
    isInteger,
    isString,
    isLabel,
    isAsIsLabel,
    isDelta,
    isEntry,
    isBlock
  };
  
  unsigned Type;                      // Type of the value
  
  DIEValue(unsigned T) : Type(T) {}
  virtual ~DIEValue() {}
  
  // Implement isa/cast/dyncast.
  static bool classof(const DIEValue *) { return true; }
  
  /// EmitValue - Emit value via the Dwarf writer.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const = 0;
  
  /// SizeOf - Return the size of a value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const = 0;
};

//===----------------------------------------------------------------------===//
// DWInteger - An integer value DIE.
// 
class DIEInteger : public DIEValue {
private:
  uint64_t Integer;
  
public:
  DIEInteger(uint64_t I) : DIEValue(isInteger), Integer(I) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEInteger *) { return true; }
  static bool classof(const DIEValue *I)  { return I->Type == isInteger; }
  
  /// BestForm - Choose the best form for integer.
  ///
  unsigned BestForm(bool IsSigned);

  /// EmitValue - Emit integer of appropriate size.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of integer value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
};

//===----------------------------------------------------------------------===//
// DIEString - A string value DIE.
// 
struct DIEString : public DIEValue {
  const std::string String;
  
  DIEString(const std::string &S) : DIEValue(isString), String(S) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEString *) { return true; }
  static bool classof(const DIEValue *S) { return S->Type == isString; }
  
  /// EmitValue - Emit string value.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of string value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
};

//===----------------------------------------------------------------------===//
// DIEDwarfLabel - A Dwarf internal label expression DIE.
//
struct DIEDwarfLabel : public DIEValue {
  const DWLabel Label;
  
  DIEDwarfLabel(const DWLabel &L) : DIEValue(isLabel), Label(L) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEDwarfLabel *)  { return true; }
  static bool classof(const DIEValue *L) { return L->Type == isLabel; }
  
  /// EmitValue - Emit label value.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of label value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
};


//===----------------------------------------------------------------------===//
// DIEObjectLabel - A label to an object in code or data.
//
struct DIEObjectLabel : public DIEValue {
  const std::string Label;
  
  DIEObjectLabel(const std::string &L) : DIEValue(isAsIsLabel), Label(L) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEObjectLabel *) { return true; }
  static bool classof(const DIEValue *L)    { return L->Type == isAsIsLabel; }
  
  /// EmitValue - Emit label value.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of label value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
};

//===----------------------------------------------------------------------===//
// DIEDelta - A simple label difference DIE.
// 
struct DIEDelta : public DIEValue {
  const DWLabel LabelHi;
  const DWLabel LabelLo;
  
  DIEDelta(const DWLabel &Hi, const DWLabel &Lo)
  : DIEValue(isDelta), LabelHi(Hi), LabelLo(Lo) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEDelta *)  { return true; }
  static bool classof(const DIEValue *D) { return D->Type == isDelta; }
  
  /// EmitValue - Emit delta value.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of delta value in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
};

//===----------------------------------------------------------------------===//
// DIEntry - A pointer to a debug information entry.
// 
struct DIEntry : public DIEValue {
  DIE *Entry;
  
  DIEntry(DIE *E) : DIEValue(isEntry), Entry(E) {}

  // Implement isa/cast/dyncast.
  static bool classof(const DIEntry *)   { return true; }
  static bool classof(const DIEValue *E) { return E->Type == isEntry; }
  
  /// EmitValue - Emit debug information entry offset.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of debug information entry in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
};

//===----------------------------------------------------------------------===//
// DIEBlock - A block of values.  Primarily used for location expressions.
//
struct DIEBlock : public DIEValue {
  unsigned Size;                        // Size in bytes excluding size header.
  std::vector<unsigned> Forms;          // Data forms.
  std::vector<DIEValue *> Values;       // Block values.
  
  DIEBlock()
  : DIEValue(isBlock)
  , Size(0)
  , Forms()
  , Values()
  {}
  ~DIEBlock();

  // Implement isa/cast/dyncast.
  static bool classof(const DIEBlock *)  { return true; }
  static bool classof(const DIEValue *E) { return E->Type == isBlock; }
  
  /// ComputeSize - calculate the size of the block.
  ///
  unsigned ComputeSize(DwarfWriter &DW);
  
  /// BestForm - Choose the best form for data.
  ///
  unsigned BestForm();

  /// EmitValue - Emit block data.
  ///
  virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
  
  /// SizeOf - Determine size of block data in bytes.
  ///
  virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;

  /// AddUInt - Add an unsigned integer value.
  ///
  void AddUInt(unsigned Form, uint64_t Integer);

  /// AddSInt - Add an signed integer value.
  ///
  void AddSInt(unsigned Form, int64_t Integer);
      
  /// AddString - Add a std::string value.
  ///
  void AddString(unsigned Form, const std::string &String);
      
  /// AddLabel - Add a Dwarf label value.
  ///
  void AddLabel(unsigned Form, const DWLabel &Label);
      
  /// AddObjectLabel - Add a non-Dwarf label value.
  ///
  void AddObjectLabel(unsigned Form, const std::string &Label);
      
  /// AddDelta - Add a label delta value.
  ///
  void AddDelta(unsigned Form, const DWLabel &Hi, const DWLabel &Lo);
      
  /// AddDIEntry - Add a DIE value.
  ///
  void AddDIEntry(unsigned Form, DIE *Entry);

};

//===----------------------------------------------------------------------===//
// DIE - A structured debug information entry.  Has an abbreviation which
// describes it's organization.
class DIE {
private:
  DIEAbbrev *Abbrev;                    // Temporary buffer for abbreviation.
  unsigned AbbrevID;                    // Decribing abbreviation ID.
  unsigned Offset;                      // Offset in debug info section.
  unsigned Size;                        // Size of instance + children.
  std::vector<DIE *> Children;          // Children DIEs.
  std::vector<DIEValue *> Values;       // Attributes values.
  
public:
  DIE(unsigned Tag);
  ~DIE();
  
  // Accessors.
  unsigned   getAbbrevID()                   const { return AbbrevID; }
  unsigned   getOffset()                     const { return Offset; }
  unsigned   getSize()                       const { return Size; }
  const std::vector<DIE *> &getChildren()    const { return Children; }
  const std::vector<DIEValue *> &getValues() const { return Values; }
  void setOffset(unsigned O)                 { Offset = O; }
  void setSize(unsigned S)                   { Size = S; }
  
  /// SiblingOffset - Return the offset of the debug information entry's
  /// sibling.
  unsigned SiblingOffset() const { return Offset + Size; }
  
  /// AddSiblingOffset - Add a sibling offset field to the front of the DIE.
  ///
  void AddSiblingOffset();

  /// AddUInt - Add an unsigned integer attribute data and value.
  ///
  void AddUInt(unsigned Attribute, unsigned Form, uint64_t Integer);

  /// AddSInt - Add an signed integer attribute data and value.
  ///
  void AddSInt(unsigned Attribute, unsigned Form, int64_t Integer);
      
  /// AddString - Add a std::string attribute data and value.
  ///
  void AddString(unsigned Attribute, unsigned Form,
                 const std::string &String);
      
  /// AddLabel - Add a Dwarf label attribute data and value.
  ///
  void AddLabel(unsigned Attribute, unsigned Form, const DWLabel &Label);
      
  /// AddObjectLabel - Add a non-Dwarf label attribute data and value.
  ///
  void AddObjectLabel(unsigned Attribute, unsigned Form,
                      const std::string &Label);
      
  /// AddDelta - Add a label delta attribute data and value.
  ///
  void AddDelta(unsigned Attribute, unsigned Form,
                const DWLabel &Hi, const DWLabel &Lo);
      
  /// AddDIEntry - Add a DIE attribute data and value.
  ///
  void AddDIEntry(unsigned Attribute, unsigned Form, DIE *Entry);

  /// AddBlock - Add block data.
  ///
  void AddBlock(unsigned Attribute, unsigned Form, DIEBlock *Block);

  /// Complete - Indicate that all attributes have been added and
  /// ready to get an abbreviation ID.
  ///
  void Complete(DwarfWriter &DW);
  
  /// AddChild - Add a child to the DIE.
  void AddChild(DIE *Child);
};

} // End of namespace llvm

//===----------------------------------------------------------------------===//

CompileUnit::~CompileUnit() {
  delete Die;
}

/// hasContent - Return true if this compile unit has something to write out.
///
bool CompileUnit::hasContent() const {
  return !Die->getChildren().empty();
}

/// AddGlobal - Add a new global entity to the compile unit.
///
void CompileUnit::AddGlobal(const std::string &Name, DIE *Die) {
  Globals[Name] = Die;
}

//===----------------------------------------------------------------------===//

/// operator== - Used by UniqueVector to locate entry.
///
bool DIEAbbrev::operator==(const DIEAbbrev &DA) const {
  if (Tag != DA.Tag) return false;
  if (ChildrenFlag != DA.ChildrenFlag) return false;
  if (Data.size() != DA.Data.size()) return false;
  
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    if (Data[i] != DA.Data[i]) return false;
  }
  
  return true;
}

/// operator< - Used by UniqueVector to locate entry.
///
bool DIEAbbrev::operator<(const DIEAbbrev &DA) const {
  if (Tag != DA.Tag) return Tag < DA.Tag;
  if (ChildrenFlag != DA.ChildrenFlag) return ChildrenFlag < DA.ChildrenFlag;
  if (Data.size() != DA.Data.size()) return Data.size() < DA.Data.size();
  
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    if (Data[i] != DA.Data[i]) return Data[i] < DA.Data[i];
  }
  
  return false;
}
    
/// Emit - Print the abbreviation using the specified Dwarf writer.
///
void DIEAbbrev::Emit(const DwarfWriter &DW) const {
  // Emit its Dwarf tag type.
  DW.EmitULEB128Bytes(Tag);
  DW.EOL(TagString(Tag));
  
  // Emit whether it has children DIEs.
  DW.EmitULEB128Bytes(ChildrenFlag);
  DW.EOL(ChildrenString(ChildrenFlag));
  
  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    const DIEAbbrevData &AttrData = Data[i];
    
    // Emit attribute type.
    DW.EmitULEB128Bytes(AttrData.getAttribute());
    DW.EOL(AttributeString(AttrData.getAttribute()));
    
    // Emit form type.
    DW.EmitULEB128Bytes(AttrData.getForm());
    DW.EOL(FormEncodingString(AttrData.getForm()));
  }

  // Mark end of abbreviation.
  DW.EmitULEB128Bytes(0); DW.EOL("EOM(1)");
  DW.EmitULEB128Bytes(0); DW.EOL("EOM(2)");
}

#ifndef NDEBUG
  void DIEAbbrev::print(std::ostream &O) {
    O << "Abbreviation @"
      << std::hex << (intptr_t)this << std::dec
      << "  "
      << TagString(Tag)
      << " "
      << ChildrenString(ChildrenFlag)
      << "\n";
    
    for (unsigned i = 0, N = Data.size(); i < N; ++i) {
      O << "  "
        << AttributeString(Data[i].getAttribute())
        << "  "
        << FormEncodingString(Data[i].getForm())
        << "\n";
    }
  }
  void DIEAbbrev::dump() { print(std::cerr); }
#endif

//===----------------------------------------------------------------------===//

/// BestForm - Choose the best form for integer.
///
unsigned DIEInteger::BestForm(bool IsSigned) {
  if (IsSigned) {
    if ((char)Integer == (signed)Integer)   return DW_FORM_data1;
    if ((short)Integer == (signed)Integer)  return DW_FORM_data2;
    if ((int)Integer == (signed)Integer)    return DW_FORM_data4;
  } else {
    if ((unsigned char)Integer == Integer)  return DW_FORM_data1;
    if ((unsigned short)Integer == Integer) return DW_FORM_data2;
    if ((unsigned int)Integer == Integer)   return DW_FORM_data4;
  }
  return DW_FORM_data8;
}
    
/// EmitValue - Emit integer of appropriate size.
///
void DIEInteger::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  switch (Form) {
  case DW_FORM_flag:  // Fall thru
  case DW_FORM_ref1:  // Fall thru
  case DW_FORM_data1: DW.EmitInt8(Integer);         break;
  case DW_FORM_ref2:  // Fall thru
  case DW_FORM_data2: DW.EmitInt16(Integer);        break;
  case DW_FORM_ref4:  // Fall thru
  case DW_FORM_data4: DW.EmitInt32(Integer);        break;
  case DW_FORM_ref8:  // Fall thru
  case DW_FORM_data8: DW.EmitInt64(Integer);        break;
  case DW_FORM_udata: DW.EmitULEB128Bytes(Integer); break;
  case DW_FORM_sdata: DW.EmitSLEB128Bytes(Integer); break;
  default: assert(0 && "DIE Value form not supported yet"); break;
  }
}

/// SizeOf - Determine size of integer value in bytes.
///
unsigned DIEInteger::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  switch (Form) {
  case DW_FORM_flag:  // Fall thru
  case DW_FORM_ref1:  // Fall thru
  case DW_FORM_data1: return sizeof(int8_t);
  case DW_FORM_ref2:  // Fall thru
  case DW_FORM_data2: return sizeof(int16_t);
  case DW_FORM_ref4:  // Fall thru
  case DW_FORM_data4: return sizeof(int32_t);
  case DW_FORM_ref8:  // Fall thru
  case DW_FORM_data8: return sizeof(int64_t);
  case DW_FORM_udata: return DW.SizeULEB128(Integer);
  case DW_FORM_sdata: return DW.SizeSLEB128(Integer);
  default: assert(0 && "DIE Value form not supported yet"); break;
  }
  return 0;
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit string value.
///
void DIEString::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitString(String);
}

/// SizeOf - Determine size of string value in bytes.
///
unsigned DIEString::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return String.size() + sizeof(char); // sizeof('\0');
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIEDwarfLabel::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitReference(Label);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIEDwarfLabel::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return DW.getAddressSize();
}
    
//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIEObjectLabel::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitReference(Label);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIEObjectLabel::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return DW.getAddressSize();
}
    
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIEDelta::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitDifference(LabelHi, LabelLo);
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEDelta::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return DW.getAddressSize();
}

//===----------------------------------------------------------------------===//
/// EmitValue - Emit debug information entry offset.
///
void DIEntry::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitInt32(Entry->getOffset());
}

/// SizeOf - Determine size of debug information entry value in bytes.
///
unsigned DIEntry::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return sizeof(int32_t);
}
    
//===----------------------------------------------------------------------===//

DIEBlock::~DIEBlock() {
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    delete Values[i];
  }
}

/// ComputeSize - calculate the size of the block.
///
unsigned DIEBlock::ComputeSize(DwarfWriter &DW) {
  Size = 0;
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    Size += Values[i]->SizeOf(DW, Forms[i]);
  }
  return Size;
}

/// BestForm - Choose the best form for data.
///
unsigned DIEBlock::BestForm() {
  if ((unsigned char)Size == Size)  return DW_FORM_block1;
  if ((unsigned short)Size == Size) return DW_FORM_block2;
  if ((unsigned int)Size == Size)   return DW_FORM_block4;
  return DW_FORM_block;
}

/// EmitValue - Emit block data.
///
void DIEBlock::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  switch (Form) {
  case DW_FORM_block1: DW.EmitInt8(Size);         break;
  case DW_FORM_block2: DW.EmitInt16(Size);        break;
  case DW_FORM_block4: DW.EmitInt32(Size);        break;
  case DW_FORM_block:  DW.EmitULEB128Bytes(Size); break;
  default: assert(0 && "Improper form for block"); break;
  }
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    DW.EOL("");
    Values[i]->EmitValue(DW, Forms[i]);
  }
}

/// SizeOf - Determine size of block data in bytes.
///
unsigned DIEBlock::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  switch (Form) {
  case DW_FORM_block1: return Size + sizeof(int8_t);
  case DW_FORM_block2: return Size + sizeof(int16_t);
  case DW_FORM_block4: return Size + sizeof(int32_t);
  case DW_FORM_block: return Size + DW.SizeULEB128(Size);
  default: assert(0 && "Improper form for block"); break;
  }
  return 0;
}

/// AddUInt - Add an unsigned integer value.
///
void DIEBlock::AddUInt(unsigned Form, uint64_t Integer) {
  DIEInteger *DI = new DIEInteger(Integer);
  Values.push_back(DI);
  if (Form == 0) Form = DI->BestForm(false);
  Forms.push_back(Form);
}

/// AddSInt - Add an signed integer value.
///
void DIEBlock::AddSInt(unsigned Form, int64_t Integer) {
  DIEInteger *DI = new DIEInteger(Integer);
  Values.push_back(DI);
  if (Form == 0) Form = DI->BestForm(true);
  Forms.push_back(Form);
}
    
/// AddString - Add a std::string value.
///
void DIEBlock::AddString(unsigned Form, const std::string &String) {
  Values.push_back(new DIEString(String));
  Forms.push_back(Form);
}
    
/// AddLabel - Add a Dwarf label value.
///
void DIEBlock::AddLabel(unsigned Form, const DWLabel &Label) {
  Values.push_back(new DIEDwarfLabel(Label));
  Forms.push_back(Form);
}
    
/// AddObjectLabel - Add a non-Dwarf label value.
///
void DIEBlock::AddObjectLabel(unsigned Form, const std::string &Label) {
  Values.push_back(new DIEObjectLabel(Label));
  Forms.push_back(Form);
}
    
/// AddDelta - Add a label delta value.
///
void DIEBlock::AddDelta(unsigned Form, const DWLabel &Hi, const DWLabel &Lo) {
  Values.push_back(new DIEDelta(Hi, Lo));
  Forms.push_back(Form);
}
    
/// AddDIEntry - Add a DIE value.
///
void DIEBlock::AddDIEntry(unsigned Form, DIE *Entry) {
  Values.push_back(new DIEntry(Entry));
  Forms.push_back(Form);
}

//===----------------------------------------------------------------------===//

DIE::DIE(unsigned Tag)
: Abbrev(new DIEAbbrev(Tag, DW_CHILDREN_no))
, AbbrevID(0)
, Offset(0)
, Size(0)
, Children()
, Values()
{}

DIE::~DIE() {
  if (Abbrev) delete Abbrev;
  
  for (unsigned i = 0, N = Children.size(); i < N; ++i) {
    delete Children[i];
  }

  for (unsigned j = 0, M = Values.size(); j < M; ++j) {
    delete Values[j];
  }
}
    
/// AddSiblingOffset - Add a sibling offset field to the front of the DIE.
///
void DIE::AddSiblingOffset() {
  DIEInteger *DI = new DIEInteger(0);
  Values.insert(Values.begin(), DI);
  Abbrev->AddFirstAttribute(DW_AT_sibling, DW_FORM_ref4);
}

/// AddUInt - Add an unsigned integer attribute data and value.
///
void DIE::AddUInt(unsigned Attribute, unsigned Form, uint64_t Integer) {
  DIEInteger *DI = new DIEInteger(Integer);
  Values.push_back(DI);
  if (!Form) Form = DI->BestForm(false);
  Abbrev->AddAttribute(Attribute, Form);
}
    
/// AddSInt - Add an signed integer attribute data and value.
///
void DIE::AddSInt(unsigned Attribute, unsigned Form, int64_t Integer) {
  DIEInteger *DI = new DIEInteger(Integer);
  Values.push_back(DI);
  if (!Form) Form = DI->BestForm(true);
  Abbrev->AddAttribute(Attribute, Form);
}
    
/// AddString - Add a std::string attribute data and value.
///
void DIE::AddString(unsigned Attribute, unsigned Form,
                    const std::string &String) {
  Values.push_back(new DIEString(String));
  Abbrev->AddAttribute(Attribute, Form);
}
    
/// AddLabel - Add a Dwarf label attribute data and value.
///
void DIE::AddLabel(unsigned Attribute, unsigned Form,
                   const DWLabel &Label) {
  Values.push_back(new DIEDwarfLabel(Label));
  Abbrev->AddAttribute(Attribute, Form);
}
    
/// AddObjectLabel - Add an non-Dwarf label attribute data and value.
///
void DIE::AddObjectLabel(unsigned Attribute, unsigned Form,
                         const std::string &Label) {
  Values.push_back(new DIEObjectLabel(Label));
  Abbrev->AddAttribute(Attribute, Form);
}
    
/// AddDelta - Add a label delta attribute data and value.
///
void DIE::AddDelta(unsigned Attribute, unsigned Form,
                   const DWLabel &Hi, const DWLabel &Lo) {
  Values.push_back(new DIEDelta(Hi, Lo));
  Abbrev->AddAttribute(Attribute, Form);
}
    
/// AddDIEntry - Add a DIE attribute data and value.
///
void DIE::AddDIEntry(unsigned Attribute, unsigned Form, DIE *Entry) {
  Values.push_back(new DIEntry(Entry));
  Abbrev->AddAttribute(Attribute, Form);
}

/// AddBlock - Add block data.
///
void DIE::AddBlock(unsigned Attribute, unsigned Form, DIEBlock *Block) {
  assert(Block->Size && "Block size has not been computed");
  Values.push_back(Block);
  if (!Form) Form = Block->BestForm();
  Abbrev->AddAttribute(Attribute, Form);
}

/// Complete - Indicate that all attributes have been added and ready to get an
/// abbreviation ID.
void DIE::Complete(DwarfWriter &DW) {
  AbbrevID = DW.NewAbbreviation(Abbrev);
  delete Abbrev;
  Abbrev = NULL;
}

/// AddChild - Add a child to the DIE.
///
void DIE::AddChild(DIE *Child) {
  assert(Abbrev && "Adding children without an abbreviation");
  Abbrev->setChildrenFlag(DW_CHILDREN_yes);
  Children.push_back(Child);
}

//===----------------------------------------------------------------------===//

/// DWContext

//===----------------------------------------------------------------------===//

/// PrintHex - Print a value as a hexidecimal value.
///
void DwarfWriter::PrintHex(int Value) const { 
  O << "0x" << std::hex << Value << std::dec;
}

/// EOL - Print a newline character to asm stream.  If a comment is present
/// then it will be printed first.  Comments should not contain '\n'.
void DwarfWriter::EOL(const std::string &Comment) const {
  if (DwarfVerbose && !Comment.empty()) {
    O << "\t"
      << Asm->CommentString
      << " "
      << Comment;
  }
  O << "\n";
}

/// EmitAlign - Print a align directive.
///
void DwarfWriter::EmitAlign(unsigned Alignment) const {
  O << Asm->AlignDirective << Alignment << "\n";
}

/// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
/// unsigned leb128 value.
void DwarfWriter::EmitULEB128Bytes(unsigned Value) const {
  if (hasLEB128) {
    O << "\t.uleb128\t"
      << Value;
  } else {
    O << Asm->Data8bitsDirective;
    PrintULEB128(Value);
  }
}

/// EmitSLEB128Bytes - Emit an assembler byte data directive to compose a
/// signed leb128 value.
void DwarfWriter::EmitSLEB128Bytes(int Value) const {
  if (hasLEB128) {
    O << "\t.sleb128\t"
      << Value;
  } else {
    O << Asm->Data8bitsDirective;
    PrintSLEB128(Value);
  }
}

/// PrintULEB128 - Print a series of hexidecimal values (separated by commas)
/// representing an unsigned leb128 value.
void DwarfWriter::PrintULEB128(unsigned Value) const {
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    if (Value) Byte |= 0x80;
    PrintHex(Byte);
    if (Value) O << ", ";
  } while (Value);
}

/// SizeULEB128 - Compute the number of bytes required for an unsigned leb128
/// value.
unsigned DwarfWriter::SizeULEB128(unsigned Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

/// PrintSLEB128 - Print a series of hexidecimal values (separated by commas)
/// representing a signed leb128 value.
void DwarfWriter::PrintSLEB128(int Value) const {
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    PrintHex(Byte);
    if (IsMore) O << ", ";
  } while (IsMore);
}

/// SizeSLEB128 - Compute the number of bytes required for a signed leb128
/// value.
unsigned DwarfWriter::SizeSLEB128(int Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}

/// EmitInt8 - Emit a byte directive and value.
///
void DwarfWriter::EmitInt8(int Value) const {
  O << Asm->Data8bitsDirective;
  PrintHex(Value & 0xFF);
}

/// EmitInt16 - Emit a short directive and value.
///
void DwarfWriter::EmitInt16(int Value) const {
  O << Asm->Data16bitsDirective;
  PrintHex(Value & 0xFFFF);
}

/// EmitInt32 - Emit a long directive and value.
///
void DwarfWriter::EmitInt32(int Value) const {
  O << Asm->Data32bitsDirective;
  PrintHex(Value);
}

/// EmitInt64 - Emit a long long directive and value.
///
void DwarfWriter::EmitInt64(uint64_t Value) const {
  if (Asm->Data64bitsDirective) {
    O << Asm->Data64bitsDirective << "0x" << std::hex << Value << std::dec;
  } else {
    if (TD->isBigEndian()) {
      EmitInt32(unsigned(Value >> 32)); O << "\n";
      EmitInt32(unsigned(Value));
    } else {
      EmitInt32(unsigned(Value)); O << "\n";
      EmitInt32(unsigned(Value >> 32));
    }
  }
}

/// EmitString - Emit a string with quotes and a null terminator.
/// Special characters are emitted properly. (Eg. '\t')
void DwarfWriter::EmitString(const std::string &String) const {
  O << Asm->AsciiDirective
    << "\"";
  for (unsigned i = 0, N = String.size(); i < N; ++i) {
    unsigned char C = String[i];
    
    if (!isascii(C) || iscntrl(C)) {
      switch(C) {
      case '\b': O << "\\b"; break;
      case '\f': O << "\\f"; break;
      case '\n': O << "\\n"; break;
      case '\r': O << "\\r"; break;
      case '\t': O << "\\t"; break;
      default:
        O << '\\';
        O << char('0' + (C >> 6));
        O << char('0' + (C >> 3));
        O << char('0' + (C >> 0));
        break;
      }
    } else if (C == '\"') {
      O << "\\\"";
    } else if (C == '\'') {
      O << "\\\'";
    } else {
     O << C;
    }
  }
  O << "\\0\"";
}

/// PrintLabelName - Print label name in form used by Dwarf writer.
///
void DwarfWriter::PrintLabelName(const char *Tag, unsigned Number) const {
  O << Asm->PrivateGlobalPrefix
    << "debug_"
    << Tag;
  if (Number) O << Number;
}

/// EmitLabel - Emit location label for internal use by Dwarf.
///
void DwarfWriter::EmitLabel(const char *Tag, unsigned Number) const {
  PrintLabelName(Tag, Number);
  O << ":\n";
}

/// EmitReference - Emit a reference to a label.
///
void DwarfWriter::EmitReference(const char *Tag, unsigned Number) const {
  if (AddressSize == 4)
    O << Asm->Data32bitsDirective;
  else
    O << Asm->Data64bitsDirective;
    
  PrintLabelName(Tag, Number);
}
void DwarfWriter::EmitReference(const std::string &Name) const {
  if (AddressSize == 4)
    O << Asm->Data32bitsDirective;
  else
    O << Asm->Data64bitsDirective;
    
  O << Name;
}

/// EmitDifference - Emit an label difference as sizeof(pointer) value.  Some
/// assemblers do not accept absolute expressions with data directives, so there 
/// is an option (needsSet) to use an intermediary 'set' expression.
void DwarfWriter::EmitDifference(const char *TagHi, unsigned NumberHi,
                                 const char *TagLo, unsigned NumberLo) const {
  if (needsSet) {
    static unsigned SetCounter = 0;
    
    O << "\t.set\t";
    PrintLabelName("set", SetCounter);
    O << ",";
    PrintLabelName(TagHi, NumberHi);
    O << "-";
    PrintLabelName(TagLo, NumberLo);
    O << "\n";
    
    if (AddressSize == sizeof(int32_t))
      O << Asm->Data32bitsDirective;
    else
      O << Asm->Data64bitsDirective;
      
    PrintLabelName("set", SetCounter);
    
    ++SetCounter;
  } else {
    if (AddressSize == sizeof(int32_t))
      O << Asm->Data32bitsDirective;
    else
      O << Asm->Data64bitsDirective;
      
    PrintLabelName(TagHi, NumberHi);
    O << "-";
    PrintLabelName(TagLo, NumberLo);
  }
}

/// NewAbbreviation - Add the abbreviation to the Abbreviation vector.
///  
unsigned DwarfWriter::NewAbbreviation(DIEAbbrev *Abbrev) {
  return Abbreviations.insert(*Abbrev);
}

/// NewString - Add a string to the constant pool and returns a label.
///
DWLabel DwarfWriter::NewString(const std::string &String) {
  unsigned StringID = StringPool.insert(String);
  return DWLabel("string", StringID);
}

/// AddSourceLine - Add location information to specified debug information
/// entry.
void DwarfWriter::AddSourceLine(DIE *Die, CompileUnitDesc *File, unsigned Line){
  if (File && Line) {
    CompileUnit *FileUnit = FindCompileUnit(File);
    unsigned FileID = FileUnit->getID();
    Die->AddUInt(DW_AT_decl_file, 0, FileID);
    Die->AddUInt(DW_AT_decl_line, 0, Line);
  }
}

/// AddAddress - Add an address attribute to a die based on the location
/// provided.
void DwarfWriter::AddAddress(DIE *Die, unsigned Attribute,
                             const MachineLocation &Location) {
  DIEBlock *Block = new DIEBlock();
  if (Location.isRegister()) {
    Block->AddUInt(DW_FORM_data1,
                   DW_OP_reg0 + RI->getDwarfRegNum(Location.getRegister()));
  } else {
    Block->AddUInt(DW_FORM_data1,
                   DW_OP_breg0 + RI->getDwarfRegNum(Location.getRegister()));
    Block->AddUInt(DW_FORM_sdata, Location.getOffset());
  }
  Block->ComputeSize(*this);
  Die->AddBlock(Attribute, 0, Block);
}

/// getDieMapSlotFor - Returns the debug information entry map slot for the
/// specified debug descriptor.
DIE *&DwarfWriter::getDieMapSlotFor(DebugInfoDesc *DD) {
  return DescToDieMap[DD];
}
                                 
/// NewType - Create a new type DIE.
///
DIE *DwarfWriter::NewType(DIE *Context, TypeDesc *TyDesc, CompileUnit *Unit) {
  if (!TyDesc) {
    // FIXME - Hack for missing types
    DIE *Die = new DIE(DW_TAG_base_type);
    Die->AddUInt(DW_AT_byte_size, 0, 4);
    Die->AddUInt(DW_AT_encoding, DW_FORM_data1, DW_ATE_signed);
    Unit->getDie()->AddChild(Die);
    return Die;
  }
  
  // FIXME - Should handle other contexts that compile units.

  // Check for pre-existence.
  DIE *&Slot = Unit->getDieMapSlotFor(TyDesc);
  if (Slot) return Slot;

  // Get core information.
  const std::string &Name = TyDesc->getName();
  uint64_t Size = TyDesc->getSize() >> 3;
  
  DIE *Ty = NULL;
  
  if (BasicTypeDesc *BasicTy = dyn_cast<BasicTypeDesc>(TyDesc)) {
    // Fundamental types like int, float, bool
    Slot = Ty = new DIE(DW_TAG_base_type);
    unsigned Encoding = BasicTy->getEncoding();
    Ty->AddUInt(DW_AT_encoding,  DW_FORM_data1, Encoding);
  } else if (DerivedTypeDesc *DerivedTy = dyn_cast<DerivedTypeDesc>(TyDesc)) {
    // Create specific DIE.
    Slot = Ty = new DIE(DerivedTy->getTag());
    
    // Map to main type, void will not have a type.
    if (TypeDesc *FromTy = DerivedTy->getFromType()) {
      Ty->AddDIEntry(DW_AT_type, DW_FORM_ref4,
                     NewType(Context, FromTy, Unit));
    }
  } else if (CompositeTypeDesc *CompTy = dyn_cast<CompositeTypeDesc>(TyDesc)) {
    // Fetch tag
    unsigned Tag = CompTy->getTag();
    
    // Create specific DIE.
    Slot = Ty = Tag == DW_TAG_vector_type ? new DIE(DW_TAG_array_type) :
                                            new DIE(Tag);
    
    std::vector<DebugInfoDesc *> &Elements = CompTy->getElements();
    
    switch (Tag) {
    case DW_TAG_vector_type: Ty->AddUInt(DW_AT_GNU_vector, DW_FORM_flag, 1);
      // Fall thru
    case DW_TAG_array_type: {
      // Add element type.
      if (TypeDesc *FromTy = CompTy->getFromType()) {
        Ty->AddDIEntry(DW_AT_type, DW_FORM_ref4,
                       NewType(Context, FromTy, Unit));
      }
      
      // Don't emit size attribute.
      Size = 0;
      
      // Construct an anonymous type for index type.
      DIE *IndexTy = new DIE(DW_TAG_base_type);
      IndexTy->AddUInt(DW_AT_byte_size, 0, 4);
      IndexTy->AddUInt(DW_AT_encoding, DW_FORM_data1, DW_ATE_signed);
      // Add to context.
      Context->AddChild(IndexTy);
    
      // Add subranges to array type.
      for(unsigned i = 0, N = Elements.size(); i < N; ++i) {
        SubrangeDesc *SRD = cast<SubrangeDesc>(Elements[i]);
        int64_t Lo = SRD->getLo();
        int64_t Hi = SRD->getHi();
        DIE *Subrange = new DIE(DW_TAG_subrange_type);
        
        // If a range is available.
        if (Lo != Hi) {
          Subrange->AddDIEntry(DW_AT_type, DW_FORM_ref4, IndexTy);
          // Only add low if non-zero.
          if (Lo) Subrange->AddSInt(DW_AT_lower_bound, 0, Lo);
          Subrange->AddSInt(DW_AT_upper_bound, 0, Hi);
        }
        Ty->AddChild(Subrange);
      }
      
      break;
    }
    case DW_TAG_structure_type:
    case DW_TAG_union_type: {
      // FIXME - this is just the basics.
      // Add elements to structure type.
      for(unsigned i = 0, N = Elements.size(); i < N; ++i) {
        DerivedTypeDesc *MemberDesc = cast<DerivedTypeDesc>(Elements[i]);
        
        // Extract the basic information.
        const std::string &Name = MemberDesc->getName();
        TypeDesc *MemTy = MemberDesc->getFromType();
        uint64_t Size = MemberDesc->getSize();
        uint64_t Align = MemberDesc->getAlign();
        uint64_t Offset = MemberDesc->getOffset();
   
        // Construct member debug information entry.
        DIE *Member = new DIE(DW_TAG_member);
        
        // Add name if not "".
        if (!Name.empty()) Member->AddString(DW_AT_name, DW_FORM_string, Name);
        // Add location if available.
        AddSourceLine(Member, MemberDesc->getFile(), MemberDesc->getLine());
        
        // Most of the time the field info is the same as the members.
        uint64_t FieldSize = Size;
        uint64_t FieldAlign = Align;
        uint64_t FieldOffset = Offset;
        
        if (TypeDesc *FromTy = MemberDesc->getFromType()) {
          Member->AddDIEntry(DW_AT_type, DW_FORM_ref4,
                             NewType(Context, FromTy, Unit));
          FieldSize = FromTy->getSize();
          FieldAlign = FromTy->getSize();
        }
        
        // Unless we have a bit field.
        if (FieldSize != Size) {
          // Construct the alignment mask.
          uint64_t AlignMask = ~(FieldAlign - 1);
          // Determine the high bit + 1 of the declared size.
          uint64_t HiMark = (Offset + FieldSize) & AlignMask;
          // Work backwards to determine the base offset of the field.
          FieldOffset = HiMark - FieldSize;
          // Now normalize offset to the field.
          Offset -= FieldOffset;
          
          // Maybe we need to work from the other end.
          if (TD->isLittleEndian()) Offset = FieldSize - (Offset + Size);
          
          Member->AddUInt(DW_AT_byte_size, 0, FieldSize >> 3);
          Member->AddUInt(DW_AT_bit_size, 0, Size);
          Member->AddUInt(DW_AT_bit_offset, 0, Offset);
        }
        
        // Add computation for offset.
        DIEBlock *Block = new DIEBlock();
        Block->AddUInt(DW_FORM_data1, DW_OP_plus_uconst);
        Block->AddUInt(DW_FORM_udata, FieldOffset >> 3);
        Block->ComputeSize(*this);
        Member->AddBlock(DW_AT_data_member_location, 0, Block);

        if (MemberDesc->isProtected()) {
          Member->AddUInt(DW_AT_accessibility, 0, DW_ACCESS_protected);
        } else if (MemberDesc->isPrivate()) {
          Member->AddUInt(DW_AT_accessibility, 0, DW_ACCESS_private);
        }
        
        Ty->AddChild(Member);
      }
      break;
    }
    case DW_TAG_enumeration_type: {
      // Add enumerators to enumeration type.
      for(unsigned i = 0, N = Elements.size(); i < N; ++i) {
        EnumeratorDesc *ED = cast<EnumeratorDesc>(Elements[i]);
        const std::string &Name = ED->getName();
        int64_t Value = ED->getValue();
        DIE *Enumerator = new DIE(DW_TAG_enumerator);
        Enumerator->AddString(DW_AT_name, DW_FORM_string, Name);
        Enumerator->AddSInt(DW_AT_const_value, DW_FORM_sdata, Value);
        Ty->AddChild(Enumerator);
      }

      break;
    }
    case DW_TAG_subroutine_type: {
      // Add prototype flag.
      Ty->AddUInt(DW_AT_prototyped, DW_FORM_flag, 1);
      // Add return type.
      Ty->AddDIEntry(DW_AT_type, DW_FORM_ref4,
                     NewType(Context, dyn_cast<TypeDesc>(Elements[0]), Unit));
      
      // Add arguments.
      for(unsigned i = 1, N = Elements.size(); i < N; ++i) {
        DIE *Arg = new DIE(DW_TAG_formal_parameter);
        Arg->AddDIEntry(DW_AT_type, DW_FORM_ref4,
                        NewType(Context, cast<TypeDesc>(Elements[i]), Unit));
        Ty->AddChild(Arg);
      }
      
      break;
    }
    default: break;
    }
  }
  
  assert(Ty && "Type not supported yet");
 
  // Add size if non-zero (derived types don't have a size.)
  if (Size) Ty->AddUInt(DW_AT_byte_size, 0, Size);
  // Add name if not anonymous or intermediate type.
  if (!Name.empty()) Ty->AddString(DW_AT_name, DW_FORM_string, Name);
  // Add source line info if available.
  AddSourceLine(Ty, TyDesc->getFile(), TyDesc->getLine());

  // Add to context owner.
  Context->AddChild(Ty);
  
  return Slot;
}

/// NewCompileUnit - Create new compile unit and it's debug information entry.
///
CompileUnit *DwarfWriter::NewCompileUnit(CompileUnitDesc *UnitDesc,
                                         unsigned ID) {
  // Construct debug information entry.
  DIE *Die = new DIE(DW_TAG_compile_unit);
  Die->AddDelta (DW_AT_stmt_list, DW_FORM_data4,  DWLabel("line", 0),
                                                  DWLabel("section_line", 0));
//  Die->AddLabel (DW_AT_high_pc,   DW_FORM_addr,   DWLabel("text_end", 0));
//  Die->AddLabel (DW_AT_low_pc,    DW_FORM_addr,   DWLabel("text_begin", 0));
  Die->AddString(DW_AT_producer,  DW_FORM_string, UnitDesc->getProducer());
  Die->AddUInt  (DW_AT_language,  DW_FORM_data1,  UnitDesc->getLanguage());
  Die->AddString(DW_AT_name,      DW_FORM_string, UnitDesc->getFileName());
  Die->AddString(DW_AT_comp_dir,  DW_FORM_string, UnitDesc->getDirectory());
  
  // Add debug information entry to descriptor map.
  DIE *&Slot = getDieMapSlotFor(UnitDesc);
  Slot = Die;
  
  // Construct compile unit.
  CompileUnit *Unit = new CompileUnit(UnitDesc, ID, Die);
  
  // Add Unit to compile unit map.
  DescToUnitMap[UnitDesc] = Unit;
  
  return Unit;
}

/// FindCompileUnit - Get the compile unit for the given descriptor.
///
CompileUnit *DwarfWriter::FindCompileUnit(CompileUnitDesc *UnitDesc) {
  CompileUnit *Unit = DescToUnitMap[UnitDesc];
  assert(Unit && "Missing compile unit.");
  return Unit;
}

/// NewGlobalVariable - Add a new global variable DIE.
///
DIE *DwarfWriter::NewGlobalVariable(GlobalVariableDesc *GVD) {
  // Get the compile unit context.
  CompileUnitDesc *UnitDesc = static_cast<CompileUnitDesc *>(GVD->getContext());
  CompileUnit *Unit = FindCompileUnit(UnitDesc);

  // Check for pre-existence.
  DIE *&Slot = Unit->getDieMapSlotFor(GVD);
  if (Slot) return Slot;
  
  // Get the global variable itself.
  GlobalVariable *GV = GVD->getGlobalVariable();
  // Generate the mangled name.
  std::string MangledName = Asm->Mang->getValueName(GV);

  // Gather the details (simplify add attribute code.)
  const std::string &Name = GVD->getName();
  
  // Get the global's type.
  DIE *Type = NewType(Unit->getDie(), GVD->getType(), Unit); 

  // Create the globale variable DIE.
  DIE *VariableDie = new DIE(DW_TAG_variable);
  VariableDie->AddString     (DW_AT_name,      DW_FORM_string, Name);
  VariableDie->AddDIEntry    (DW_AT_type,      DW_FORM_ref4,   Type);
  VariableDie->AddUInt       (DW_AT_external,  DW_FORM_flag,   1);
  
  // Add source line info if available.
  AddSourceLine(VariableDie, UnitDesc, GVD->getLine());

  // Add address.
  DIEBlock *Block = new DIEBlock();
  Block->AddUInt(DW_FORM_data1, DW_OP_addr);
  Block->AddObjectLabel(DW_FORM_udata, MangledName);
  Block->ComputeSize(*this);
  VariableDie->AddBlock(DW_AT_location,  0, Block);
  
  // Add to map.
  Slot = VariableDie;
 
  // Add to context owner.
  Unit->getDie()->AddChild(VariableDie);
  
  // Expose as global.
  // FIXME - need to check external flag.
  Unit->AddGlobal(Name, VariableDie);
  
  return VariableDie;
}

/// NewSubprogram - Add a new subprogram DIE.
///
DIE *DwarfWriter::NewSubprogram(SubprogramDesc *SPD) {
  // Get the compile unit context.
  CompileUnitDesc *UnitDesc = static_cast<CompileUnitDesc *>(SPD->getContext());
  CompileUnit *Unit = FindCompileUnit(UnitDesc);

  // Check for pre-existence.
  DIE *&Slot = Unit->getDieMapSlotFor(SPD);
  if (Slot) return Slot;
  
  // Gather the details (simplify add attribute code.)
  const std::string &Name = SPD->getName();
  DIE *Type = NewType(Unit->getDie(), SPD->getType(), Unit); 
  unsigned IsExternal = SPD->isStatic() ? 0 : 1;
                                    
  DIE *SubprogramDie = new DIE(DW_TAG_subprogram);
  SubprogramDie->AddString     (DW_AT_name,      DW_FORM_string, Name);
  if (Type) {
    SubprogramDie->AddDIEntry    (DW_AT_type,      DW_FORM_ref4,   Type);
  }
  SubprogramDie->AddUInt       (DW_AT_external,    DW_FORM_flag,   IsExternal);
  SubprogramDie->AddUInt       (DW_AT_prototyped,  DW_FORM_flag,   1);
  
  // Add source line info if available.
  AddSourceLine(SubprogramDie, UnitDesc, SPD->getLine());

  // Add to map.
  Slot = SubprogramDie;
 
  // Add to context owner.
  Unit->getDie()->AddChild(SubprogramDie);
  
  // Expose as global.
  Unit->AddGlobal(Name, SubprogramDie);
  
  return SubprogramDie;
}

/// NewScopeVariable - Create a new scope variable.
///
DIE *DwarfWriter::NewScopeVariable(DebugVariable *DV, CompileUnit *Unit) {
  // Get the descriptor.
  VariableDesc *VD = DV->getDesc();

  // Translate tag to proper Dwarf tag.  The result variable is dropped for now.
  unsigned Tag;
  switch (VD->getTag()) {
  case DW_TAG_return_variable:  return NULL;
  case DW_TAG_arg_variable:     Tag = DW_TAG_formal_parameter; break;
  case DW_TAG_auto_variable:    // fall thru
  default:                      Tag = DW_TAG_variable; break;
  }

  // Define variable debug information entry.
  DIE *VariableDie = new DIE(Tag);
  VariableDie->AddString(DW_AT_name, DW_FORM_string, VD->getName());

  // Add source line info if available.
  AddSourceLine(VariableDie, VD->getFile(), VD->getLine());
  
  // Add variable type.
  DIE *Type = NewType(Unit->getDie(), VD->getType(), Unit); 
  VariableDie->AddDIEntry(DW_AT_type, DW_FORM_ref4, Type);
  
  // Add variable address.
  MachineLocation Location;
  RI->getLocation(*MF, DV->getFrameIndex(), Location);
  AddAddress(VariableDie, DW_AT_location, Location);
  
  return VariableDie;
}

/// ConstructScope - Construct the components of a scope.
///
void DwarfWriter::ConstructScope(DebugScope *ParentScope,
                                 DIE *ParentDie, CompileUnit *Unit) {
  // Add variables to scope.
  std::vector<DebugVariable *> &Variables = ParentScope->getVariables();
  for (unsigned i = 0, N = Variables.size(); i < N; ++i) {
    DIE *VariableDie = NewScopeVariable(Variables[i], Unit);
    if (VariableDie) ParentDie->AddChild(VariableDie);
  }
  
  // Add nested scopes.
  std::vector<DebugScope *> &Scopes = ParentScope->getScopes();
  for (unsigned j = 0, M = Scopes.size(); j < M; ++j) {
    // Define the Scope debug information entry.
    DebugScope *Scope = Scopes[j];
    // FIXME - Ignore inlined functions for the time being.
    if (!Scope->getParent()) continue;
    
    DIE *ScopeDie = new DIE(DW_TAG_lexical_block);
    
    // Add the scope bounds.
    if (unsigned StartID = Scope->getStartLabelID()) {
      ScopeDie->AddLabel(DW_AT_low_pc, DW_FORM_addr,
                         DWLabel("loc", StartID));
    } else {
      ScopeDie->AddLabel(DW_AT_low_pc, DW_FORM_addr,
                         DWLabel("func_begin", SubprogramCount));
    }
    if (unsigned EndID = Scope->getEndLabelID()) {
      ScopeDie->AddLabel(DW_AT_high_pc, DW_FORM_addr,
                         DWLabel("loc", EndID));
    } else {
      ScopeDie->AddLabel(DW_AT_high_pc, DW_FORM_addr,
                         DWLabel("func_end", SubprogramCount));
    }
                       
    // Add the scope contents.
    ConstructScope(Scope, ScopeDie, Unit);
    ParentDie->AddChild(ScopeDie);
  }
}

/// ConstructRootScope - Construct the scope for the subprogram.
///
void DwarfWriter::ConstructRootScope(DebugScope *RootScope) {
  // Exit if there is no root scope.
  if (!RootScope) return;
  
  // Get the subprogram debug information entry. 
  SubprogramDesc *SPD = cast<SubprogramDesc>(RootScope->getDesc());
  
  // Get the compile unit context.
  CompileUnitDesc *UnitDesc = static_cast<CompileUnitDesc *>(SPD->getContext());
  CompileUnit *Unit = FindCompileUnit(UnitDesc);
  
  // Get the subprogram die.
  DIE *SPDie = Unit->getDieMapSlotFor(SPD);
  assert(SPDie && "Missing subprogram descriptor");
  
  // Add the function bounds.
  SPDie->AddLabel(DW_AT_low_pc, DW_FORM_addr,
                  DWLabel("func_begin", SubprogramCount));
  SPDie->AddLabel(DW_AT_high_pc, DW_FORM_addr,
                  DWLabel("func_end", SubprogramCount));
  MachineLocation Location(RI->getFrameRegister(*MF));
  AddAddress(SPDie, DW_AT_frame_base, Location);
                  
  ConstructScope(RootScope, SPDie, Unit);
}

/// EmitInitial - Emit initial Dwarf declarations.  This is necessary for cc
/// tools to recognize the object file contains Dwarf information.
///
void DwarfWriter::EmitInitial() {
  // Check to see if we already emitted intial headers.
  if (didInitial) return;
  didInitial = true;
  
  // Dwarf sections base addresses.
  Asm->SwitchToDataSection(DwarfFrameSection, 0);
  EmitLabel("section_frame", 0);
  Asm->SwitchToDataSection(DwarfInfoSection, 0);
  EmitLabel("section_info", 0);
  EmitLabel("info", 0);
  Asm->SwitchToDataSection(DwarfAbbrevSection, 0);
  EmitLabel("section_abbrev", 0);
  EmitLabel("abbrev", 0);
  Asm->SwitchToDataSection(DwarfARangesSection, 0);
  EmitLabel("section_aranges", 0);
  Asm->SwitchToDataSection(DwarfMacInfoSection, 0);
  EmitLabel("section_macinfo", 0);
  Asm->SwitchToDataSection(DwarfLineSection, 0);
  EmitLabel("section_line", 0);
  EmitLabel("line", 0);
  Asm->SwitchToDataSection(DwarfLocSection, 0);
  EmitLabel("section_loc", 0);
  Asm->SwitchToDataSection(DwarfPubNamesSection, 0);
  EmitLabel("section_pubnames", 0);
  Asm->SwitchToDataSection(DwarfStrSection, 0);
  EmitLabel("section_str", 0);
  Asm->SwitchToDataSection(DwarfRangesSection, 0);
  EmitLabel("section_ranges", 0);

  Asm->SwitchToTextSection(TextSection, 0);
  EmitLabel("text_begin", 0);
  Asm->SwitchToDataSection(DataSection, 0);
  EmitLabel("data_begin", 0);

  // Emit common frame information.
  EmitInitialDebugFrame();
}

/// EmitDIE - Recusively Emits a debug information entry.
///
void DwarfWriter::EmitDIE(DIE *Die) const {
  // Get the abbreviation for this DIE.
  unsigned AbbrevID = Die->getAbbrevID();
  const DIEAbbrev &Abbrev = Abbreviations[AbbrevID];
  
  O << "\n";

  // Emit the code (index) for the abbreviation.
  EmitULEB128Bytes(AbbrevID);
  EOL(std::string("Abbrev [" +
      utostr(AbbrevID) +
      "] 0x" + utohexstr(Die->getOffset()) +
      ":0x" + utohexstr(Die->getSize()) + " " +
      TagString(Abbrev.getTag())));
  
  const std::vector<DIEValue *> &Values = Die->getValues();
  const std::vector<DIEAbbrevData> &AbbrevData = Abbrev.getData();
  
  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    unsigned Attr = AbbrevData[i].getAttribute();
    unsigned Form = AbbrevData[i].getForm();
    assert(Form && "Too many attributes for DIE (check abbreviation)");
    
    switch (Attr) {
    case DW_AT_sibling: {
      EmitInt32(Die->SiblingOffset());
      break;
    }
    default: {
      // Emit an attribute using the defined form.
      Values[i]->EmitValue(*this, Form);
      break;
    }
    }
    
    EOL(AttributeString(Attr));
  }
  
  // Emit the DIE children if any.
  if (Abbrev.getChildrenFlag() == DW_CHILDREN_yes) {
    const std::vector<DIE *> &Children = Die->getChildren();
    
    for (unsigned j = 0, M = Children.size(); j < M; ++j) {
      EmitDIE(Children[j]);
    }
    
    EmitInt8(0); EOL("End Of Children Mark");
  }
}

/// SizeAndOffsetDie - Compute the size and offset of a DIE.
///
unsigned DwarfWriter::SizeAndOffsetDie(DIE *Die, unsigned Offset, bool Last) {
  // Get the children.
  const std::vector<DIE *> &Children = Die->getChildren();
  
  // If not last sibling and has children then add sibling offset attribute.
  if (!Last && !Children.empty()) Die->AddSiblingOffset();

  // Record the abbreviation.
  Die->Complete(*this);
  
  // Get the abbreviation for this DIE.
  unsigned AbbrevID = Die->getAbbrevID();
  const DIEAbbrev &Abbrev = Abbreviations[AbbrevID];

  // Set DIE offset
  Die->setOffset(Offset);
  
  // Start the size with the size of abbreviation code.
  Offset += SizeULEB128(AbbrevID);
  
  const std::vector<DIEValue *> &Values = Die->getValues();
  const std::vector<DIEAbbrevData> &AbbrevData = Abbrev.getData();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    // Size attribute value.
    Offset += Values[i]->SizeOf(*this, AbbrevData[i].getForm());
  }
  
  // Emit the DIE children if any.
  if (!Children.empty()) {
    assert(Abbrev.getChildrenFlag() == DW_CHILDREN_yes &&
           "Children flag not set");
    
    for (unsigned j = 0, M = Children.size(); j < M; ++j) {
      Offset = SizeAndOffsetDie(Children[j], Offset, (j + 1) == M);
    }
    
    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die->setSize(Offset - Die->getOffset());
  return Offset;
}

/// SizeAndOffsets - Compute the size and offset of all the DIEs.
///
void DwarfWriter::SizeAndOffsets() {
  
  // Process each compile unit.
  for (unsigned i = 0, N = CompileUnits.size(); i < N; ++i) {
    CompileUnit *Unit = CompileUnits[i];
    if (Unit->hasContent()) {
      // Compute size of compile unit header
      unsigned Offset = sizeof(int32_t) + // Length of Compilation Unit Info
                        sizeof(int16_t) + // DWARF version number
                        sizeof(int32_t) + // Offset Into Abbrev. Section
                        sizeof(int8_t);   // Pointer Size (in bytes)
      SizeAndOffsetDie(Unit->getDie(), Offset, (i + 1) == N);
    }
  }
}

/// EmitFrameMoves - Emit frame instructions to describe the layout of the
/// frame.
void DwarfWriter::EmitFrameMoves(const char *BaseLabel, unsigned BaseLabelID,
                                 std::vector<MachineMove *> &Moves) {
  for (unsigned i = 0, N = Moves.size(); i < N; ++i) {
    MachineMove *Move = Moves[i];
    unsigned LabelID = Move->getLabelID();
    const MachineLocation &Dst = Move->getDestination();
    const MachineLocation &Src = Move->getSource();
    
    // Advance row if new location.
    if (BaseLabel && LabelID && BaseLabelID != LabelID) {
      EmitULEB128Bytes(DW_CFA_advance_loc4);
      EOL("DW_CFA_advance_loc4");
      EmitDifference("loc", LabelID, BaseLabel, BaseLabelID);
      EOL("");
      
      BaseLabelID = LabelID;
      BaseLabel = "loc";
    }
    
    // If advancing cfa.
    if (Dst.isRegister() && Dst.getRegister() == MachineLocation::VirtualFP) {
      if (!Src.isRegister()) {
        if (Src.getRegister() == MachineLocation::VirtualFP) {
          EmitULEB128Bytes(DW_CFA_def_cfa_offset);
          EOL("DW_CFA_def_cfa_offset");
        } else {
          EmitULEB128Bytes(DW_CFA_def_cfa);
          EOL("DW_CFA_def_cfa");
          
          EmitULEB128Bytes(RI->getDwarfRegNum(Src.getRegister()));
          EOL("Register");
        }
        
        int stackGrowth =
            Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
              TargetFrameInfo::StackGrowsUp ?
                AddressSize : -AddressSize;
        
        EmitULEB128Bytes(Src.getOffset() / stackGrowth);
        EOL("Offset");
      } else {
      }
    } else {
    }
  }
}

/// EmitDebugInfo - Emit the debug info section.
///
void DwarfWriter::EmitDebugInfo() const {
  // Start debug info section.
  Asm->SwitchToDataSection(DwarfInfoSection, 0);
  
  // Process each compile unit.
  for (unsigned i = 0, N = CompileUnits.size(); i < N; ++i) {
    CompileUnit *Unit = CompileUnits[i];
    
    if (Unit->hasContent()) {
      DIE *Die = Unit->getDie();
      // Emit the compile units header.
      EmitLabel("info_begin", Unit->getID());
      // Emit size of content not including length itself
      unsigned ContentSize = Die->getSize() +
                             sizeof(int16_t) + // DWARF version number
                             sizeof(int32_t) + // Offset Into Abbrev. Section
                             sizeof(int8_t);   // Pointer Size (in bytes)
                             
      EmitInt32(ContentSize);  EOL("Length of Compilation Unit Info");
      EmitInt16(DWARF_VERSION); EOL("DWARF version number");
      EmitDifference("abbrev_begin", 0, "section_abbrev", 0);
      EOL("Offset Into Abbrev. Section");
      EmitInt8(AddressSize); EOL("Address Size (in bytes)");
    
      EmitDIE(Die);
      EmitLabel("info_end", Unit->getID());
    }
    
    O << "\n";
  }
}

/// EmitAbbreviations - Emit the abbreviation section.
///
void DwarfWriter::EmitAbbreviations() const {
  // Check to see if it is worth the effort.
  if (!Abbreviations.empty()) {
    // Start the debug abbrev section.
    Asm->SwitchToDataSection(DwarfAbbrevSection, 0);
    
    EmitLabel("abbrev_begin", 0);
    
    // For each abbrevation.
    for (unsigned AbbrevID = 1, NAID = Abbreviations.size();
                  AbbrevID <= NAID; ++AbbrevID) {
      // Get abbreviation data
      const DIEAbbrev &Abbrev = Abbreviations[AbbrevID];
      
      // Emit the abbrevations code (base 1 index.)
      EmitULEB128Bytes(AbbrevID); EOL("Abbreviation Code");
      
      // Emit the abbreviations data.
      Abbrev.Emit(*this);
  
      O << "\n";
    }
    
    EmitLabel("abbrev_end", 0);
  
    O << "\n";
  }
}

/// EmitDebugLines - Emit source line information.
///
void DwarfWriter::EmitDebugLines() const {
  // Minimum line delta, thus ranging from -10..(255-10).
  const int MinLineDelta = -(DW_LNS_fixed_advance_pc + 1);
  // Maximum line delta, thus ranging from -10..(255-10).
  const int MaxLineDelta = 255 + MinLineDelta;

  // Start the dwarf line section.
  Asm->SwitchToDataSection(DwarfLineSection, 0);
  
  // Construct the section header.
  
  EmitDifference("line_end", 0, "line_begin", 0);
  EOL("Length of Source Line Info");
  EmitLabel("line_begin", 0);
  
  EmitInt16(DWARF_VERSION); EOL("DWARF version number");
  
  EmitDifference("line_prolog_end", 0, "line_prolog_begin", 0);
  EOL("Prolog Length");
  EmitLabel("line_prolog_begin", 0);
  
  EmitInt8(1); EOL("Minimum Instruction Length");

  EmitInt8(1); EOL("Default is_stmt_start flag");

  EmitInt8(MinLineDelta);  EOL("Line Base Value (Special Opcodes)");
  
  EmitInt8(MaxLineDelta); EOL("Line Range Value (Special Opcodes)");

  EmitInt8(-MinLineDelta); EOL("Special Opcode Base");
  
  // Line number standard opcode encodings argument count
  EmitInt8(0); EOL("DW_LNS_copy arg count");
  EmitInt8(1); EOL("DW_LNS_advance_pc arg count");
  EmitInt8(1); EOL("DW_LNS_advance_line arg count");
  EmitInt8(1); EOL("DW_LNS_set_file arg count");
  EmitInt8(1); EOL("DW_LNS_set_column arg count");
  EmitInt8(0); EOL("DW_LNS_negate_stmt arg count");
  EmitInt8(0); EOL("DW_LNS_set_basic_block arg count");
  EmitInt8(0); EOL("DW_LNS_const_add_pc arg count");
  EmitInt8(1); EOL("DW_LNS_fixed_advance_pc arg count");

  const UniqueVector<std::string> &Directories = DebugInfo->getDirectories();
  const UniqueVector<SourceFileInfo> &SourceFiles = DebugInfo->getSourceFiles();

  // Emit directories.
  for (unsigned DirectoryID = 1, NDID = Directories.size();
                DirectoryID <= NDID; ++DirectoryID) {
    EmitString(Directories[DirectoryID]); EOL("Directory");
  }
  EmitInt8(0); EOL("End of directories");
  
  // Emit files.
  for (unsigned SourceID = 1, NSID = SourceFiles.size();
               SourceID <= NSID; ++SourceID) {
    const SourceFileInfo &SourceFile = SourceFiles[SourceID];
    EmitString(SourceFile.getName()); EOL("Source");
    EmitULEB128Bytes(SourceFile.getDirectoryID());  EOL("Directory #");
    EmitULEB128Bytes(0);  EOL("Mod date");
    EmitULEB128Bytes(0);  EOL("File size");
  }
  EmitInt8(0); EOL("End of files");
  
  EmitLabel("line_prolog_end", 0);
  
  // A sequence for each text section.
  for (unsigned j = 0, M = SectionSourceLines.size(); j < M; ++j) {
    // Isolate current sections line info.
    const std::vector<SourceLineInfo *> &LineInfos = SectionSourceLines[j];
    
    if (DwarfVerbose) {
      O << "\t"
        << Asm->CommentString << " "
        << "Section "
        << SectionMap[j + 1].c_str() << "\n";
    }

    // Dwarf assumes we start with first line of first source file.
    unsigned Source = 1;
    unsigned Line = 1;
    
    // Construct rows of the address, source, line, column matrix.
    for (unsigned i = 0, N = LineInfos.size(); i < N; ++i) {
      SourceLineInfo *LineInfo = LineInfos[i];
      
      if (DwarfVerbose) {
        unsigned SourceID = LineInfo->getSourceID();
        const SourceFileInfo &SourceFile = SourceFiles[SourceID];
        unsigned DirectoryID = SourceFile.getDirectoryID();
        O << "\t"
          << Asm->CommentString << " "
          << Directories[DirectoryID]
          << SourceFile.getName() << ":"
          << LineInfo->getLine() << "\n"; 
      }

      // Define the line address.
      EmitInt8(0); EOL("Extended Op");
      EmitInt8(4 + 1); EOL("Op size");
      EmitInt8(DW_LNE_set_address); EOL("DW_LNE_set_address");
      EmitReference("loc",  LineInfo->getLabelID()); EOL("Location label");
      
      // If change of source, then switch to the new source.
      if (Source != LineInfo->getSourceID()) {
        Source = LineInfo->getSourceID();
        EmitInt8(DW_LNS_set_file); EOL("DW_LNS_set_file");
        EmitULEB128Bytes(Source); EOL("New Source");
      }
      
      // If change of line.
      if (Line != LineInfo->getLine()) {
        // Determine offset.
        int Offset = LineInfo->getLine() - Line;
        int Delta = Offset - MinLineDelta;
        
        // Update line.
        Line = LineInfo->getLine();
        
        // If delta is small enough and in range...
        if (Delta >= 0 && Delta < (MaxLineDelta - 1)) {
          // ... then use fast opcode.
          EmitInt8(Delta - MinLineDelta); EOL("Line Delta");
        } else {
          // ... otherwise use long hand.
          EmitInt8(DW_LNS_advance_line); EOL("DW_LNS_advance_line");
          EmitSLEB128Bytes(Offset); EOL("Line Offset");
          EmitInt8(DW_LNS_copy); EOL("DW_LNS_copy");
        }
      } else {
        // Copy the previous row (different address or source)
        EmitInt8(DW_LNS_copy); EOL("DW_LNS_copy");
      }
    }

    // Define last address of section.
    EmitInt8(0); EOL("Extended Op");
    EmitInt8(4 + 1); EOL("Op size");
    EmitInt8(DW_LNE_set_address); EOL("DW_LNE_set_address");
    EmitReference("section_end", j + 1); EOL("Section end label");

    // Mark end of matrix.
    EmitInt8(0); EOL("DW_LNE_end_sequence");
    EmitULEB128Bytes(1);  O << "\n";
    EmitInt8(1); O << "\n";
  }
  
  EmitLabel("line_end", 0);
  
  O << "\n";
}
  
/// EmitInitialDebugFrame - Emit common frame info into a debug frame section.
///
void DwarfWriter::EmitInitialDebugFrame() {
  int stackGrowth =
      Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
        TargetFrameInfo::StackGrowsUp ?
      AddressSize : -AddressSize;

  // Start the dwarf frame section.
  Asm->SwitchToDataSection(DwarfFrameSection, 0);

  EmitLabel("frame_common", 0);
  EmitDifference("frame_common_end", 0,
                 "frame_common_begin", 0);
  EOL("Length of Common Information Entry");

  EmitLabel("frame_common_begin", 0);
  EmitInt32(DW_CIE_ID); EOL("CIE Identifier Tag");
  EmitInt8(DW_CIE_VERSION); EOL("CIE Version");
  EmitString("");  EOL("CIE Augmentation");
  EmitULEB128Bytes(1); EOL("CIE Code Alignment Factor");
  EmitSLEB128Bytes(stackGrowth); EOL("CIE Data Alignment Factor");   
  EmitInt8(RI->getDwarfRegNum(RI->getRARegister())); EOL("CIE RA Column");
  
  std::vector<MachineMove *> Moves;
  RI->getInitialFrameState(Moves);
  EmitFrameMoves(NULL, 0, Moves);
  for (unsigned i = 0, N = Moves.size(); i < N; ++i) delete Moves[i];

  EmitAlign(2);
  EmitLabel("frame_common_end", 0);
  
  O << "\n";
}

/// EmitFunctionDebugFrame - Emit per function frame info into a debug frame
/// section.
void DwarfWriter::EmitFunctionDebugFrame() {
  // Start the dwarf frame section.
  Asm->SwitchToDataSection(DwarfFrameSection, 0);
  
  EmitDifference("frame_end", SubprogramCount,
                 "frame_begin", SubprogramCount);
  EOL("Length of Frame Information Entry");
  
  EmitLabel("frame_begin", SubprogramCount);
  
  EmitDifference("frame_common", 0, "section_frame", 0);
  EOL("FDE CIE offset");

  EmitReference("func_begin", SubprogramCount); EOL("FDE initial location");
  EmitDifference("func_end", SubprogramCount,
                 "func_begin", SubprogramCount);
  EOL("FDE address range");
  
  std::vector<MachineMove *> &Moves = DebugInfo->getFrameMoves();
  
  EmitFrameMoves("func_begin", SubprogramCount, Moves);
  
  EmitAlign(2);
  EmitLabel("frame_end", SubprogramCount);

  O << "\n";
}

/// EmitDebugPubNames - Emit visible names into a debug pubnames section.
///
void DwarfWriter::EmitDebugPubNames() {
  // Start the dwarf pubnames section.
  Asm->SwitchToDataSection(DwarfPubNamesSection, 0);
    
  // Process each compile unit.
  for (unsigned i = 0, N = CompileUnits.size(); i < N; ++i) {
    CompileUnit *Unit = CompileUnits[i];
    
    if (Unit->hasContent()) {
      EmitDifference("pubnames_end", Unit->getID(),
                     "pubnames_begin", Unit->getID());
      EOL("Length of Public Names Info");
      
      EmitLabel("pubnames_begin", Unit->getID());
      
      EmitInt16(DWARF_VERSION); EOL("DWARF Version");
      
      EmitDifference("info_begin", Unit->getID(), "section_info", 0);
      EOL("Offset of Compilation Unit Info");

      EmitDifference("info_end", Unit->getID(), "info_begin", Unit->getID());
      EOL("Compilation Unit Length");
      
      std::map<std::string, DIE *> &Globals = Unit->getGlobals();
      
      for (std::map<std::string, DIE *>::iterator GI = Globals.begin(),
                                                  GE = Globals.end();
           GI != GE; ++GI) {
        const std::string &Name = GI->first;
        DIE * Entity = GI->second;
        
        EmitInt32(Entity->getOffset()); EOL("DIE offset");
        EmitString(Name); EOL("External Name");
      }
    
      EmitInt32(0); EOL("End Mark");
      EmitLabel("pubnames_end", Unit->getID());
    
      O << "\n";
    }
  }
}

/// EmitDebugStr - Emit visible names into a debug str section.
///
void DwarfWriter::EmitDebugStr() {
  // Check to see if it is worth the effort.
  if (!StringPool.empty()) {
    // Start the dwarf str section.
    Asm->SwitchToDataSection(DwarfStrSection, 0);
    
    // For each of strings in the string pool.
    for (unsigned StringID = 1, N = StringPool.size();
         StringID <= N; ++StringID) {
      // Emit a label for reference from debug information entries.
      EmitLabel("string", StringID);
      // Emit the string itself.
      const std::string &String = StringPool[StringID];
      EmitString(String); O << "\n";
    }
  
    O << "\n";
  }
}

/// EmitDebugLoc - Emit visible names into a debug loc section.
///
void DwarfWriter::EmitDebugLoc() {
  // Start the dwarf loc section.
  Asm->SwitchToDataSection(DwarfLocSection, 0);
  
  O << "\n";
}

/// EmitDebugARanges - Emit visible names into a debug aranges section.
///
void DwarfWriter::EmitDebugARanges() {
  // Start the dwarf aranges section.
  Asm->SwitchToDataSection(DwarfARangesSection, 0);
  
  // FIXME - Mock up
#if 0
  // Process each compile unit.
  for (unsigned i = 0, N = CompileUnits.size(); i < N; ++i) {
    CompileUnit *Unit = CompileUnits[i];
    
    if (Unit->hasContent()) {
      // Don't include size of length
      EmitInt32(0x1c); EOL("Length of Address Ranges Info");
      
      EmitInt16(DWARF_VERSION); EOL("Dwarf Version");
      
      EmitReference("info_begin", Unit->getID());
      EOL("Offset of Compilation Unit Info");

      EmitInt8(AddressSize); EOL("Size of Address");

      EmitInt8(0); EOL("Size of Segment Descriptor");

      EmitInt16(0);  EOL("Pad (1)");
      EmitInt16(0);  EOL("Pad (2)");

      // Range 1
      EmitReference("text_begin", 0); EOL("Address");
      EmitDifference("text_end", 0, "text_begin", 0); EOL("Length");

      EmitInt32(0); EOL("EOM (1)");
      EmitInt32(0); EOL("EOM (2)");
      
      O << "\n";
    }
  }
#endif
}

/// EmitDebugRanges - Emit visible names into a debug ranges section.
///
void DwarfWriter::EmitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->SwitchToDataSection(DwarfRangesSection, 0);
  
  O << "\n";
}

/// EmitDebugMacInfo - Emit visible names into a debug macinfo section.
///
void DwarfWriter::EmitDebugMacInfo() {
  // Start the dwarf macinfo section.
  Asm->SwitchToDataSection(DwarfMacInfoSection, 0);
  
  O << "\n";
}

/// ConstructCompileUnitDIEs - Create a compile unit DIE for each source and
/// header file.
void DwarfWriter::ConstructCompileUnitDIEs() {
  const UniqueVector<CompileUnitDesc *> CUW = DebugInfo->getCompileUnits();
  
  for (unsigned i = 1, N = CUW.size(); i <= N; ++i) {
    CompileUnit *Unit = NewCompileUnit(CUW[i], i);
    CompileUnits.push_back(Unit);
  }
}

/// ConstructGlobalDIEs - Create DIEs for each of the externally visible global
/// variables.
void DwarfWriter::ConstructGlobalDIEs() {
  std::vector<GlobalVariableDesc *> GlobalVariables =
      DebugInfo->getAnchoredDescriptors<GlobalVariableDesc>(*M);
  
  for (unsigned i = 0, N = GlobalVariables.size(); i < N; ++i) {
    GlobalVariableDesc *GVD = GlobalVariables[i];
    NewGlobalVariable(GVD);
  }
}

/// ConstructSubprogramDIEs - Create DIEs for each of the externally visible
/// subprograms.
void DwarfWriter::ConstructSubprogramDIEs() {
  std::vector<SubprogramDesc *> Subprograms =
      DebugInfo->getAnchoredDescriptors<SubprogramDesc>(*M);
  
  for (unsigned i = 0, N = Subprograms.size(); i < N; ++i) {
    SubprogramDesc *SPD = Subprograms[i];
    NewSubprogram(SPD);
  }
}

//===----------------------------------------------------------------------===//
// Main entry points.
//
  
DwarfWriter::DwarfWriter(std::ostream &OS, AsmPrinter *A)
: O(OS)
, Asm(A)
, TD(Asm->TM.getTargetData())
, RI(Asm->TM.getRegisterInfo())
, M(NULL)
, MF(NULL)
, DebugInfo(NULL)
, didInitial(false)
, shouldEmit(false)
, SubprogramCount(0)
, CompileUnits()
, Abbreviations()
, StringPool()
, DescToUnitMap()
, DescToDieMap()
, SectionMap()
, SectionSourceLines()
, AddressSize(sizeof(int32_t))
, hasLEB128(false)
, hasDotLoc(false)
, hasDotFile(false)
, needsSet(false)
, DwarfAbbrevSection(".debug_abbrev")
, DwarfInfoSection(".debug_info")
, DwarfLineSection(".debug_line")
, DwarfFrameSection(".debug_frame")
, DwarfPubNamesSection(".debug_pubnames")
, DwarfPubTypesSection(".debug_pubtypes")
, DwarfStrSection(".debug_str")
, DwarfLocSection(".debug_loc")
, DwarfARangesSection(".debug_aranges")
, DwarfRangesSection(".debug_ranges")
, DwarfMacInfoSection(".debug_macinfo")
, TextSection(".text")
, DataSection(".data")
{}
DwarfWriter::~DwarfWriter() {
  for (unsigned i = 0, N = CompileUnits.size(); i < N; ++i) {
    delete CompileUnits[i];
  }
}

/// SetDebugInfo - Set DebugInfo when it's known that pass manager has
/// created it.  Set by the target AsmPrinter.
void DwarfWriter::SetDebugInfo(MachineDebugInfo *DI) {
  // Make sure initial declarations are made.
  if (!DebugInfo && DI->hasInfo()) {
    DebugInfo = DI;
    shouldEmit = true;
    
    // Emit initial sections
    EmitInitial();
  
    // Create all the compile unit DIEs.
    ConstructCompileUnitDIEs();
    
    // Create DIEs for each of the externally visible global variables.
    ConstructGlobalDIEs();

    // Create DIEs for each of the externally visible subprograms.
    ConstructSubprogramDIEs();
    
    // Prime section data.
    SectionMap.insert(std::string("\t") + TextSection);
  }
}

/// BeginModule - Emit all Dwarf sections that should come prior to the content.
///
void DwarfWriter::BeginModule(Module *M) {
  this->M = M;
  
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf Begin Module");
}

/// EndModule - Emit all Dwarf sections that should come after the content.
///
void DwarfWriter::EndModule() {
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf End Module");
  
  // Standard sections final addresses.
  Asm->SwitchToTextSection(TextSection, 0);
  EmitLabel("text_end", 0);
  Asm->SwitchToDataSection(DataSection, 0);
  EmitLabel("data_end", 0);
  
  // End text sections.
  for (unsigned i = 1, N = SectionMap.size(); i <= N; ++i) {
    Asm->SwitchToTextSection(SectionMap[i].c_str(), 0);
    EmitLabel("section_end", i);
  }
  
  // Compute DIE offsets and sizes.
  SizeAndOffsets();
  
  // Emit all the DIEs into a debug info section
  EmitDebugInfo();
  
  // Corresponding abbreviations into a abbrev section.
  EmitAbbreviations();
  
  // Emit source line correspondence into a debug line section.
  EmitDebugLines();
  
  // Emit info into a debug pubnames section.
  EmitDebugPubNames();
  
  // Emit info into a debug str section.
  EmitDebugStr();
  
  // Emit info into a debug loc section.
  EmitDebugLoc();
  
  // Emit info into a debug aranges section.
  EmitDebugARanges();
  
  // Emit info into a debug ranges section.
  EmitDebugRanges();
  
  // Emit info into a debug macinfo section.
  EmitDebugMacInfo();
}

/// BeginFunction - Gather pre-function debug information.  Assumes being 
/// emitted immediately after the function entry point.
void DwarfWriter::BeginFunction(MachineFunction *MF) {
  this->MF = MF;
  
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf Begin Function");

  // Begin accumulating function debug information.
  DebugInfo->BeginFunction(MF);
  
  // Assumes in correct section after the entry point.
  EmitLabel("func_begin", ++SubprogramCount);
}

/// EndFunction - Gather and emit post-function debug information.
///
void DwarfWriter::EndFunction() {
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf End Function");
  
  // Define end label for subprogram.
  EmitLabel("func_end", SubprogramCount);
    
  // Get function line info.
  std::vector<SourceLineInfo *> &LineInfos = DebugInfo->getSourceLines();

  if (!LineInfos.empty()) {
    // Get section line info.
    unsigned ID = SectionMap.insert(Asm->CurrentSection);
    if (SectionSourceLines.size() < ID) SectionSourceLines.resize(ID);
    std::vector<SourceLineInfo *> &SectionLineInfos =SectionSourceLines[ID-1];
    // Append the function info to section info.
    SectionLineInfos.insert(SectionLineInfos.end(),
                            LineInfos.begin(), LineInfos.end());
  }
  
  // Construct scopes for subprogram.
  ConstructRootScope(DebugInfo->getRootScope());
  
  // Emit function frame information.
  EmitFunctionDebugFrame();
  
  // Reset the line numbers for the next function.
  LineInfos.clear();

  // Clear function debug information.
  DebugInfo->EndFunction();
}
