//===--- lib/CodeGen/DIE.cpp - DWARF Info Entries -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures for DWARF info entries.
// 
//===----------------------------------------------------------------------===//

#include "DIE.h"
#include "DwarfPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/ErrorHandling.h"
#include <ostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// DIEAbbrevData Implementation
//===----------------------------------------------------------------------===//

/// Profile - Used to gather unique data for the abbreviation folding set.
///
void DIEAbbrevData::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(Attribute);
  ID.AddInteger(Form);
}

//===----------------------------------------------------------------------===//
// DIEAbbrev Implementation
//===----------------------------------------------------------------------===//

/// Profile - Used to gather unique data for the abbreviation folding set.
///
void DIEAbbrev::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(Tag);
  ID.AddInteger(ChildrenFlag);

  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i)
    Data[i].Profile(ID);
}

/// Emit - Print the abbreviation using the specified asm printer.
///
void DIEAbbrev::Emit(const AsmPrinter *Asm) const {
  // Emit its Dwarf tag type.
  Asm->EmitULEB128Bytes(Tag);
  Asm->EOL(dwarf::TagString(Tag));

  // Emit whether it has children DIEs.
  Asm->EmitULEB128Bytes(ChildrenFlag);
  Asm->EOL(dwarf::ChildrenString(ChildrenFlag));

  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    const DIEAbbrevData &AttrData = Data[i];

    // Emit attribute type.
    Asm->EmitULEB128Bytes(AttrData.getAttribute());
    Asm->EOL(dwarf::AttributeString(AttrData.getAttribute()));

    // Emit form type.
    Asm->EmitULEB128Bytes(AttrData.getForm());
    Asm->EOL(dwarf::FormEncodingString(AttrData.getForm()));
  }

  // Mark end of abbreviation.
  Asm->EmitULEB128Bytes(0); Asm->EOL("EOM(1)");
  Asm->EmitULEB128Bytes(0); Asm->EOL("EOM(2)");
}

#ifndef NDEBUG
void DIEAbbrev::print(std::ostream &O) {
  O << "Abbreviation @"
    << std::hex << (intptr_t)this << std::dec
    << "  "
    << dwarf::TagString(Tag)
    << " "
    << dwarf::ChildrenString(ChildrenFlag)
    << "\n";

  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << "  "
      << dwarf::AttributeString(Data[i].getAttribute())
      << "  "
      << dwarf::FormEncodingString(Data[i].getForm())
      << "\n";
  }
}
void DIEAbbrev::dump() { print(cerr); }
#endif

//===----------------------------------------------------------------------===//
// DIE Implementation
//===----------------------------------------------------------------------===//

DIE::~DIE() {
  for (unsigned i = 0, N = Children.size(); i < N; ++i)
    delete Children[i];
}

/// AddSiblingOffset - Add a sibling offset field to the front of the DIE.
///
void DIE::AddSiblingOffset() {
  DIEInteger *DI = new DIEInteger(0);
  Values.insert(Values.begin(), DI);
  Abbrev.AddFirstAttribute(dwarf::DW_AT_sibling, dwarf::DW_FORM_ref4);
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIE::Profile(FoldingSetNodeID &ID) {
  Abbrev.Profile(ID);

  for (unsigned i = 0, N = Children.size(); i < N; ++i)
    ID.AddPointer(Children[i]);

  for (unsigned j = 0, M = Values.size(); j < M; ++j)
    ID.AddPointer(Values[j]);
}

#ifndef NDEBUG
void DIE::print(std::ostream &O, unsigned IncIndent) {
  IndentCount += IncIndent;
  const std::string Indent(IndentCount, ' ');
  bool isBlock = Abbrev.getTag() == 0;

  if (!isBlock) {
    O << Indent
      << "Die: "
      << "0x" << std::hex << (intptr_t)this << std::dec
      << ", Offset: " << Offset
      << ", Size: " << Size
      << "\n";

    O << Indent
      << dwarf::TagString(Abbrev.getTag())
      << " "
      << dwarf::ChildrenString(Abbrev.getChildrenFlag());
  } else {
    O << "Size: " << Size;
  }
  O << "\n";

  const SmallVector<DIEAbbrevData, 8> &Data = Abbrev.getData();

  IndentCount += 2;
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << Indent;

    if (!isBlock)
      O << dwarf::AttributeString(Data[i].getAttribute());
    else
      O << "Blk[" << i << "]";

    O <<  "  "
      << dwarf::FormEncodingString(Data[i].getForm())
      << " ";
    Values[i]->print(O);
    O << "\n";
  }
  IndentCount -= 2;

  for (unsigned j = 0, M = Children.size(); j < M; ++j) {
    Children[j]->print(O, 4);
  }

  if (!isBlock) O << "\n";
  IndentCount -= IncIndent;
}

void DIE::dump() {
  print(cerr);
}
#endif


#ifndef NDEBUG
void DIEValue::dump() {
  print(cerr);
}
#endif

//===----------------------------------------------------------------------===//
// DIEInteger Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit integer of appropriate size.
///
void DIEInteger::EmitValue(Dwarf *D, unsigned Form) const {
  const AsmPrinter *Asm = D->getAsm();
  switch (Form) {
  case dwarf::DW_FORM_flag:  // Fall thru
  case dwarf::DW_FORM_ref1:  // Fall thru
  case dwarf::DW_FORM_data1: Asm->EmitInt8(Integer);         break;
  case dwarf::DW_FORM_ref2:  // Fall thru
  case dwarf::DW_FORM_data2: Asm->EmitInt16(Integer);        break;
  case dwarf::DW_FORM_ref4:  // Fall thru
  case dwarf::DW_FORM_data4: Asm->EmitInt32(Integer);        break;
  case dwarf::DW_FORM_ref8:  // Fall thru
  case dwarf::DW_FORM_data8: Asm->EmitInt64(Integer);        break;
  case dwarf::DW_FORM_udata: Asm->EmitULEB128Bytes(Integer); break;
  case dwarf::DW_FORM_sdata: Asm->EmitSLEB128Bytes(Integer); break;
  default: llvm_unreachable("DIE Value form not supported yet");
  }
}

/// SizeOf - Determine size of integer value in bytes.
///
unsigned DIEInteger::SizeOf(const TargetData *TD, unsigned Form) const {
  switch (Form) {
  case dwarf::DW_FORM_flag:  // Fall thru
  case dwarf::DW_FORM_ref1:  // Fall thru
  case dwarf::DW_FORM_data1: return sizeof(int8_t);
  case dwarf::DW_FORM_ref2:  // Fall thru
  case dwarf::DW_FORM_data2: return sizeof(int16_t);
  case dwarf::DW_FORM_ref4:  // Fall thru
  case dwarf::DW_FORM_data4: return sizeof(int32_t);
  case dwarf::DW_FORM_ref8:  // Fall thru
  case dwarf::DW_FORM_data8: return sizeof(int64_t);
  case dwarf::DW_FORM_udata: return MCAsmInfo::getULEB128Size(Integer);
  case dwarf::DW_FORM_sdata: return MCAsmInfo::getSLEB128Size(Integer);
  default: llvm_unreachable("DIE Value form not supported yet"); break;
  }
  return 0;
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIEInteger::Profile(FoldingSetNodeID &ID, unsigned Int) {
  ID.AddInteger(isInteger);
  ID.AddInteger(Int);
}
void DIEInteger::Profile(FoldingSetNodeID &ID) {
  Profile(ID, Integer);
}

#ifndef NDEBUG
void DIEInteger::print(std::ostream &O) {
  O << "Int: " << (int64_t)Integer
    << "  0x" << std::hex << Integer << std::dec;
}
#endif

//===----------------------------------------------------------------------===//
// DIEString Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit string value.
///
void DIEString::EmitValue(Dwarf *D, unsigned Form) const {
  D->getAsm()->EmitString(Str);
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIEString::Profile(FoldingSetNodeID &ID, const std::string &Str) {
  ID.AddInteger(isString);
  ID.AddString(Str);
}
void DIEString::Profile(FoldingSetNodeID &ID) {
  Profile(ID, Str);
}

#ifndef NDEBUG
void DIEString::print(std::ostream &O) {
  O << "Str: \"" << Str << "\"";
}
#endif

//===----------------------------------------------------------------------===//
// DIEDwarfLabel Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIEDwarfLabel::EmitValue(Dwarf *D, unsigned Form) const {
  bool IsSmall = Form == dwarf::DW_FORM_data4;
  D->EmitReference(Label, false, IsSmall);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIEDwarfLabel::SizeOf(const TargetData *TD, unsigned Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  return TD->getPointerSize();
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIEDwarfLabel::Profile(FoldingSetNodeID &ID, const DWLabel &Label) {
  ID.AddInteger(isLabel);
  Label.Profile(ID);
}
void DIEDwarfLabel::Profile(FoldingSetNodeID &ID) {
  Profile(ID, Label);
}

#ifndef NDEBUG
void DIEDwarfLabel::print(std::ostream &O) {
  O << "Lbl: ";
  Label.print(O);
}
#endif

//===----------------------------------------------------------------------===//
// DIEObjectLabel Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIEObjectLabel::EmitValue(Dwarf *D, unsigned Form) const {
  bool IsSmall = Form == dwarf::DW_FORM_data4;
  D->EmitReference(Label, false, IsSmall);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIEObjectLabel::SizeOf(const TargetData *TD, unsigned Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  return TD->getPointerSize();
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIEObjectLabel::Profile(FoldingSetNodeID &ID, const std::string &Label) {
  ID.AddInteger(isAsIsLabel);
  ID.AddString(Label);
}
void DIEObjectLabel::Profile(FoldingSetNodeID &ID) {
  Profile(ID, Label.c_str());
}

#ifndef NDEBUG
void DIEObjectLabel::print(std::ostream &O) {
  O << "Obj: " << Label;
}
#endif

//===----------------------------------------------------------------------===//
// DIESectionOffset Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIESectionOffset::EmitValue(Dwarf *D, unsigned Form) const {
  bool IsSmall = Form == dwarf::DW_FORM_data4;
  D->EmitSectionOffset(Label.getTag(), Section.getTag(),
                       Label.getNumber(), Section.getNumber(),
                       IsSmall, IsEH, UseSet);
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIESectionOffset::SizeOf(const TargetData *TD, unsigned Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  return TD->getPointerSize();
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIESectionOffset::Profile(FoldingSetNodeID &ID, const DWLabel &Label,
                               const DWLabel &Section) {
  ID.AddInteger(isSectionOffset);
  Label.Profile(ID);
  Section.Profile(ID);
  // IsEH and UseSet are specific to the Label/Section that we will emit the
  // offset for; so Label/Section are enough for uniqueness.
}
void DIESectionOffset::Profile(FoldingSetNodeID &ID) {
  Profile(ID, Label, Section);
}

#ifndef NDEBUG
void DIESectionOffset::print(std::ostream &O) {
  O << "Off: ";
  Label.print(O);
  O << "-";
  Section.print(O);
  O << "-" << IsEH << "-" << UseSet;
}
#endif

//===----------------------------------------------------------------------===//
// DIEDelta Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIEDelta::EmitValue(Dwarf *D, unsigned Form) const {
  bool IsSmall = Form == dwarf::DW_FORM_data4;
  D->EmitDifference(LabelHi, LabelLo, IsSmall);
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEDelta::SizeOf(const TargetData *TD, unsigned Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  return TD->getPointerSize();
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIEDelta::Profile(FoldingSetNodeID &ID, const DWLabel &LabelHi,
                       const DWLabel &LabelLo) {
  ID.AddInteger(isDelta);
  LabelHi.Profile(ID);
  LabelLo.Profile(ID);
}
void DIEDelta::Profile(FoldingSetNodeID &ID) {
  Profile(ID, LabelHi, LabelLo);
}

#ifndef NDEBUG
void DIEDelta::print(std::ostream &O) {
  O << "Del: ";
  LabelHi.print(O);
  O << "-";
  LabelLo.print(O);
}
#endif

//===----------------------------------------------------------------------===//
// DIEEntry Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit debug information entry offset.
///
void DIEEntry::EmitValue(Dwarf *D, unsigned Form) const {
  D->getAsm()->EmitInt32(Entry->getOffset());
}

/// Profile - Used to gather unique data for the value folding set.
///
void DIEEntry::Profile(FoldingSetNodeID &ID, DIE *Entry) {
  ID.AddInteger(isEntry);
  ID.AddPointer(Entry);
}
void DIEEntry::Profile(FoldingSetNodeID &ID) {
  ID.AddInteger(isEntry);

  if (Entry)
    ID.AddPointer(Entry);
  else
    ID.AddPointer(this);
}

#ifndef NDEBUG
void DIEEntry::print(std::ostream &O) {
  O << "Die: 0x" << std::hex << (intptr_t)Entry << std::dec;
}
#endif

//===----------------------------------------------------------------------===//
// DIEBlock Implementation
//===----------------------------------------------------------------------===//

/// ComputeSize - calculate the size of the block.
///
unsigned DIEBlock::ComputeSize(const TargetData *TD) {
  if (!Size) {
    const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev.getData();
    for (unsigned i = 0, N = Values.size(); i < N; ++i)
      Size += Values[i]->SizeOf(TD, AbbrevData[i].getForm());
  }

  return Size;
}

/// EmitValue - Emit block data.
///
void DIEBlock::EmitValue(Dwarf *D, unsigned Form) const {
  const AsmPrinter *Asm = D->getAsm();
  switch (Form) {
  case dwarf::DW_FORM_block1: Asm->EmitInt8(Size);         break;
  case dwarf::DW_FORM_block2: Asm->EmitInt16(Size);        break;
  case dwarf::DW_FORM_block4: Asm->EmitInt32(Size);        break;
  case dwarf::DW_FORM_block:  Asm->EmitULEB128Bytes(Size); break;
  default: llvm_unreachable("Improper form for block");         break;
  }

  const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev.getData();
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    Asm->EOL();
    Values[i]->EmitValue(D, AbbrevData[i].getForm());
  }
}

/// SizeOf - Determine size of block data in bytes.
///
unsigned DIEBlock::SizeOf(const TargetData *TD, unsigned Form) const {
  switch (Form) {
  case dwarf::DW_FORM_block1: return Size + sizeof(int8_t);
  case dwarf::DW_FORM_block2: return Size + sizeof(int16_t);
  case dwarf::DW_FORM_block4: return Size + sizeof(int32_t);
  case dwarf::DW_FORM_block: return Size + MCAsmInfo::getULEB128Size(Size);
  default: llvm_unreachable("Improper form for block"); break;
  }
  return 0;
}

void DIEBlock::Profile(FoldingSetNodeID &ID) {
  ID.AddInteger(isBlock);
  DIE::Profile(ID);
}

#ifndef NDEBUG
void DIEBlock::print(std::ostream &O) {
  O << "Blk: ";
  DIE::print(O, 5);
}
#endif
