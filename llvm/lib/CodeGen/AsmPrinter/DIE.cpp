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
#include "DwarfDebug.h"
#include "DwarfUnit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// DIEAbbrevData Implementation
//===----------------------------------------------------------------------===//

/// Profile - Used to gather unique data for the abbreviation folding set.
///
void DIEAbbrevData::Profile(FoldingSetNodeID &ID) const {
  // Explicitly cast to an integer type for which FoldingSetNodeID has
  // overloads.  Otherwise MSVC 2010 thinks this call is ambiguous.
  ID.AddInteger(unsigned(Attribute));
  ID.AddInteger(unsigned(Form));
}

//===----------------------------------------------------------------------===//
// DIEAbbrev Implementation
//===----------------------------------------------------------------------===//

/// Profile - Used to gather unique data for the abbreviation folding set.
///
void DIEAbbrev::Profile(FoldingSetNodeID &ID) const {
  ID.AddInteger(unsigned(Tag));
  ID.AddInteger(unsigned(Children));

  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i)
    Data[i].Profile(ID);
}

/// Emit - Print the abbreviation using the specified asm printer.
///
void DIEAbbrev::Emit(AsmPrinter *AP) const {
  // Emit its Dwarf tag type.
  AP->EmitULEB128(Tag, dwarf::TagString(Tag));

  // Emit whether it has children DIEs.
  AP->EmitULEB128((unsigned)Children, dwarf::ChildrenString(Children));

  // For each attribute description.
  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    const DIEAbbrevData &AttrData = Data[i];

    // Emit attribute type.
    AP->EmitULEB128(AttrData.getAttribute(),
                    dwarf::AttributeString(AttrData.getAttribute()));

    // Emit form type.
    AP->EmitULEB128(AttrData.getForm(),
                    dwarf::FormEncodingString(AttrData.getForm()));
  }

  // Mark end of abbreviation.
  AP->EmitULEB128(0, "EOM(1)");
  AP->EmitULEB128(0, "EOM(2)");
}

#ifndef NDEBUG
void DIEAbbrev::print(raw_ostream &O) {
  O << "Abbreviation @"
    << format("0x%lx", (long)(intptr_t)this)
    << "  "
    << dwarf::TagString(Tag)
    << " "
    << dwarf::ChildrenString(Children)
    << '\n';

  for (unsigned i = 0, N = Data.size(); i < N; ++i) {
    O << "  "
      << dwarf::AttributeString(Data[i].getAttribute())
      << "  "
      << dwarf::FormEncodingString(Data[i].getForm())
      << '\n';
  }
}
void DIEAbbrev::dump() { print(dbgs()); }
#endif

/// Climb up the parent chain to get the unit DIE to which this DIE
/// belongs.
const DIE *DIE::getUnit() const {
  const DIE *Cu = getUnitOrNull();
  assert(Cu && "We should not have orphaned DIEs.");
  return Cu;
}

/// Climb up the parent chain to get the unit DIE this DIE belongs
/// to. Return NULL if DIE is not added to an owner yet.
const DIE *DIE::getUnitOrNull() const {
  const DIE *p = this;
  while (p) {
    if (p->getTag() == dwarf::DW_TAG_compile_unit ||
        p->getTag() == dwarf::DW_TAG_type_unit)
      return p;
    p = p->getParent();
  }
  return nullptr;
}

DIEValue *DIE::findAttribute(dwarf::Attribute Attribute) const {
  const SmallVectorImpl<DIEValue *> &Values = getValues();
  const DIEAbbrev &Abbrevs = getAbbrev();

  // Iterate through all the attributes until we find the one we're
  // looking for, if we can't find it return NULL.
  for (size_t i = 0; i < Values.size(); ++i)
    if (Abbrevs.getData()[i].getAttribute() == Attribute)
      return Values[i];
  return nullptr;
}

#ifndef NDEBUG
void DIE::print(raw_ostream &O, unsigned IndentCount) const {
  const std::string Indent(IndentCount, ' ');
  bool isBlock = Abbrev.getTag() == 0;

  if (!isBlock) {
    O << Indent
      << "Die: "
      << format("0x%lx", (long)(intptr_t)this)
      << ", Offset: " << Offset
      << ", Size: " << Size << "\n";

    O << Indent
      << dwarf::TagString(Abbrev.getTag())
      << " "
      << dwarf::ChildrenString(Abbrev.hasChildren()) << "\n";
  } else {
    O << "Size: " << Size << "\n";
  }

  const SmallVectorImpl<DIEAbbrevData> &Data = Abbrev.getData();

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
    Children[j]->print(O, IndentCount+4);
  }

  if (!isBlock) O << "\n";
}

void DIE::dump() {
  print(dbgs());
}
#endif

void DIEValue::anchor() { }

#ifndef NDEBUG
void DIEValue::dump() const {
  print(dbgs());
}
#endif

//===----------------------------------------------------------------------===//
// DIEInteger Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit integer of appropriate size.
///
void DIEInteger::EmitValue(AsmPrinter *Asm, dwarf::Form Form) const {
  unsigned Size = ~0U;
  switch (Form) {
  case dwarf::DW_FORM_flag_present:
    // Emit something to keep the lines and comments in sync.
    // FIXME: Is there a better way to do this?
    Asm->OutStreamer.AddBlankLine();
    return;
  case dwarf::DW_FORM_flag:  // Fall thru
  case dwarf::DW_FORM_ref1:  // Fall thru
  case dwarf::DW_FORM_data1: Size = 1; break;
  case dwarf::DW_FORM_ref2:  // Fall thru
  case dwarf::DW_FORM_data2: Size = 2; break;
  case dwarf::DW_FORM_sec_offset: // Fall thru
  case dwarf::DW_FORM_ref4:  // Fall thru
  case dwarf::DW_FORM_data4: Size = 4; break;
  case dwarf::DW_FORM_ref8:  // Fall thru
  case dwarf::DW_FORM_ref_sig8:  // Fall thru
  case dwarf::DW_FORM_data8: Size = 8; break;
  case dwarf::DW_FORM_GNU_str_index: Asm->EmitULEB128(Integer); return;
  case dwarf::DW_FORM_GNU_addr_index: Asm->EmitULEB128(Integer); return;
  case dwarf::DW_FORM_udata: Asm->EmitULEB128(Integer); return;
  case dwarf::DW_FORM_sdata: Asm->EmitSLEB128(Integer); return;
  case dwarf::DW_FORM_addr:
    Size = Asm->getDataLayout().getPointerSize(); break;
  default: llvm_unreachable("DIE Value form not supported yet");
  }
  Asm->OutStreamer.EmitIntValue(Integer, Size);
}

/// SizeOf - Determine size of integer value in bytes.
///
unsigned DIEInteger::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  switch (Form) {
  case dwarf::DW_FORM_flag_present: return 0;
  case dwarf::DW_FORM_flag:  // Fall thru
  case dwarf::DW_FORM_ref1:  // Fall thru
  case dwarf::DW_FORM_data1: return sizeof(int8_t);
  case dwarf::DW_FORM_ref2:  // Fall thru
  case dwarf::DW_FORM_data2: return sizeof(int16_t);
  case dwarf::DW_FORM_sec_offset: // Fall thru
  case dwarf::DW_FORM_ref4:  // Fall thru
  case dwarf::DW_FORM_data4: return sizeof(int32_t);
  case dwarf::DW_FORM_ref8:  // Fall thru
  case dwarf::DW_FORM_ref_sig8:  // Fall thru
  case dwarf::DW_FORM_data8: return sizeof(int64_t);
  case dwarf::DW_FORM_GNU_str_index: return getULEB128Size(Integer);
  case dwarf::DW_FORM_GNU_addr_index: return getULEB128Size(Integer);
  case dwarf::DW_FORM_udata: return getULEB128Size(Integer);
  case dwarf::DW_FORM_sdata: return getSLEB128Size(Integer);
  case dwarf::DW_FORM_addr:  return AP->getDataLayout().getPointerSize();
  default: llvm_unreachable("DIE Value form not supported yet");
  }
}

#ifndef NDEBUG
void DIEInteger::print(raw_ostream &O) const {
  O << "Int: " << (int64_t)Integer << "  0x";
  O.write_hex(Integer);
}
#endif

//===----------------------------------------------------------------------===//
// DIEExpr Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit expression value.
///
void DIEExpr::EmitValue(AsmPrinter *AP, dwarf::Form Form) const {
  AP->OutStreamer.EmitValue(Expr, SizeOf(AP, Form));
}

/// SizeOf - Determine size of expression value in bytes.
///
unsigned DIEExpr::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  if (Form == dwarf::DW_FORM_sec_offset) return 4;
  if (Form == dwarf::DW_FORM_strp) return 4;
  return AP->getDataLayout().getPointerSize();
}

#ifndef NDEBUG
void DIEExpr::print(raw_ostream &O) const {
  O << "Expr: ";
  Expr->print(O);
}
#endif

//===----------------------------------------------------------------------===//
// DIELabel Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIELabel::EmitValue(AsmPrinter *AP, dwarf::Form Form) const {
  AP->EmitLabelReference(Label, SizeOf(AP, Form),
                         Form == dwarf::DW_FORM_strp ||
                             Form == dwarf::DW_FORM_sec_offset ||
                             Form == dwarf::DW_FORM_ref_addr);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIELabel::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  if (Form == dwarf::DW_FORM_sec_offset) return 4;
  if (Form == dwarf::DW_FORM_strp) return 4;
  return AP->getDataLayout().getPointerSize();
}

#ifndef NDEBUG
void DIELabel::print(raw_ostream &O) const {
  O << "Lbl: " << Label->getName();
}
#endif

//===----------------------------------------------------------------------===//
// DIEDelta Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIEDelta::EmitValue(AsmPrinter *AP, dwarf::Form Form) const {
  AP->EmitLabelDifference(LabelHi, LabelLo, SizeOf(AP, Form));
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEDelta::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  if (Form == dwarf::DW_FORM_data4) return 4;
  if (Form == dwarf::DW_FORM_sec_offset) return 4;
  if (Form == dwarf::DW_FORM_strp) return 4;
  return AP->getDataLayout().getPointerSize();
}

#ifndef NDEBUG
void DIEDelta::print(raw_ostream &O) const {
  O << "Del: " << LabelHi->getName() << "-" << LabelLo->getName();
}
#endif

//===----------------------------------------------------------------------===//
// DIEString Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit string value.
///
void DIEString::EmitValue(AsmPrinter *AP, dwarf::Form Form) const {
  Access->EmitValue(AP, Form);
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEString::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  return Access->SizeOf(AP, Form);
}

#ifndef NDEBUG
void DIEString::print(raw_ostream &O) const {
  O << "String: " << Str << "\tSymbol: ";
  Access->print(O);
}
#endif

//===----------------------------------------------------------------------===//
// DIEEntry Implementation
//===----------------------------------------------------------------------===//

/// EmitValue - Emit debug information entry offset.
///
void DIEEntry::EmitValue(AsmPrinter *AP, dwarf::Form Form) const {

  if (Form == dwarf::DW_FORM_ref_addr) {
    const DwarfDebug *DD = AP->getDwarfDebug();
    unsigned Addr = Entry.getOffset();
    assert(!DD->useSplitDwarf() && "TODO: dwo files can't have relocations.");
    // For DW_FORM_ref_addr, output the offset from beginning of debug info
    // section. Entry->getOffset() returns the offset from start of the
    // compile unit.
    DwarfCompileUnit *CU = DD->lookupUnit(Entry.getUnit());
    assert(CU && "CUDie should belong to a CU.");
    Addr += CU->getDebugInfoOffset();
    if (AP->MAI->doesDwarfUseRelocationsAcrossSections())
      AP->EmitLabelPlusOffset(CU->getSectionSym(), Addr,
                              DIEEntry::getRefAddrSize(AP));
    else
      AP->EmitLabelOffsetDifference(CU->getSectionSym(), Addr,
                                    CU->getSectionSym(),
                                    DIEEntry::getRefAddrSize(AP));
  } else
    AP->EmitInt32(Entry.getOffset());
}

unsigned DIEEntry::getRefAddrSize(AsmPrinter *AP) {
  // DWARF4: References that use the attribute form DW_FORM_ref_addr are
  // specified to be four bytes in the DWARF 32-bit format and eight bytes
  // in the DWARF 64-bit format, while DWARF Version 2 specifies that such
  // references have the same size as an address on the target system.
  const DwarfDebug *DD = AP->getDwarfDebug();
  assert(DD && "Expected Dwarf Debug info to be available");
  if (DD->getDwarfVersion() == 2)
    return AP->getDataLayout().getPointerSize();
  return sizeof(int32_t);
}

#ifndef NDEBUG
void DIEEntry::print(raw_ostream &O) const {
  O << format("Die: 0x%lx", (long)(intptr_t)&Entry);
}
#endif

//===----------------------------------------------------------------------===//
// DIETypeSignature Implementation
//===----------------------------------------------------------------------===//
void DIETypeSignature::EmitValue(AsmPrinter *Asm, dwarf::Form Form) const {
  assert(Form == dwarf::DW_FORM_ref_sig8);
  Asm->OutStreamer.EmitIntValue(Unit.getTypeSignature(), 8);
}

#ifndef NDEBUG
void DIETypeSignature::print(raw_ostream &O) const {
  O << format("Type Unit: 0x%lx", Unit.getTypeSignature());
}

void DIETypeSignature::dump() const { print(dbgs()); }
#endif

//===----------------------------------------------------------------------===//
// DIELoc Implementation
//===----------------------------------------------------------------------===//

/// ComputeSize - calculate the size of the location expression.
///
unsigned DIELoc::ComputeSize(AsmPrinter *AP) const {
  if (!Size) {
    const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();
    for (unsigned i = 0, N = Values.size(); i < N; ++i)
      Size += Values[i]->SizeOf(AP, AbbrevData[i].getForm());
  }

  return Size;
}

/// EmitValue - Emit location data.
///
void DIELoc::EmitValue(AsmPrinter *Asm, dwarf::Form Form) const {
  switch (Form) {
  default: llvm_unreachable("Improper form for block");
  case dwarf::DW_FORM_block1: Asm->EmitInt8(Size);    break;
  case dwarf::DW_FORM_block2: Asm->EmitInt16(Size);   break;
  case dwarf::DW_FORM_block4: Asm->EmitInt32(Size);   break;
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_exprloc:
    Asm->EmitULEB128(Size); break;
  }

  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    Values[i]->EmitValue(Asm, AbbrevData[i].getForm());
}

/// SizeOf - Determine size of location data in bytes.
///
unsigned DIELoc::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  switch (Form) {
  case dwarf::DW_FORM_block1: return Size + sizeof(int8_t);
  case dwarf::DW_FORM_block2: return Size + sizeof(int16_t);
  case dwarf::DW_FORM_block4: return Size + sizeof(int32_t);
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_exprloc:
    return Size + getULEB128Size(Size);
  default: llvm_unreachable("Improper form for block");
  }
}

#ifndef NDEBUG
void DIELoc::print(raw_ostream &O) const {
  O << "ExprLoc: ";
  DIE::print(O, 5);
}
#endif

//===----------------------------------------------------------------------===//
// DIEBlock Implementation
//===----------------------------------------------------------------------===//

/// ComputeSize - calculate the size of the block.
///
unsigned DIEBlock::ComputeSize(AsmPrinter *AP) const {
  if (!Size) {
    const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();
    for (unsigned i = 0, N = Values.size(); i < N; ++i)
      Size += Values[i]->SizeOf(AP, AbbrevData[i].getForm());
  }

  return Size;
}

/// EmitValue - Emit block data.
///
void DIEBlock::EmitValue(AsmPrinter *Asm, dwarf::Form Form) const {
  switch (Form) {
  default: llvm_unreachable("Improper form for block");
  case dwarf::DW_FORM_block1: Asm->EmitInt8(Size);    break;
  case dwarf::DW_FORM_block2: Asm->EmitInt16(Size);   break;
  case dwarf::DW_FORM_block4: Asm->EmitInt32(Size);   break;
  case dwarf::DW_FORM_block:  Asm->EmitULEB128(Size); break;
  }

  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    Values[i]->EmitValue(Asm, AbbrevData[i].getForm());
}

/// SizeOf - Determine size of block data in bytes.
///
unsigned DIEBlock::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  switch (Form) {
  case dwarf::DW_FORM_block1: return Size + sizeof(int8_t);
  case dwarf::DW_FORM_block2: return Size + sizeof(int16_t);
  case dwarf::DW_FORM_block4: return Size + sizeof(int32_t);
  case dwarf::DW_FORM_block:  return Size + getULEB128Size(Size);
  default: llvm_unreachable("Improper form for block");
  }
}

#ifndef NDEBUG
void DIEBlock::print(raw_ostream &O) const {
  O << "Blk: ";
  DIE::print(O, 5);
}
#endif

//===----------------------------------------------------------------------===//
// DIELocList Implementation
//===----------------------------------------------------------------------===//

unsigned DIELocList::SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
  if (Form == dwarf::DW_FORM_data4)
    return 4;
  if (Form == dwarf::DW_FORM_sec_offset)
    return 4;
  return AP->getDataLayout().getPointerSize();
}

/// EmitValue - Emit label value.
///
void DIELocList::EmitValue(AsmPrinter *AP, dwarf::Form Form) const {
  DwarfDebug *DD = AP->getDwarfDebug();
  MCSymbol *Label = DD->getDebugLocEntries()[Index].Label;

  if (AP->MAI->doesDwarfUseRelocationsAcrossSections() && !DD->useSplitDwarf())
    AP->EmitSectionOffset(Label, DD->getDebugLocSym());
  else
    AP->EmitLabelDifference(Label, DD->getDebugLocSym(), 4);
}

#ifndef NDEBUG
void DIELocList::print(raw_ostream &O) const {
  O << "LocList: " << Index;

}
#endif
