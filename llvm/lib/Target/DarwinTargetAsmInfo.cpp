//===-- DarwinTargetAsmInfo.cpp - Darwin asm properties ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

using namespace llvm;

DarwinTargetAsmInfo::DarwinTargetAsmInfo(const TargetMachine &TM) 
  : TargetAsmInfo(TM) {

  CStringSection_ = getUnnamedSection("\t.cstring",
                                SectionFlags::Mergeable |SectionFlags::Strings);
  FourByteConstantSection = getUnnamedSection("\t.literal4\n",
                                              SectionFlags::Mergeable);
  EightByteConstantSection = getUnnamedSection("\t.literal8\n",
                                               SectionFlags::Mergeable);

  // Note: 16-byte constant section is subtarget specific and should be provided
  // there, if needed.
  SixteenByteConstantSection = 0;

  ReadOnlySection = getUnnamedSection("\t.const\n", SectionFlags::None);

  TextCoalSection =
    getNamedSection("\t__TEXT,__textcoal_nt,coalesced,pure_instructions",
                    SectionFlags::Code);
  ConstTextCoalSection = getNamedSection("\t__TEXT,__const_coal,coalesced",
                                         SectionFlags::None);
  ConstDataCoalSection = getNamedSection("\t__DATA,__const_coal,coalesced",
                                         SectionFlags::None);
  ConstDataSection = getUnnamedSection(".const_data", SectionFlags::None);
  DataCoalSection = getNamedSection("\t__DATA,__datacoal_nt,coalesced",
                                    SectionFlags::Writeable);
    
  
  // Common settings for all Darwin targets.
  // Syntax:
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  LinkerPrivateGlobalPrefix = "l";  // Marker for some ObjC metadata
  NeedsSet = true;
  NeedsIndirectEncoding = true;
  AllowQuotesInName = true;
  HasSingleParameterDotFile = false;

  // In non-PIC modes, emit a special label before jump tables so that the
  // linker can perform more accurate dead code stripping.  We do not check the
  // relocation model here since it can be overridden later.
  JumpTableSpecialLabelPrefix = "l";
    
  // Directives:
  WeakDefDirective = "\t.weak_definition ";
  WeakRefDirective = "\t.weak_reference ";
  HiddenDirective = "\t.private_extern ";
    
  // Sections:
  CStringSection = "\t.cstring";
  JumpTableDataSection = "\t.const\n";
  BSSSection = 0;

  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorsSection = ".constructor";
    StaticDtorsSection = ".destructor";
  } else {
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
  }
    
  // _foo.eh symbols are currently always exported so that the linker knows
  // about them.  This may not strictly be necessary on 10.6 and later, but it
  // doesn't hurt anything.
  Is_EHSymbolPrivate = false;
    
  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
  DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
  DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
  DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
  DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
  DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
  DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
  DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
  DwarfMacroInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
}

/// emitUsedDirectiveFor - On Darwin, internally linked data beginning with
/// the PrivateGlobalPrefix or the LinkerPrivateGlobalPrefix does not have the
/// directive emitted (this occurs in ObjC metadata).
bool DarwinTargetAsmInfo::emitUsedDirectiveFor(const GlobalValue* GV,
                                               Mangler *Mang) const {
  if (!GV) return false;
  
  // Check whether the mangled name has the "Private" or "LinkerPrivate" prefix.
  if (GV->hasLocalLinkage() && !isa<Function>(GV)) {
    // FIXME: ObjC metadata is currently emitted as internal symbols that have
    // \1L and \0l prefixes on them.  Fix them to be Private/LinkerPrivate and
    // this horrible hack can go away.
    const std::string &Name = Mang->getMangledName(GV);
    if (Name[0] == 'L' || Name[0] == 'l')
      return false;
  }

  return true;
}

const Section*
DarwinTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);
  bool isWeak = GV->isWeakForLinker();
  bool isNonStatic = TM.getRelocationModel() != Reloc::Static;

  switch (Kind) {
  case SectionKind::Text:
    if (isWeak)
      return TextCoalSection;
    return TextSection;
  case SectionKind::Data:
  case SectionKind::ThreadData:
  case SectionKind::BSS:
  case SectionKind::ThreadBSS:
    if (cast<GlobalVariable>(GV)->isConstant())
      return isWeak ? ConstDataCoalSection : ConstDataSection;
    return isWeak ? DataCoalSection : DataSection;

  case SectionKind::ROData:
    return (isWeak ? ConstDataCoalSection :
            (isNonStatic ? ConstDataSection : getReadOnlySection()));
  case SectionKind::RODataMergeStr:
    return (isWeak ?
            ConstTextCoalSection :
            MergeableStringSection(cast<GlobalVariable>(GV)));
  case SectionKind::RODataMergeConst: {
    if (isWeak) return ConstDataCoalSection;
    const Type *Ty = cast<GlobalVariable>(GV)->getInitializer()->getType();
    const TargetData *TD = TM.getTargetData();
    return getSectionForMergableConstant(TD->getTypeAllocSize(Ty), 0);
  }
  default:
    llvm_unreachable("Unsuported section kind for global");
  }

  // FIXME: Do we have any extra special weird cases?
  return NULL;
}

const Section*
DarwinTargetAsmInfo::MergeableStringSection(const GlobalVariable *GV) const {
  const TargetData *TD = TM.getTargetData();
  Constant *C = cast<GlobalVariable>(GV)->getInitializer();
  const Type *Ty = cast<ArrayType>(C->getType())->getElementType();

  unsigned Size = TD->getTypeAllocSize(Ty);
  if (Size) {
    unsigned Align = TD->getPreferredAlignment(GV);
    if (Align <= 32)
      return getCStringSection_();
  }

  return getReadOnlySection();
}

const Section *
DarwinTargetAsmInfo::getSectionForMergableConstant(uint64_t Size,
                                                   unsigned ReloInfo) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (ReloInfo != 0)
    return ConstDataSection;
  
  switch (Size) {
  default: break;
  case 4:
    return FourByteConstantSection;
  case 8:
    return EightByteConstantSection;
  case 16:
    if (SixteenByteConstantSection)
      return SixteenByteConstantSection;
    break;
  }
  
  return ReadOnlySection;  // .const
}

