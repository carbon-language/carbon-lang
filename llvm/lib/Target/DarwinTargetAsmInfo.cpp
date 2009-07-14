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
                                SectionFlags::Mergeable | SectionFlags::Strings);
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
  LessPrivateGlobalPrefix = "l";  // Marker for some ObjC metadata
  StringConstantPrefix = "\1LC";
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
/// the PrivateGlobalPrefix or the LessPrivateGlobalPrefix does not have the
/// directive emitted (this occurs in ObjC metadata).
bool
DarwinTargetAsmInfo::emitUsedDirectiveFor(const GlobalValue* GV,
                                          Mangler *Mang) const {
  if (GV==0)
    return false;
  
  /// FIXME: WHAT IS THIS?
  
  if (GV->hasLocalLinkage() && !isa<Function>(GV) &&
      ((strlen(getPrivateGlobalPrefix()) != 0 &&
        Mang->getMangledName(GV).substr(0,strlen(getPrivateGlobalPrefix())) ==
          getPrivateGlobalPrefix()) ||
       (strlen(getLessPrivateGlobalPrefix()) != 0 &&
        Mang->getMangledName(GV).substr(0,
                                        strlen(getLessPrivateGlobalPrefix())) ==
          getLessPrivateGlobalPrefix())))
    return false;
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
    else
      return TextSection;
   case SectionKind::Data:
   case SectionKind::ThreadData:
   case SectionKind::BSS:
   case SectionKind::ThreadBSS:
    if (cast<GlobalVariable>(GV)->isConstant())
      return (isWeak ? ConstDataCoalSection : ConstDataSection);
    else
      return (isWeak ? DataCoalSection : DataSection);
   case SectionKind::ROData:
    return (isWeak ? ConstDataCoalSection :
            (isNonStatic ? ConstDataSection : getReadOnlySection()));
   case SectionKind::RODataMergeStr:
    return (isWeak ?
            ConstTextCoalSection :
            MergeableStringSection(cast<GlobalVariable>(GV)));
   case SectionKind::RODataMergeConst:
    return (isWeak ?
            ConstDataCoalSection:
            MergeableConstSection(cast<GlobalVariable>(GV)));
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

const Section*
DarwinTargetAsmInfo::MergeableConstSection(const GlobalVariable *GV) const {
  Constant *C = GV->getInitializer();

  return MergeableConstSection(C->getType());
}

inline const Section*
DarwinTargetAsmInfo::MergeableConstSection(const Type *Ty) const {
  const TargetData *TD = TM.getTargetData();

  unsigned Size = TD->getTypeAllocSize(Ty);
  if (Size == 4)
    return FourByteConstantSection;
  else if (Size == 8)
    return EightByteConstantSection;
  else if (Size == 16 && SixteenByteConstantSection)
    return SixteenByteConstantSection;

  return getReadOnlySection();
}

const Section*
DarwinTargetAsmInfo::SelectSectionForMachineConst(const Type *Ty) const {
  const Section* S = MergeableConstSection(Ty);

  // Handle weird special case, when compiling PIC stuff.
  if (S == getReadOnlySection() &&
      TM.getRelocationModel() != Reloc::Static)
    return ConstDataSection;

  return S;
}

std::string
DarwinTargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const {
  llvm_unreachable("Darwin does not use unique sections");
  return "";
}
