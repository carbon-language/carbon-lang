//===-- llvm/Target/TargetLoweringObjectFile.cpp - Object File Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Mangler.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Generic Code
//===----------------------------------------------------------------------===//

TargetLoweringObjectFile::TargetLoweringObjectFile() : Ctx(0) {
  TextSection = 0;
  DataSection = 0;
  BSSSection_ = 0;
  ReadOnlySection = 0;
  TLSDataSection = 0;
  TLSBSSSection = 0;
  CStringSection_ = 0;
}

TargetLoweringObjectFile::~TargetLoweringObjectFile() {
}

static bool isSuitableForBSS(const GlobalVariable *GV) {
  Constant *C = GV->getInitializer();
  
  // Must have zero initializer.
  if (!C->isNullValue())
    return false;
  
  // Leave constant zeros in readonly constant sections, so they can be shared.
  if (GV->isConstant())
    return false;
  
  // If the global has an explicit section specified, don't put it in BSS.
  if (!GV->getSection().empty())
    return false;
  
  // If -nozero-initialized-in-bss is specified, don't ever use BSS.
  if (NoZerosInBSS)
    return false;
  
  // Otherwise, put it in BSS!
  return true;
}

static bool isConstantString(const Constant *C) {
  // First check: is we have constant array of i8 terminated with zero
  const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
  // Check, if initializer is a null-terminated string
  if (CVA && CVA->isCString())
    return true;

  // Another possibility: [1 x i8] zeroinitializer
  if (isa<ConstantAggregateZero>(C))
    if (const ArrayType *Ty = dyn_cast<ArrayType>(C->getType()))
      return (Ty->getElementType() == Type::Int8Ty &&
              Ty->getNumElements() == 1);

  return false;
}

static SectionKind::Kind SectionKindForGlobal(const GlobalValue *GV,
                                              const TargetMachine &TM) {
  Reloc::Model ReloModel = TM.getRelocationModel();
  
  // Early exit - functions should be always in text sections.
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (GVar == 0)
    return SectionKind::Text;

  
  // Handle thread-local data first.
  if (GVar->isThreadLocal()) {
    if (isSuitableForBSS(GVar))
      return SectionKind::ThreadBSS;
    return SectionKind::ThreadData;
  }

  // Variable can be easily put to BSS section.
  if (isSuitableForBSS(GVar))
    return SectionKind::BSS;

  Constant *C = GVar->getInitializer();
  
  // If the global is marked constant, we can put it into a mergable section,
  // a mergable string section, or general .data if it contains relocations.
  if (GVar->isConstant()) {
    // If the initializer for the global contains something that requires a
    // relocation, then we may have to drop this into a wriable data section
    // even though it is marked const.
    switch (C->getRelocationInfo()) {
    default: llvm_unreachable("unknown relocation info kind");
    case Constant::NoRelocation:
      // If initializer is a null-terminated string, put it in a "cstring"
      // section if the target has it.
      if (isConstantString(C))
        return SectionKind::MergeableCString;
      
      // Otherwise, just drop it into a mergable constant section.  If we have
      // a section for this size, use it, otherwise use the arbitrary sized
      // mergable section.
      switch (TM.getTargetData()->getTypeAllocSize(C->getType())) {
      case 4:  return SectionKind::MergeableConst4;
      case 8:  return SectionKind::MergeableConst8;
      case 16: return SectionKind::MergeableConst16;
      default: return SectionKind::MergeableConst;
      }
      
    case Constant::LocalRelocation:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.  However, we can't put this into a mergable section, because
      // the linker doesn't take relocations into consideration when it tries to
      // merge entries in the section.
      if (ReloModel == Reloc::Static)
        return SectionKind::ReadOnly;
              
      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel.local section.
      return SectionKind::ReadOnlyWithRelLocal;
              
    case Constant::GlobalRelocations:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.  However, we can't put this into a mergable section, because
      // the linker doesn't take relocations into consideration when it tries to
      // merge entries in the section.
      if (ReloModel == Reloc::Static)
        return SectionKind::ReadOnly;
      
      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel section.
      return SectionKind::ReadOnlyWithRel;
    }
  }

  // Okay, this isn't a constant.  If the initializer for the global is going
  // to require a runtime relocation by the dynamic linker, put it into a more
  // specific section to improve startup time of the app.  This coalesces these
  // globals together onto fewer pages, improving the locality of the dynamic
  // linker.
  if (ReloModel == Reloc::Static)
    return SectionKind::DataNoRel;

  switch (C->getRelocationInfo()) {
  default: llvm_unreachable("unknown relocation info kind");
  case Constant::NoRelocation:
    return SectionKind::DataNoRel;
  case Constant::LocalRelocation:
    return SectionKind::DataRelLocal;
  case Constant::GlobalRelocations:
    return SectionKind::DataRel;
  }
}

/// SectionForGlobal - This method computes the appropriate section to emit
/// the specified global variable or function definition.  This should not
/// be passed external (or available externally) globals.
const MCSection *TargetLoweringObjectFile::
SectionForGlobal(const GlobalValue *GV, Mangler *Mang,
                 const TargetMachine &TM) const {
  assert(!GV->isDeclaration() && !GV->hasAvailableExternallyLinkage() &&
         "Can only be used for global definitions");
  
  SectionKind::Kind GVKind = SectionKindForGlobal(GV, TM);
  
  SectionKind Kind = SectionKind::get(GVKind, GV->isWeakForLinker(),
                                      GV->hasSection());


  // Select section name.
  if (GV->hasSection()) {
    // If the target has special section hacks for specifically named globals,
    // return them now.
    if (const MCSection *TS = getSpecialCasedSectionGlobals(GV, Mang, Kind))
      return TS;
    
    // If the target has magic semantics for certain section names, make sure to
    // pick up the flags.  This allows the user to write things with attribute
    // section and still get the appropriate section flags printed.
    GVKind = getKindForNamedSection(GV->getSection().c_str(), GVKind);
    
    return getOrCreateSection(GV->getSection().c_str(), false, GVKind);
  }

  
  // Use default section depending on the 'type' of global
  return SelectSectionForGlobal(GV, Kind, Mang, TM);
}

// Lame default implementation. Calculate the section name for global.
const MCSection *
TargetLoweringObjectFile::SelectSectionForGlobal(const GlobalValue *GV,
                                                 SectionKind Kind,
                                                 Mangler *Mang,
                                                 const TargetMachine &TM) const{
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");
  
  if (Kind.isText())
    return getTextSection();
  
  if (Kind.isBSS() && BSSSection_ != 0)
    return BSSSection_;
  
  if (Kind.isReadOnly() && ReadOnlySection != 0)
    return ReadOnlySection;

  return getDataSection();
}

/// getSectionForMergableConstant - Given a mergable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const MCSection *
TargetLoweringObjectFile::
getSectionForMergeableConstant(SectionKind Kind) const {
  if (Kind.isReadOnly() && ReadOnlySection != 0)
    return ReadOnlySection;
  
  return DataSection;
}


const MCSection *TargetLoweringObjectFile::
getOrCreateSection(const char *Name, bool isDirective,
                   SectionKind::Kind Kind) const {
  if (MCSection *S = Ctx->GetSection(Name))
    return S;
  SectionKind K = SectionKind::get(Kind, false /*weak*/, !isDirective);
  return MCSectionWithKind::Create(Name, K, *Ctx);
}



//===----------------------------------------------------------------------===//
//                                  ELF
//===----------------------------------------------------------------------===//

void TargetLoweringObjectFileELF::Initialize(MCContext &Ctx,
                                             const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  if (!HasCrazyBSS)
    BSSSection_ = getOrCreateSection("\t.bss", true, SectionKind::BSS);
  else
    // PPC/Linux doesn't support the .bss directive, it needs .section .bss.
    // FIXME: Does .section .bss work everywhere??
    BSSSection_ = getOrCreateSection("\t.bss", false, SectionKind::BSS);

    
  TextSection = getOrCreateSection("\t.text", true, SectionKind::Text);
  DataSection = getOrCreateSection("\t.data", true, SectionKind::DataRel);
  ReadOnlySection =
    getOrCreateSection("\t.rodata", false, SectionKind::ReadOnly);
  TLSDataSection =
    getOrCreateSection("\t.tdata", false, SectionKind::ThreadData);
  CStringSection_ = getOrCreateSection("\t.rodata.str", true,
                                       SectionKind::MergeableCString);

  TLSBSSSection = getOrCreateSection("\t.tbss", false, SectionKind::ThreadBSS);

  DataRelSection = getOrCreateSection("\t.data.rel", false,
                                      SectionKind::DataRel);
  DataRelLocalSection = getOrCreateSection("\t.data.rel.local", false,
                                           SectionKind::DataRelLocal);
  DataRelROSection = getOrCreateSection("\t.data.rel.ro", false,
                                        SectionKind::ReadOnlyWithRel);
  DataRelROLocalSection =
    getOrCreateSection("\t.data.rel.ro.local", false,
                       SectionKind::ReadOnlyWithRelLocal);
    
  MergeableConst4Section = getOrCreateSection(".rodata.cst4", false,
                                              SectionKind::MergeableConst4);
  MergeableConst8Section = getOrCreateSection(".rodata.cst8", false,
                                              SectionKind::MergeableConst8);
  MergeableConst16Section = getOrCreateSection(".rodata.cst16", false,
                                               SectionKind::MergeableConst16);
}


SectionKind::Kind TargetLoweringObjectFileELF::
getKindForNamedSection(const char *Name, SectionKind::Kind K) const {
  if (Name[0] != '.') return K;
  
  // Some lame default implementation based on some magic section names.
  if (strncmp(Name, ".gnu.linkonce.b.", 16) == 0 ||
      strncmp(Name, ".llvm.linkonce.b.", 17) == 0 ||
      strncmp(Name, ".gnu.linkonce.sb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.sb.", 18) == 0)
    return SectionKind::BSS;
  
  if (strcmp(Name, ".tdata") == 0 ||
      strncmp(Name, ".tdata.", 7) == 0 ||
      strncmp(Name, ".gnu.linkonce.td.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.td.", 18) == 0)
    return SectionKind::ThreadData;
  
  if (strcmp(Name, ".tbss") == 0 ||
      strncmp(Name, ".tbss.", 6) == 0 ||
      strncmp(Name, ".gnu.linkonce.tb.", 17) == 0 ||
      strncmp(Name, ".llvm.linkonce.tb.", 18) == 0)
    return SectionKind::ThreadBSS;
  
  return K;
}

void TargetLoweringObjectFileELF::
getSectionFlagsAsString(SectionKind Kind, SmallVectorImpl<char> &Str) const {
  Str.push_back(',');
  Str.push_back('"');
  
  if (!Kind.isMetadata())
    Str.push_back('a');
  if (Kind.isText())
    Str.push_back('x');
  if (Kind.isWriteable())
    Str.push_back('w');
  if (Kind.isMergeableCString() ||
      Kind.isMergeableConst4() ||
      Kind.isMergeableConst8() ||
      Kind.isMergeableConst16())
    Str.push_back('M');
  if (Kind.isMergeableCString())
    Str.push_back('S');
  if (Kind.isThreadLocal())
    Str.push_back('T');
  
  Str.push_back('"');
  Str.push_back(',');
  
  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (AtIsCommentChar)
    Str.push_back('%');
  else
    Str.push_back('@');
  
  const char *KindStr;
  if (Kind.isBSS() || Kind.isThreadBSS())
    KindStr = "nobits";
  else
    KindStr = "progbits";
  
  Str.append(KindStr, KindStr+strlen(KindStr));
  
  if (Kind.isMergeableCString()) {
    // TODO: Eventually handle multiple byte character strings.  For now, all
    // mergable C strings are single byte.
    Str.push_back(',');
    Str.push_back('1');
  } else if (Kind.isMergeableConst4()) {
    Str.push_back(',');
    Str.push_back('4');
  } else if (Kind.isMergeableConst8()) {
    Str.push_back(',');
    Str.push_back('8');
  } else if (Kind.isMergeableConst16()) {
    Str.push_back(',');
    Str.push_back('1');
    Str.push_back('6');
  }
}


static const char *getSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())                 return ".gnu.linkonce.t.";
  if (Kind.isReadOnly())             return ".gnu.linkonce.r.";
  
  if (Kind.isThreadData())           return ".gnu.linkonce.td.";
  if (Kind.isThreadBSS())            return ".gnu.linkonce.tb.";
  
  if (Kind.isBSS())                  return ".gnu.linkonce.b.";
  if (Kind.isDataNoRel())            return ".gnu.linkonce.d.";
  if (Kind.isDataRelLocal())         return ".gnu.linkonce.d.rel.local.";
  if (Kind.isDataRel())              return ".gnu.linkonce.d.rel.";
  if (Kind.isReadOnlyWithRelLocal()) return ".gnu.linkonce.d.rel.ro.local.";
  
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".gnu.linkonce.d.rel.ro.";
}

const MCSection *TargetLoweringObjectFileELF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  
  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (Kind.isWeak()) {
    const char *Prefix = getSectionPrefixForUniqueGlobal(Kind);
    std::string Name = Mang->makeNameProper(GV->getNameStr());
    return getOrCreateSection((Prefix+Name).c_str(), false, Kind.getKind());
  }
  
  if (Kind.isText()) return TextSection;
  
  if (Kind.isMergeableCString()) {
   assert(CStringSection_ && "Should have string section prefix");
    
    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the
    // alignment of the global!
    unsigned Align = 
      TM.getTargetData()->getPreferredAlignment(cast<GlobalVariable>(GV));
    
    std::string Name = CStringSection_->getName() + "1." + utostr(Align);
    return getOrCreateSection(Name.c_str(), false,
                              SectionKind::MergeableCString);
  }
  
  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return MergeableConst4Section;
    if (Kind.isMergeableConst8())
      return MergeableConst8Section;
    if (Kind.isMergeableConst16())
      return MergeableConst16Section;
    return ReadOnlySection;  // .const
  }
  
  if (Kind.isReadOnly())             return ReadOnlySection;
  
  if (Kind.isThreadData())           return TLSDataSection;
  if (Kind.isThreadBSS())            return TLSBSSSection;
  
  if (Kind.isBSS())                  return BSSSection_;
  
  if (Kind.isDataNoRel())            return DataSection;
  if (Kind.isDataRelLocal())         return DataRelLocalSection;
  if (Kind.isDataRel())              return DataRelSection;
  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;
  
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

/// getSectionForMergeableConstant - Given a mergeable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const MCSection *TargetLoweringObjectFileELF::
getSectionForMergeableConstant(SectionKind Kind) const {
  if (Kind.isMergeableConst4())
    return MergeableConst4Section;
  if (Kind.isMergeableConst8())
    return MergeableConst8Section;
  if (Kind.isMergeableConst16())
    return MergeableConst16Section;
  if (Kind.isReadOnly())
    return ReadOnlySection;
  
  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

//===----------------------------------------------------------------------===//
//                                 MachO
//===----------------------------------------------------------------------===//

void TargetLoweringObjectFileMachO::Initialize(MCContext &Ctx,
                                               const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  TextSection = getOrCreateSection("\t.text", true, SectionKind::Text);
  DataSection = getOrCreateSection("\t.data", true, SectionKind::DataRel);
  
  CStringSection_ = getOrCreateSection("\t.cstring", true,
                                       SectionKind::MergeableCString);
  FourByteConstantSection = getOrCreateSection("\t.literal4\n", true,
                                               SectionKind::MergeableConst4);
  EightByteConstantSection = getOrCreateSection("\t.literal8\n", true,
                                                SectionKind::MergeableConst8);
  
  // ld_classic doesn't support .literal16 in 32-bit mode, and ld64 falls back
  // to using it in -static mode.
  if (TM.getRelocationModel() != Reloc::Static &&
      TM.getTargetData()->getPointerSize() == 32)
    SixteenByteConstantSection = 
      getOrCreateSection("\t.literal16\n", true, SectionKind::MergeableConst16);
  else
    SixteenByteConstantSection = 0;
  
  ReadOnlySection = getOrCreateSection("\t.const", true, SectionKind::ReadOnly);
  
  TextCoalSection =
  getOrCreateSection("\t__TEXT,__textcoal_nt,coalesced,pure_instructions",
                     false, SectionKind::Text);
  ConstTextCoalSection = getOrCreateSection("\t__TEXT,__const_coal,coalesced",
                                            false, SectionKind::Text);
  ConstDataCoalSection = getOrCreateSection("\t__DATA,__const_coal,coalesced",
                                            false, SectionKind::Text);
  ConstDataSection = getOrCreateSection("\t.const_data", true,
                                        SectionKind::ReadOnlyWithRel);
  DataCoalSection = getOrCreateSection("\t__DATA,__datacoal_nt,coalesced",
                                       false, SectionKind::DataRel);
}

const MCSection *TargetLoweringObjectFileMachO::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  assert(!Kind.isThreadLocal() && "Darwin doesn't support TLS");
  
  if (Kind.isText())
    return Kind.isWeak() ? TextCoalSection : TextSection;
  
  // If this is weak/linkonce, put this in a coalescable section, either in text
  // or data depending on if it is writable.
  if (Kind.isWeak()) {
    if (Kind.isReadOnly())
      return ConstTextCoalSection;
    return DataCoalSection;
  }
  
  // FIXME: Alignment check should be handled by section classifier.
  if (Kind.isMergeableCString()) {
    Constant *C = cast<GlobalVariable>(GV)->getInitializer();
    const Type *Ty = cast<ArrayType>(C->getType())->getElementType();
    const TargetData &TD = *TM.getTargetData();
    unsigned Size = TD.getTypeAllocSize(Ty);
    if (Size) {
      unsigned Align = TD.getPreferredAlignment(cast<GlobalVariable>(GV));
      if (Align <= 32)
        return CStringSection_;
    }
    
    return ReadOnlySection;
  }
  
  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return FourByteConstantSection;
    if (Kind.isMergeableConst8())
      return EightByteConstantSection;
    if (Kind.isMergeableConst16() && SixteenByteConstantSection)
      return SixteenByteConstantSection;
    return ReadOnlySection;  // .const
  }
  
  // FIXME: ROData -> const in -static mode that is relocatable but they happen
  // by the static linker.  Why not mergeable?
  if (Kind.isReadOnly())
    return ReadOnlySection;

  // If this is marked const, put it into a const section.  But if the dynamic
  // linker needs to write to it, put it in the data segment.
  if (Kind.isReadOnlyWithRel())
    return ConstDataSection;
  
  // Otherwise, just drop the variable in the normal data section.
  return DataSection;
}

const MCSection *
TargetLoweringObjectFileMachO::
getSectionForMergeableConstant(SectionKind Kind) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (Kind.isDataRel())
    return ConstDataSection;
  
  if (Kind.isMergeableConst4())
    return FourByteConstantSection;
  if (Kind.isMergeableConst8())
    return EightByteConstantSection;
  if (Kind.isMergeableConst16() && SixteenByteConstantSection)
    return SixteenByteConstantSection;
  return ReadOnlySection;  // .const
}

//===----------------------------------------------------------------------===//
//                                  COFF
//===----------------------------------------------------------------------===//

void TargetLoweringObjectFileCOFF::Initialize(MCContext &Ctx,
                                              const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  TextSection = getOrCreateSection("\t.text", true, SectionKind::Text);
  DataSection = getOrCreateSection("\t.data", true, SectionKind::DataRel);
}

void TargetLoweringObjectFileCOFF::
getSectionFlagsAsString(SectionKind Kind, SmallVectorImpl<char> &Str) const {
  // FIXME: Inefficient.
  std::string Res = ",\"";
  if (Kind.isText())
    Res += 'x';
  if (Kind.isWriteable())
    Res += 'w';
  Res += "\"";
  
  Str.append(Res.begin(), Res.end());
}

static const char *getCOFFSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text$linkonce";
  if (Kind.isWriteable())
    return ".data$linkonce";
  return ".rdata$linkonce";
}


const MCSection *TargetLoweringObjectFileCOFF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");
  
  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (Kind.isWeak()) {
    const char *Prefix = getCOFFSectionPrefixForUniqueGlobal(Kind);
    // FIXME: Use mangler interface (PR4584).
    std::string Name = Prefix+GV->getNameStr();
    return getOrCreateSection(Name.c_str(), false, Kind.getKind());
  }
  
  if (Kind.isText())
    return getTextSection();
  
  if (Kind.isBSS() && BSSSection_ != 0)
    return BSSSection_;
  
  if (Kind.isReadOnly() && ReadOnlySection != 0)
    return ReadOnlySection;
  
  return getDataSection();
}

