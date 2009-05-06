//===-- PIC16TargetAsmInfo.cpp - PIC16 asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the PIC16TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PIC16TargetAsmInfo.h"
#include "PIC16TargetMachine.h"
#include "llvm/GlobalValue.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"

using namespace llvm;

PIC16TargetAsmInfo::
PIC16TargetAsmInfo(const PIC16TargetMachine &TM) 
  : TargetAsmInfo(TM) {
  CommentString = ";";
  Data8bitsDirective = " db ";
  Data16bitsDirective = " dw ";
  Data32bitsDirective = " dl ";
  RomData8bitsDirective = " dw ";
  RomData16bitsDirective = " rom_di ";
  RomData32bitsDirective = " rom_dl ";
  ZeroDirective = NULL;
  AsciiDirective = " dt ";
  AscizDirective = NULL;
  BSSSection_  = getNamedSection("udata.# UDATA",
                              SectionFlags::Writeable | SectionFlags::BSS);
  ReadOnlySection = getNamedSection("romdata.# ROMDATA", SectionFlags::None);
  DataSection = getNamedSection("idata.# IDATA", SectionFlags::Writeable);
  SwitchToSectionDirective = "";
  // Need because otherwise a .text symbol is emitted by DwarfWriter
  // in BeginModule, and gpasm cribbs for that .text symbol.
  TextSection = getUnnamedSection("", SectionFlags::Code);
}

const char *PIC16TargetAsmInfo::getRomDirective(unsigned size) const
{
  if (size == 8)
    return RomData8bitsDirective;
  else if (size == 16)
    return RomData16bitsDirective;
  else if (size == 32)
    return RomData32bitsDirective;
  else
    return NULL;
}


const char *PIC16TargetAsmInfo::getASDirective(unsigned size, 
                                               unsigned AS) const {
  if (AS == PIC16ISD::ROM_SPACE)
    return getRomDirective(size);
  else
    return NULL;
}

const Section *
PIC16TargetAsmInfo::getBSSSectionForGlobal(const GlobalVariable *GV) const {
  assert (GV->hasInitializer() && "This global doesn't need space");
  Constant *C = GV->getInitializer();
  assert (C->isNullValue() && "Unitialized globals has non-zero initializer");

  // Find how much space this global needs.
  const TargetData *TD = TM.getTargetData();
  const Type *Ty = C->getType(); 
  unsigned ValSize = TD->getTypePaddedSize(Ty);
 
  // Go through all BSS Sections and assign this variable
  // to the first available section having enough space.
  PIC16Section *FoundBSS = NULL;
  for (unsigned i = 0; i < BSSSections.size(); i++) {
    if (DataBankSize - BSSSections[i]->Size >= ValSize) {
      FoundBSS = BSSSections[i];
      break;
    }
  }

  // No BSS section spacious enough was found. Crate a new one.
  if (! FoundBSS) {
    char *name = new char[32];
    sprintf (name, "udata.%d.# UDATA", BSSSections.size());
    const Section *NewSection = getNamedSection (name);

    FoundBSS = new PIC16Section(NewSection);

    // Add this newly created BSS section to the list of BSSSections.
    BSSSections.push_back(FoundBSS);
  }
  
  // Insert the GV into this BSS.
  FoundBSS->Items.push_back(GV);
  FoundBSS->Size += ValSize;

  // We can't do this here because GV is const .
  // const std::string SName = FoundBSS->S_->getName();
  // GV->setSection(SName);

  return FoundBSS->S_;
} 

const Section *
PIC16TargetAsmInfo::getIDATASectionForGlobal(const GlobalVariable *GV) const {
  assert (GV->hasInitializer() && "This global doesn't need space");
  Constant *C = GV->getInitializer();
  assert (!C->isNullValue() && "initialized globals has zero initializer");
  assert (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE &&
          "can split initialized RAM data only");

  // Find how much space this global needs.
  const TargetData *TD = TM.getTargetData();
  const Type *Ty = C->getType(); 
  unsigned ValSize = TD->getTypePaddedSize(Ty);
 
  // Go through all IDATA Sections and assign this variable
  // to the first available section having enough space.
  PIC16Section *FoundIDATA = NULL;
  for (unsigned i = 0; i < IDATASections.size(); i++) {
    if ( DataBankSize - IDATASections[i]->Size >= ValSize) {
      FoundIDATA = IDATASections[i]; 
      break;
    }
  }

  // No IDATA section spacious enough was found. Crate a new one.
  if (! FoundIDATA) {
    char *name = new char[32];
    sprintf (name, "idata.%d.# IDATA", IDATASections.size());
    const Section *NewSection = getNamedSection (name);

    FoundIDATA = new PIC16Section(NewSection);

    // Add this newly created IDATA section to the list of IDATASections.
    IDATASections.push_back(FoundIDATA);
  }
  
  // Insert the GV into this IDATA.
  FoundIDATA->Items.push_back(GV);
  FoundIDATA->Size += ValSize;

  // We can't do this here because GV is const .
  // GV->setSection(FoundIDATA->S->getName());

  return FoundIDATA->S_;
} 

// Override default implementation to put the true globals into
// multiple data sections if required.
const Section*
PIC16TargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV1) const {
  // We select the section based on the initializer here, so it really
  // has to be a GlobalVariable.
  if (!isa<GlobalVariable>(GV1))
    return TargetAsmInfo::SelectSectionForGlobal(GV1);

  const GlobalVariable *GV = dyn_cast<GlobalVariable>(GV1); 
  // We are only dealing with true globals here. So names with a "."
  // are local globals. Also declarations are not entertained.
  std::string name = GV->getName();
  if (name.find(".auto.") != std::string::npos
      || name.find(".arg.") != std::string::npos || !GV->hasInitializer())
    return TargetAsmInfo::SelectSectionForGlobal(GV);

  const Constant *C = GV->getInitializer();
  // See if this is an uninitialized global.
  if (C->isNullValue()) 
    return getBSSSectionForGlobal(GV); 

  // This is initialized data. We only deal with initialized data in RAM.
  if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE) 
    return getIDATASectionForGlobal(GV);

  // Else let the default implementation take care of it.
  return TargetAsmInfo::SelectSectionForGlobal(GV);
}

void PIC16TargetAsmInfo::SetSectionForGVs(Module &M) {
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer())   // External global require no code.
      continue;

    // Any variables reaching here with "." in its name is a local scope
    // variable and should not be printed in global data section.
    std::string name = I->getName();
    if (name.find(".auto.") != std::string::npos
      || name.find(".arg.") != std::string::npos)
      continue;
    int AddrSpace = I->getType()->getAddressSpace();

    if (AddrSpace == PIC16ISD::RAM_SPACE)
      I->setSection(SectionForGlobal(I)->getName());
  }
}


// Helper routine.
// Func name starts after prefix and followed by a .
static std::string getFuncNameForSym(const std::string &Sym, 
                                      PIC16ABINames::IDs PrefixType) {

  const char *prefix = getIDName (PIC16ABINames::PREFIX_SYMBOL);

  // This name may or may not start with prefix;
  // Func names start after prfix in that case.
  size_t func_name_start = 0;
  if (Sym.find(prefix, 0, strlen(prefix)) != std::string::npos)
    func_name_start = strlen(prefix);

  // Position of the . after func name.
  size_t func_name_end = Sym.find ('.', func_name_start);

  return Sym.substr (func_name_start, func_name_end);
}

// Helper routine to create a section name given the section prefix
// and func name.
static std::string
getSectionNameForFunc (const std::string &Fname,
                       const PIC16ABINames::IDs sec_id) {
  std::string sec_id_string = getIDName (sec_id);
  return sec_id_string + "." + Fname + ".#";
}


// Get the section for the given external symbol names.
// This function is meant for only mangled external symbol names.
std::string 
llvm::getSectionNameForSym(const std::string &Sym) {
  std::string SectionName;

  PIC16ABINames::IDs id = getID (Sym);
  std::string Fname = getFuncNameForSym (Sym, id);

  switch (id) {
    default : assert (0 && "Could not determine external symbol type");
    case PIC16ABINames::FUNC_FRAME: 
    case PIC16ABINames::FUNC_RET: 
    case PIC16ABINames::FUNC_TEMPS: 
    case PIC16ABINames::FUNC_ARGS:  {
      return getSectionNameForFunc (Fname, PIC16ABINames::FRAME_SECTION);
    }
    case PIC16ABINames::FUNC_AUTOS: { 
      return getSectionNameForFunc (Fname, PIC16ABINames::AUTOS_SECTION);
    }
  }
}

PIC16TargetAsmInfo::~PIC16TargetAsmInfo() {
  
  for (unsigned i = 0; i < BSSSections.size(); i++) {
      delete BSSSections[i]; 
  }

  for (unsigned i = 0; i < IDATASections.size(); i++) {
      delete IDATASections[i]; 
  }
}
