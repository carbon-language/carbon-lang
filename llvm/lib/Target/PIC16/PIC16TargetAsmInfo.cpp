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
  GlobalPrefix = PAN::getTagName(PAN::PREFIX_SYMBOL);
  GlobalDirective = "\tglobal\t";
  ExternDirective = "\textern\t";

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
  ROSection = new PIC16Section(getReadOnlySection());
  ExternalVarDecls = new PIC16Section(getNamedSection("ExternalVarDecls"));
  ExternalVarDefs = new PIC16Section(getNamedSection("ExternalVarDefs"));
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
  unsigned ValSize = TD->getTypeAllocSize(Ty);
 
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
    std::string name = PAN::getUdataSectionName(BSSSections.size());
    const Section *NewSection = getNamedSection (name.c_str());

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
  unsigned ValSize = TD->getTypeAllocSize(Ty);
 
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
    std::string name = PAN::getIdataSectionName(IDATASections.size());
    const Section *NewSection = getNamedSection (name.c_str());

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

// Get the section for an automatic variable of a function.
// For PIC16 they are globals only with mangled names.
const Section *
PIC16TargetAsmInfo::getSectionForAuto(const GlobalVariable *GV) const {

  const std::string name = PAN::getSectionNameForSym(GV->getName());

  // Go through all Auto Sections and assign this variable
  // to the appropriate section.
  PIC16Section *FoundAutoSec = NULL;
  for (unsigned i = 0; i < AutosSections.size(); i++) {
    if ( AutosSections[i]->S_->getName() == name) {
      FoundAutoSec = AutosSections[i];
      break;
    }
  }

  // No Auto section was found. Crate a new one.
  if (! FoundAutoSec) {
    const Section *NewSection = getNamedSection (name.c_str());

    FoundAutoSec = new PIC16Section(NewSection);

    // Add this newly created autos section to the list of AutosSections.
    AutosSections.push_back(FoundAutoSec);
  }

  // Insert the auto into this section.
  FoundAutoSec->Items.push_back(GV);

  return FoundAutoSec->S_;
}


// Override default implementation to put the true globals into
// multiple data sections if required.
const Section*
PIC16TargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV1) const {
  // We select the section based on the initializer here, so it really
  // has to be a GlobalVariable.
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(GV1); 

  if (!GV)
    return TargetAsmInfo::SelectSectionForGlobal(GV1);

  // Record Exteranl Var Decls.
  if (GV->isDeclaration()) {
    ExternalVarDecls->Items.push_back(GV);
    return ExternalVarDecls->S_;
  }
    
  assert (GV->hasInitializer() && "A def without initializer?");

  // First, if this is an automatic variable for a function, get the section
  // name for it and return.
  const std::string name = GV->getName();
  if (PAN::isLocalName(name)) {
    return getSectionForAuto(GV);
  }

  // Record Exteranl Var Defs.
  if (GV->hasExternalLinkage() || GV->hasCommonLinkage()) {
    ExternalVarDefs->Items.push_back(GV);
  }

  // See if this is an uninitialized global.
  const Constant *C = GV->getInitializer();
  if (C->isNullValue()) 
    return getBSSSectionForGlobal(GV); 

  // If this is initialized data in RAM. Put it in the correct IDATA section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE) 
    return getIDATASectionForGlobal(GV);

  // This is initialized data in rom, put it in the readonly section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE) {
    ROSection->Items.push_back(GV);
    return ROSection->S_;
  }

  // Else let the default implementation take care of it.
  return TargetAsmInfo::SelectSectionForGlobal(GV);
}

PIC16TargetAsmInfo::~PIC16TargetAsmInfo() {
  
  for (unsigned i = 0; i < BSSSections.size(); i++) {
      delete BSSSections[i]; 
  }

  for (unsigned i = 0; i < IDATASections.size(); i++) {
      delete IDATASections[i]; 
  }

  for (unsigned i = 0; i < AutosSections.size(); i++) {
      delete AutosSections[i]; 
  }

  delete ROSection;
  delete ExternalVarDecls;
  delete ExternalVarDefs;
}
