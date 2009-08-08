//===-- PIC16TargetObjectFile.cpp - PIC16 object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PIC16TargetObjectFile.h"
#include "PIC16Section.h"
#include "PIC16ISelLowering.h"
#include "PIC16TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
using namespace llvm;

PIC16TargetObjectFile::PIC16TargetObjectFile()
  : ExternalVarDecls(0), ExternalVarDefs(0) {
}

const MCSection *PIC16TargetObjectFile::
getPIC16Section(const char *Name, bool isDirective, SectionKind Kind) const {
  if (MCSection *S = getContext().GetSection(Name))
    return S;
  return MCSectionPIC16::Create(Name, isDirective, Kind, getContext());
}


void PIC16TargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &tm){
  TargetLoweringObjectFile::Initialize(Ctx, tm);
  TM = &tm;
  
  BSSSection = getPIC16Section("udata.# UDATA", false, 
                               SectionKind::getBSS());
  ReadOnlySection = getPIC16Section("romdata.# ROMDATA", false,
                                    SectionKind::getReadOnly());
  DataSection = getPIC16Section("idata.# IDATA", false,
                                SectionKind::getDataRel());
  
  // Need because otherwise a .text symbol is emitted by DwarfWriter
  // in BeginModule, and gpasm cribbs for that .text symbol.
  TextSection = getPIC16Section("", true, SectionKind::getText());

  ROSections.push_back(new PIC16Section(ReadOnlySection));
  
  // FIXME: I don't know what the classification of these sections really is.
  ExternalVarDecls = new PIC16Section(getPIC16Section("ExternalVarDecls",
                                                      false,
                                      SectionKind::getMetadata()));
  ExternalVarDefs = new PIC16Section(getPIC16Section("ExternalVarDefs",
                                                     false,
                                      SectionKind::getMetadata()));
}

const MCSection *PIC16TargetObjectFile::
getSectionForFunction(const std::string &FnName) const {
  std::string T = PAN::getCodeSectionName(FnName);
  return getPIC16Section(T.c_str(), false, SectionKind::getText());
}


const MCSection *PIC16TargetObjectFile::
getSectionForFunctionFrame(const std::string &FnName) const {
  std::string T = PAN::getFrameSectionName(FnName);
  return getPIC16Section(T.c_str(), false, SectionKind::getDataRel());
}

const MCSection *
PIC16TargetObjectFile::getBSSSectionForGlobal(const GlobalVariable *GV) const {
  assert(GV->hasInitializer() && "This global doesn't need space");
  Constant *C = GV->getInitializer();
  assert(C->isNullValue() && "Unitialized globals has non-zero initializer");

  // Find how much space this global needs.
  const TargetData *TD = TM->getTargetData();
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
  if (!FoundBSS) {
    std::string name = PAN::getUdataSectionName(BSSSections.size());
    const MCSection *NewSection = getPIC16Section(name.c_str(), false,
                                                  // FIXME.
                                                  SectionKind::getMetadata());

    FoundBSS = new PIC16Section(NewSection);

    // Add this newly created BSS section to the list of BSSSections.
    BSSSections.push_back(FoundBSS);
  }
  
  // Insert the GV into this BSS.
  FoundBSS->Items.push_back(GV);
  FoundBSS->Size += ValSize;
  return FoundBSS->S_;
} 

const MCSection *
PIC16TargetObjectFile::getIDATASectionForGlobal(const GlobalVariable *GV) const{
  assert(GV->hasInitializer() && "This global doesn't need space");
  Constant *C = GV->getInitializer();
  assert(!C->isNullValue() && "initialized globals has zero initializer");
  assert(GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE &&
         "can split initialized RAM data only");

  // Find how much space this global needs.
  const TargetData *TD = TM->getTargetData();
  const Type *Ty = C->getType(); 
  unsigned ValSize = TD->getTypeAllocSize(Ty);
 
  // Go through all IDATA Sections and assign this variable
  // to the first available section having enough space.
  PIC16Section *FoundIDATA = NULL;
  for (unsigned i = 0; i < IDATASections.size(); i++) {
    if (DataBankSize - IDATASections[i]->Size >= ValSize) {
      FoundIDATA = IDATASections[i]; 
      break;
    }
  }

  // No IDATA section spacious enough was found. Crate a new one.
  if (!FoundIDATA) {
    std::string name = PAN::getIdataSectionName(IDATASections.size());
    const MCSection *NewSection = getPIC16Section(name.c_str(), false,
                                                  // FIXME.
                                                  SectionKind::getMetadata());

    FoundIDATA = new PIC16Section(NewSection);

    // Add this newly created IDATA section to the list of IDATASections.
    IDATASections.push_back(FoundIDATA);
  }
  
  // Insert the GV into this IDATA.
  FoundIDATA->Items.push_back(GV);
  FoundIDATA->Size += ValSize;
  return FoundIDATA->S_;
} 

// Get the section for an automatic variable of a function.
// For PIC16 they are globals only with mangled names.
const MCSection *
PIC16TargetObjectFile::getSectionForAuto(const GlobalVariable *GV) const {

  const std::string name = PAN::getSectionNameForSym(GV->getName());

  // Go through all Auto Sections and assign this variable
  // to the appropriate section.
  PIC16Section *FoundAutoSec = NULL;
  for (unsigned i = 0; i < AutosSections.size(); i++) {
    if (AutosSections[i]->S_->getName() == name) {
      FoundAutoSec = AutosSections[i];
      break;
    }
  }

  // No Auto section was found. Crate a new one.
  if (!FoundAutoSec) {
    const MCSection *NewSection = getPIC16Section(name.c_str(),
                                                  // FIXME.
                                                  false,
                                       SectionKind::getMetadata());

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
const MCSection *
PIC16TargetObjectFile::SelectSectionForGlobal(const GlobalValue *GV1,
                                              SectionKind Kind,
                                              Mangler *Mang,
                                              const TargetMachine &TM) const {
  // We select the section based on the initializer here, so it really
  // has to be a GlobalVariable.
  const GlobalVariable *GV = dyn_cast<GlobalVariable>(GV1); 
  if (!GV)
    return TargetLoweringObjectFile::SelectSectionForGlobal(GV1, Kind, Mang,TM);

  // Record External Var Decls.
  if (GV->isDeclaration()) {
    ExternalVarDecls->Items.push_back(GV);
    return ExternalVarDecls->S_;
  }
    
  assert(GV->hasInitializer() && "A def without initializer?");

  // First, if this is an automatic variable for a function, get the section
  // name for it and return.
  std::string name = GV->getName();
  if (PAN::isLocalName(name))
    return getSectionForAuto(GV);

  // Record Exteranl Var Defs.
  if (GV->hasExternalLinkage() || GV->hasCommonLinkage())
    ExternalVarDefs->Items.push_back(GV);

  // See if this is an uninitialized global.
  const Constant *C = GV->getInitializer();
  if (C->isNullValue()) 
    return getBSSSectionForGlobal(GV); 

  // If this is initialized data in RAM. Put it in the correct IDATA section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE) 
    return getIDATASectionForGlobal(GV);

  // This is initialized data in rom, put it in the readonly section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE) 
    return getROSectionForGlobal(GV);

  // Else let the default implementation take care of it.
  return TargetLoweringObjectFile::SelectSectionForGlobal(GV, Kind, Mang,TM);
}

PIC16TargetObjectFile::~PIC16TargetObjectFile() {
  for (unsigned i = 0; i < BSSSections.size(); i++)
    delete BSSSections[i]; 
  for (unsigned i = 0; i < IDATASections.size(); i++)
    delete IDATASections[i]; 
  for (unsigned i = 0; i < AutosSections.size(); i++)
    delete AutosSections[i]; 
  for (unsigned i = 0; i < ROSections.size(); i++)
    delete ROSections[i];
  delete ExternalVarDecls;
  delete ExternalVarDefs;
}


/// getSpecialCasedSectionGlobals - Allow the target to completely override
/// section assignment of a global.
const MCSection *PIC16TargetObjectFile::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                         Mangler *Mang, const TargetMachine &TM) const {
  assert(GV->hasSection());
  
  if (const GlobalVariable *GVar = cast<GlobalVariable>(GV)) {
    std::string SectName = GVar->getSection();
    // If address for a variable is specified, get the address and create
    // section.
    std::string AddrStr = "Address=";
    if (SectName.compare(0, AddrStr.length(), AddrStr) == 0) {
      std::string SectAddr = SectName.substr(AddrStr.length());
      return CreateSectionForGlobal(GVar, Mang, SectAddr);
    }
     
    // Create the section specified with section attribute. 
    return CreateSectionForGlobal(GVar, Mang);
  }

  return getPIC16Section(GV->getSection().c_str(), false, Kind);
}

// Create a new section for global variable. If Addr is given then create
// section at that address else create by name.
const MCSection *
PIC16TargetObjectFile::CreateSectionForGlobal(const GlobalVariable *GV,
                                              Mangler *Mang,
                                              const std::string &Addr) const {
  // See if this is an uninitialized global.
  const Constant *C = GV->getInitializer();
  if (C->isNullValue())
    return CreateBSSSectionForGlobal(GV, Addr);

  // If this is initialized data in RAM. Put it in the correct IDATA section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE)
    return CreateIDATASectionForGlobal(GV, Addr);

  // This is initialized data in rom, put it in the readonly section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE) 
    return CreateROSectionForGlobal(GV, Addr);

  // Else let the default implementation take care of it.
  return TargetLoweringObjectFile::SectionForGlobal(GV, Mang, *TM);
}

// Create uninitialized section for a variable.
const MCSection *
PIC16TargetObjectFile::CreateBSSSectionForGlobal(const GlobalVariable *GV,
                                                 std::string Addr) const {
  assert(GV->hasInitializer() && "This global doesn't need space");
  assert(GV->getInitializer()->isNullValue() &&
         "Unitialized global has non-zero initializer");
  std::string Name;
  // If address is given then create a section at that address else create a
  // section by section name specified in GV.
  PIC16Section *FoundBSS = NULL;
  if (Addr.empty()) { 
    Name = GV->getSection() + " UDATA";
    for (unsigned i = 0; i < BSSSections.size(); i++) {
      if (BSSSections[i]->S_->getName() == Name) {
        FoundBSS = BSSSections[i];
        break;
      }
    }
  } else {
    std::string Prefix = GV->getNameStr() + "." + Addr + ".";
    Name = PAN::getUdataSectionName(BSSSections.size(), Prefix) + " " + Addr;
  }
  
  PIC16Section *NewBSS = FoundBSS;
  if (NewBSS == NULL) {
    const MCSection *NewSection = getPIC16Section(Name.c_str(), false,
                                                  SectionKind::getBSS());
    NewBSS = new PIC16Section(NewSection);
    BSSSections.push_back(NewBSS);
  }

  // Insert the GV into this BSS.
  NewBSS->Items.push_back(GV);

  // We do not want to put any  GV without explicit section into this section
  // so set its size to DatabankSize.
  NewBSS->Size = DataBankSize;
  return NewBSS->S_;
}

// Get rom section for a variable. Currently there can be only one rom section
// unless a variable explicitly requests a section.
const MCSection *
PIC16TargetObjectFile::getROSectionForGlobal(const GlobalVariable *GV) const {
  ROSections[0]->Items.push_back(GV);
  return ROSections[0]->S_;
}

// Create initialized data section for a variable.
const MCSection *
PIC16TargetObjectFile::CreateIDATASectionForGlobal(const GlobalVariable *GV,
                                                   std::string Addr) const {
  assert(GV->hasInitializer() && "This global doesn't need space");
  assert(!GV->getInitializer()->isNullValue() &&
         "initialized global has zero initializer");
  assert(GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE &&
         "can be used for initialized RAM data only");

  std::string Name;
  // If address is given then create a section at that address else create a
  // section by section name specified in GV.
  PIC16Section *FoundIDATASec = NULL;
  if (Addr.empty()) {
    Name = GV->getSection() + " IDATA";
    for (unsigned i = 0; i < IDATASections.size(); i++) {
      if (IDATASections[i]->S_->getName() == Name) {
        FoundIDATASec = IDATASections[i];
        break;
      }
    }
  } else {
    std::string Prefix = GV->getNameStr() + "." + Addr + ".";
    Name = PAN::getIdataSectionName(IDATASections.size(), Prefix) + " " + Addr;
  }

  PIC16Section *NewIDATASec = FoundIDATASec;
  if (NewIDATASec == NULL) {
    const MCSection *NewSection = getPIC16Section(Name.c_str(), false,
                                                  // FIXME:
                                                  SectionKind::getMetadata());
    NewIDATASec = new PIC16Section(NewSection);
    IDATASections.push_back(NewIDATASec);
  }
  // Insert the GV into this IDATA Section.
  NewIDATASec->Items.push_back(GV);
  // We do not want to put any  GV without explicit section into this section 
  // so set its size to DatabankSize.
  NewIDATASec->Size = DataBankSize;
  return NewIDATASec->S_;
}

// Create a section in rom for a variable.
const MCSection *
PIC16TargetObjectFile::CreateROSectionForGlobal(const GlobalVariable *GV,
                                                std::string Addr) const {
  assert(GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE &&
         "can be used for ROM data only");

  std::string Name;
  // If address is given then create a section at that address else create a
  // section by section name specified in GV.
  PIC16Section *FoundROSec = NULL;
  if (Addr.empty()) {
    Name = GV->getSection() + " ROMDATA";
    for (unsigned i = 1; i < ROSections.size(); i++) {
      if (ROSections[i]->S_->getName() == Name) {
        FoundROSec = ROSections[i];
        break;
      }
    }
  } else {
    std::string Prefix = GV->getNameStr() + "." + Addr + ".";
    Name = PAN::getRomdataSectionName(ROSections.size(), Prefix) + " " + Addr;
  }

  PIC16Section *NewRomSec = FoundROSec;
  if (NewRomSec == NULL) {
    const MCSection *NewSection = getPIC16Section(Name.c_str(), false,
                                                  SectionKind::getReadOnly());
    NewRomSec = new PIC16Section(NewSection);
    ROSections.push_back(NewRomSec);
  }

  // Insert the GV into this ROM Section.
  NewRomSec->Items.push_back(GV);
  return NewRomSec->S_;
}

