//===-- PIC16TargetObjectFile.cpp - PIC16 object files --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PIC16TargetObjectFile.h"
#include "PIC16ISelLowering.h"
#include "PIC16TargetMachine.h"
#include "PIC16Section.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


PIC16TargetObjectFile::PIC16TargetObjectFile() {
}

PIC16TargetObjectFile::~PIC16TargetObjectFile() {
}

/// Find a pic16 section. Return null if not found. Do not create one.
PIC16Section *PIC16TargetObjectFile::
findPIC16Section(const std::string &Name) {
  /// Return if we have an already existing one.
  PIC16Section *Entry = SectionsByName[Name];
  if (Entry)
    return Entry;

  return NULL;
}


/// Find a pic16 section. If not found, create one.
PIC16Section *PIC16TargetObjectFile::
getPIC16Section(const std::string &Name, PIC16SectionType Ty, 
                const std::string &Address, int Color) const {

  /// Return if we have an already existing one.
  PIC16Section *&Entry = SectionsByName[Name];
  if (Entry)
    return Entry;


  Entry = PIC16Section::Create(Name, Ty, Address, Color, getContext());
  return Entry;
}

/// Find a standard pic16 data section. If not found, create one and keep
/// track of it by adding it to appropriate std section list.
PIC16Section *PIC16TargetObjectFile::
getPIC16DataSection(const std::string &Name, PIC16SectionType Ty, 
                    const std::string &Address, int Color) const {

  /// Return if we have an already existing one.
  PIC16Section *&Entry = SectionsByName[Name];
  if (Entry)
    return Entry;


  /// Else create a new one and add it to appropriate section list.
  Entry = PIC16Section::Create(Name, Ty, Address, Color, getContext());

  switch (Ty) {
  default: llvm_unreachable ("unknow standard section type.");
  case UDATA: UDATASections_.push_back(Entry); break;
  case IDATA: IDATASections_.push_back(Entry); break;
  case ROMDATA: ROMDATASection_ = Entry; break;
  case UDATA_SHR: SHAREDUDATASection_ = Entry; break;
  }

  return Entry;
}
    

/// Find a standard pic16 autos section. If not found, create one and keep
/// track of it by adding it to appropriate std section list.
PIC16Section *PIC16TargetObjectFile::
getPIC16AutoSection(const std::string &Name, PIC16SectionType Ty, 
                    const std::string &Address, int Color) const {

  /// Return if we have an already existing one.
  PIC16Section *&Entry = SectionsByName[Name];
  if (Entry)
    return Entry;


  /// Else create a new one and add it to appropriate section list.
  Entry = PIC16Section::Create(Name, Ty, Address, Color, getContext());

  assert (Ty == UDATA_OVR && "incorrect section type for autos");
  AUTOSections_.push_back(Entry);

  return Entry;
}
    
/// Find a pic16 user section. If not found, create one and keep
/// track of it by adding it to appropriate std section list.
PIC16Section *PIC16TargetObjectFile::
getPIC16UserSection(const std::string &Name, PIC16SectionType Ty, 
                    const std::string &Address, int Color) const {

  /// Return if we have an already existing one.
  PIC16Section *&Entry = SectionsByName[Name];
  if (Entry)
    return Entry;


  /// Else create a new one and add it to appropriate section list.
  Entry = PIC16Section::Create(Name, Ty, Address, Color, getContext());

  USERSections_.push_back(Entry);

  return Entry;
}

/// Do some standard initialization.
void PIC16TargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &tm){
  TargetLoweringObjectFile::Initialize(Ctx, tm);
  TM = &tm;
  
  ROMDATASection_ = NULL;
  SHAREDUDATASection_ = NULL;
}

/// allocateUDATA - Allocate a un-initialized global to an existing or new UDATA
/// section and return that section.
const MCSection *
PIC16TargetObjectFile::allocateUDATA(const GlobalVariable *GV) const {
  assert(GV->hasInitializer() && "This global doesn't need space");
  Constant *C = GV->getInitializer();
  assert(C->isNullValue() && "Unitialized globals has non-zero initializer");

  // Find how much space this global needs.
  const TargetData *TD = TM->getTargetData();
  const Type *Ty = C->getType(); 
  unsigned ValSize = TD->getTypeAllocSize(Ty);
 
  // Go through all UDATA Sections and assign this variable
  // to the first available section having enough space.
  PIC16Section *Found = NULL;
  for (unsigned i = 0; i < UDATASections_.size(); i++) {
    if (DataBankSize - UDATASections_[i]->getSize() >= ValSize) {
      Found = UDATASections_[i];
      break;
    }
  }

  // No UDATA section spacious enough was found. Crate a new one.
  if (!Found) {
    std::string name = PAN::getUdataSectionName(UDATASections_.size());
    Found = getPIC16DataSection(name.c_str(), UDATA);
  }
  
  // Insert the GV into this UDATA section.
  Found->Items.push_back(GV);
  Found->setSize(Found->getSize() + ValSize);
  return Found;
} 

/// allocateIDATA - allocate an initialized global into an existing
/// or new section and return that section.
const MCSection *
PIC16TargetObjectFile::allocateIDATA(const GlobalVariable *GV) const{
  assert(GV->hasInitializer() && "This global doesn't need space");
  Constant *C = GV->getInitializer();
  assert(!C->isNullValue() && "initialized globals has zero initializer");
  assert(GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE &&
         "can allocate initialized RAM data only");

  // Find how much space this global needs.
  const TargetData *TD = TM->getTargetData();
  const Type *Ty = C->getType(); 
  unsigned ValSize = TD->getTypeAllocSize(Ty);
 
  // Go through all IDATA Sections and assign this variable
  // to the first available section having enough space.
  PIC16Section *Found = NULL;
  for (unsigned i = 0; i < IDATASections_.size(); i++) {
    if (DataBankSize - IDATASections_[i]->getSize() >= ValSize) {
      Found = IDATASections_[i]; 
      break;
    }
  }

  // No IDATA section spacious enough was found. Crate a new one.
  if (!Found) {
    std::string name = PAN::getIdataSectionName(IDATASections_.size());
    Found = getPIC16DataSection(name.c_str(), IDATA);
  }
  
  // Insert the GV into this IDATA.
  Found->Items.push_back(GV);
  Found->setSize(Found->getSize() + ValSize);
  return Found;
} 

// Allocate a program memory variable into ROMDATA section.
const MCSection *
PIC16TargetObjectFile::allocateROMDATA(const GlobalVariable *GV) const {

  std::string name = PAN::getRomdataSectionName();
  PIC16Section *S = getPIC16DataSection(name.c_str(), ROMDATA);

  S->Items.push_back(GV);
  return S;
}

// Get the section for an automatic variable of a function.
// For PIC16 they are globals only with mangled names.
const MCSection *
PIC16TargetObjectFile::allocateAUTO(const GlobalVariable *GV) const {

  const std::string name = PAN::getSectionNameForSym(GV->getName());
  PIC16Section *S = getPIC16AutoSection(name.c_str());

  S->Items.push_back(GV);
  return S;
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

  assert(GV->hasInitializer() && "A def without initializer?");

  // First, if this is an automatic variable for a function, get the section
  // name for it and return.
  std::string name = GV->getName();
  if (PAN::isLocalName(name))
    return allocateAUTO(GV);

  // See if this is an uninitialized global.
  const Constant *C = GV->getInitializer();
  if (C->isNullValue()) 
    return allocateUDATA(GV);

  // If this is initialized data in RAM. Put it in the correct IDATA section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE) 
    return allocateIDATA(GV);

  // This is initialized data in rom, put it in the readonly section.
  if (GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE) 
    return allocateROMDATA(GV);

  // Else let the default implementation take care of it.
  return TargetLoweringObjectFile::SelectSectionForGlobal(GV, Kind, Mang,TM);
}




/// getExplicitSectionGlobal - Allow the target to completely override
/// section assignment of a global.
const MCSection *PIC16TargetObjectFile::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                         Mangler *Mang, const TargetMachine &TM) const {
  assert(GV->hasSection());
  
  if (const GlobalVariable *GVar = cast<GlobalVariable>(GV)) {
    std::string SectName = GVar->getSection();
    // If address for a variable is specified, get the address and create
    // section.
    // FIXME: move this attribute checking in PAN.
    std::string AddrStr = "Address=";
    if (SectName.compare(0, AddrStr.length(), AddrStr) == 0) {
      std::string SectAddr = SectName.substr(AddrStr.length());
      if (SectAddr.compare("NEAR") == 0)
        return allocateSHARED(GVar, Mang);
      else
        return allocateAtGivenAddress(GVar, SectAddr);
    }
     
    // Create the section specified with section attribute. 
    return allocateInGivenSection(GVar);
  }

  return getPIC16DataSection(GV->getSection().c_str(), UDATA);
}

const MCSection *
PIC16TargetObjectFile::allocateSHARED(const GlobalVariable *GV,
                                      Mangler *Mang) const {
  // Make sure that this is an uninitialized global.
  assert(GV->hasInitializer() && "This global doesn't need space");
  if (!GV->getInitializer()->isNullValue()) {
    // FIXME: Generate a warning in this case that near qualifier will be 
    // ignored.
    return SelectSectionForGlobal(GV, SectionKind::getDataRel(), Mang, *TM); 
  } 
  std::string Name = PAN::getSharedUDataSectionName(); 

  PIC16Section *SharedUDataSect = getPIC16DataSection(Name.c_str(), UDATA_SHR); 
  // Insert the GV into shared section.
  SharedUDataSect->Items.push_back(GV);
  return SharedUDataSect;
}


// Interface used by AsmPrinter to get a code section for a function.
const PIC16Section *
PIC16TargetObjectFile::SectionForCode(const std::string &FnName,
                                      bool isISR) const {
  const std::string &sec_name = PAN::getCodeSectionName(FnName);
  // If it is ISR, its code section starts at a specific address.
  if (isISR)
    return getPIC16Section(sec_name, CODE, PAN::getISRAddr());
  return getPIC16Section(sec_name, CODE);
}

// Interface used by AsmPrinter to get a frame section for a function.
const PIC16Section *
PIC16TargetObjectFile::SectionForFrame(const std::string &FnName) const {
  const std::string &sec_name = PAN::getFrameSectionName(FnName);
  return getPIC16Section(sec_name, UDATA_OVR);
}

// Allocate a global var in existing or new section of given name.
const MCSection *
PIC16TargetObjectFile::allocateInGivenSection(const GlobalVariable *GV) const {
  // Determine the type of section that we need to create.
  PIC16SectionType SecTy;

  // See if this is an uninitialized global.
  const Constant *C = GV->getInitializer();
  if (C->isNullValue())
    SecTy = UDATA;
  // If this is initialized data in RAM. Put it in the correct IDATA section.
  else if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE)
    SecTy = IDATA;
  // This is initialized data in rom, put it in the readonly section.
  else if (GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE) 
    SecTy = ROMDATA;
  else
    llvm_unreachable ("Could not determine section type for global");

  PIC16Section *S = getPIC16UserSection(GV->getSection().c_str(), SecTy);
  S->Items.push_back(GV);
  return S;
}

// Allocate a global var in a new absolute sections at given address.
const MCSection *
PIC16TargetObjectFile::allocateAtGivenAddress(const GlobalVariable *GV,
                                               const std::string &Addr) const {
  // Determine the type of section that we need to create.
  PIC16SectionType SecTy;

  // See if this is an uninitialized global.
  const Constant *C = GV->getInitializer();
  if (C->isNullValue())
    SecTy = UDATA;
  // If this is initialized data in RAM. Put it in the correct IDATA section.
  else if (GV->getType()->getAddressSpace() == PIC16ISD::RAM_SPACE)
    SecTy = IDATA;
  // This is initialized data in rom, put it in the readonly section.
  else if (GV->getType()->getAddressSpace() == PIC16ISD::ROM_SPACE) 
    SecTy = ROMDATA;
  else
    llvm_unreachable ("Could not determine section type for global");

  std::string Prefix = GV->getNameStr() + "." + Addr + ".";
  std::string SName = PAN::getUserSectionName(Prefix);
  PIC16Section *S = getPIC16UserSection(SName.c_str(), SecTy, Addr.c_str());
  S->Items.push_back(GV);
  return S;
}


