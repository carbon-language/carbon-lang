//===-- ELFWriter.cpp - Target-independent ELF Writer code ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent ELF writer.  This file writes out
// the ELF file in the following order:
//
//  #1. ELF Header
//  #2. '.text' section
//  #3. '.data' section
//  #4. '.bss' section  (conceptual position in file)
//  ...
//  #X. '.shstrtab' section
//  #Y. Section Table
//
// The entries in the section table are laid out as:
//  #0. Null entry [required]
//  #1. ".text" entry - the program code
//  #2. ".data" entry - global variables with initializers.     [ if needed ]
//  #3. ".bss" entry  - global variables without initializers.  [ if needed ]
//  ...
//  #N. ".shstrtab" entry - String table for the section names.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "elfwriter"

#include "ELF.h"
#include "ELFWriter.h"
#include "ELFCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/ObjectCodeEmitter.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetELFWriterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

char ELFWriter::ID = 0;

/// AddELFWriter - Add the ELF writer to the function pass manager
ObjectCodeEmitter *llvm::AddELFWriter(PassManagerBase &PM,
                                      raw_ostream &O,
                                      TargetMachine &TM) {
  ELFWriter *EW = new ELFWriter(O, TM);
  PM.add(EW);
  return EW->getObjectCodeEmitter();
}

//===----------------------------------------------------------------------===//
//                          ELFWriter Implementation
//===----------------------------------------------------------------------===//

ELFWriter::ELFWriter(raw_ostream &o, TargetMachine &tm)
  : MachineFunctionPass(&ID), O(o), TM(tm),
    is64Bit(TM.getTargetData()->getPointerSizeInBits() == 64),
    isLittleEndian(TM.getTargetData()->isLittleEndian()),
    ElfHdr(isLittleEndian, is64Bit) {

  TAI = TM.getTargetAsmInfo();
  TEW = TM.getELFWriterInfo();

  // Create the object code emitter object for this target.
  ElfCE = new ELFCodeEmitter(*this);

  // Inital number of sections
  NumSections = 0;
}

ELFWriter::~ELFWriter() {
  delete ElfCE;
}

// doInitialization - Emit the file header and all of the global variables for
// the module to the ELF file.
bool ELFWriter::doInitialization(Module &M) {
  Mang = new Mangler(M);

  // ELF Header
  // ----------
  // Fields e_shnum e_shstrndx are only known after all section have
  // been emitted. They locations in the ouput buffer are recorded so
  // to be patched up later.
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the last for *_off and *_addr elf types

  ElfHdr.emitByte(0x7f); // e_ident[EI_MAG0]
  ElfHdr.emitByte('E');  // e_ident[EI_MAG1]
  ElfHdr.emitByte('L');  // e_ident[EI_MAG2]
  ElfHdr.emitByte('F');  // e_ident[EI_MAG3]

  ElfHdr.emitByte(TEW->getEIClass()); // e_ident[EI_CLASS]
  ElfHdr.emitByte(TEW->getEIData());  // e_ident[EI_DATA]
  ElfHdr.emitByte(EV_CURRENT);        // e_ident[EI_VERSION]
  ElfHdr.emitAlignment(16);           // e_ident[EI_NIDENT-EI_PAD]

  ElfHdr.emitWord16(ET_REL);             // e_type
  ElfHdr.emitWord16(TEW->getEMachine()); // e_machine = target
  ElfHdr.emitWord32(EV_CURRENT);         // e_version
  ElfHdr.emitWord(0);                    // e_entry, no entry point in .o file
  ElfHdr.emitWord(0);                    // e_phoff, no program header for .o
  ELFHdr_e_shoff_Offset = ElfHdr.size();
  ElfHdr.emitWord(0);                    // e_shoff = sec hdr table off in bytes
  ElfHdr.emitWord32(TEW->getEFlags());   // e_flags = whatever the target wants
  ElfHdr.emitWord16(TEW->getHdrSize());  // e_ehsize = ELF header size
  ElfHdr.emitWord16(0);                  // e_phentsize = prog header entry size
  ElfHdr.emitWord16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  ElfHdr.emitWord16(TEW->getSHdrSize());

  // e_shnum     = # of section header ents
  ELFHdr_e_shnum_Offset = ElfHdr.size();
  ElfHdr.emitWord16(0); // Placeholder

  // e_shstrndx  = Section # of '.shstrtab'
  ELFHdr_e_shstrndx_Offset = ElfHdr.size();
  ElfHdr.emitWord16(0); // Placeholder

  // Add the null section, which is required to be first in the file.
  getNullSection();

  return false;
}

// getGlobalELFVisibility - Returns the ELF specific visibility type
unsigned ELFWriter::getGlobalELFVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  default:
    llvm_unreachable("unknown visibility type");
  case GlobalValue::DefaultVisibility:
    return ELFSym::STV_DEFAULT;
  case GlobalValue::HiddenVisibility:
    return ELFSym::STV_HIDDEN;
  case GlobalValue::ProtectedVisibility:
    return ELFSym::STV_PROTECTED;
  }
  return 0;
}

// getGlobalELFBinding - Returns the ELF specific binding type
unsigned ELFWriter::getGlobalELFBinding(const GlobalValue *GV) {
  if (GV->hasInternalLinkage())
    return ELFSym::STB_LOCAL;

  if (GV->hasWeakLinkage())
    return ELFSym::STB_WEAK;

  return ELFSym::STB_GLOBAL;
}

// getGlobalELFType - Returns the ELF specific type for a global
unsigned ELFWriter::getGlobalELFType(const GlobalValue *GV) {
  if (GV->isDeclaration())
    return ELFSym::STT_NOTYPE;

  if (isa<Function>(GV))
    return ELFSym::STT_FUNC;

  return ELFSym::STT_OBJECT;
}

// getElfSectionFlags - Get the ELF Section Header flags based
// on the flags defined in ELFTargetAsmInfo.
unsigned ELFWriter::getElfSectionFlags(unsigned Flags) {
  unsigned ElfSectionFlags = ELFSection::SHF_ALLOC;

  if (Flags & SectionFlags::Code)
    ElfSectionFlags |= ELFSection::SHF_EXECINSTR;
  if (Flags & SectionFlags::Writeable)
    ElfSectionFlags |= ELFSection::SHF_WRITE;
  if (Flags & SectionFlags::Mergeable)
    ElfSectionFlags |= ELFSection::SHF_MERGE;
  if (Flags & SectionFlags::TLS)
    ElfSectionFlags |= ELFSection::SHF_TLS;
  if (Flags & SectionFlags::Strings)
    ElfSectionFlags |= ELFSection::SHF_STRINGS;

  return ElfSectionFlags;
}

// isELFUndefSym - the symbol has no section and must be placed in
// the symbol table with a reference to the null section.
static bool isELFUndefSym(const GlobalValue *GV) {
  return GV->isDeclaration();
}

// isELFBssSym - for an undef or null value, the symbol must go to a bss
// section if it's not weak for linker, otherwise it's a common sym.
static bool isELFBssSym(const GlobalValue *GV) {
  return (!GV->isDeclaration() &&
          (GV->isNullValue() || isa<UndefValue>(GV)) &&
          !GV->isWeakForLinker());
}

// isELFCommonSym - for an undef or null value, the symbol must go to a
// common section if it's weak for linker, otherwise bss.
static bool isELFCommonSym(const GlobalValue *GV) {
  return (!GV->isDeclaration() &&
          (GV->isNullValue() || isa<UndefValue>(GV))
           && GV->isWeakForLinker());
}

// isELFDataSym - if the symbol is an initialized but no null constant
// it must go to some kind of data section gathered from TAI
static bool isELFDataSym(const GlobalValue *GV) {
  return (!GV->isDeclaration() &&
          !(GV->isNullValue() || isa<UndefValue>(GV)));
}

// EmitGlobal - Choose the right section for global and emit it
void ELFWriter::EmitGlobal(const GlobalValue *GV) {

  // Handle ELF Bind, Visibility and Type for the current symbol
  unsigned SymBind = getGlobalELFBinding(GV);
  ELFSym GblSym(GV);
  GblSym.setBind(SymBind);
  GblSym.setVisibility(getGlobalELFVisibility(GV));
  GblSym.setType(getGlobalELFType(GV));

  if (isELFUndefSym(GV)) {
    GblSym.SectionIdx = ELFSection::SHN_UNDEF;
  } else {
    assert(isa<GlobalVariable>(GV) && "GV not a global variable!");
    const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);

    // Get ELF section from TAI
    const Section *S = TAI->SectionForGlobal(GV);
    unsigned SectionFlags = getElfSectionFlags(S->getFlags());

    // The symbol align should update the section alignment if needed
    const TargetData *TD = TM.getTargetData();
    unsigned Align = TD->getPreferredAlignment(GVar);
    unsigned Size = TD->getTypeAllocSize(GVar->getInitializer()->getType());
    GblSym.Size = Size;

    if (isELFCommonSym(GV)) {
      GblSym.SectionIdx = ELFSection::SHN_COMMON;
      getSection(S->getName(), ELFSection::SHT_NOBITS, SectionFlags, 1);

      // A new linkonce section is created for each global in the
      // common section, the default alignment is 1 and the symbol
      // value contains its alignment.
      GblSym.Value = Align;

    } else if (isELFBssSym(GV)) {
      ELFSection &ES =
        getSection(S->getName(), ELFSection::SHT_NOBITS, SectionFlags);
      GblSym.SectionIdx = ES.SectionIdx;

      // Update the size with alignment and the next object can
      // start in the right offset in the section
      if (Align) ES.Size = (ES.Size + Align-1) & ~(Align-1);
      ES.Align = std::max(ES.Align, Align);

      // GblSym.Value should contain the virtual offset inside the section.
      // Virtual because the BSS space is not allocated on ELF objects
      GblSym.Value = ES.Size;
      ES.Size += Size;

    } else if (isELFDataSym(GV)) {
      ELFSection &ES =
        getSection(S->getName(), ELFSection::SHT_PROGBITS, SectionFlags);
      GblSym.SectionIdx = ES.SectionIdx;

      // GblSym.Value should contain the symbol offset inside the section,
      // and all symbols should start on their required alignment boundary
      ES.Align = std::max(ES.Align, Align);
      GblSym.Value = (ES.size() + (Align-1)) & (-Align);
      ES.emitAlignment(ES.Align);

      // Emit the global to the data section 'ES'
      EmitGlobalConstant(GVar->getInitializer(), ES);
    }
  }

  // Local symbols should come first on the symbol table.
  if (!GV->hasPrivateLinkage()) {
    if (SymBind == ELFSym::STB_LOCAL)
      SymbolList.push_front(GblSym);
    else
      SymbolList.push_back(GblSym);
  }
}

void ELFWriter::EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                         ELFSection &GblS) {

  // Print the fields in successive locations. Pad to align if needed!
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(CVS->getType());
  const StructLayout *cvsLayout = TD->getStructLayout(CVS->getType());
  uint64_t sizeSoFar = 0;
  for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
    const Constant* field = CVS->getOperand(i);

    // Check if padding is needed and insert one or more 0s.
    uint64_t fieldSize = TD->getTypeAllocSize(field->getType());
    uint64_t padSize = ((i == e-1 ? Size : cvsLayout->getElementOffset(i+1))
                        - cvsLayout->getElementOffset(i)) - fieldSize;
    sizeSoFar += fieldSize + padSize;

    // Now print the actual field value.
    EmitGlobalConstant(field, GblS);

    // Insert padding - this may include padding to increase the size of the
    // current field up to the ABI size (if the struct is not packed) as well
    // as padding to ensure that the next field starts at the right offset.
    for (unsigned p=0; p < padSize; p++)
      GblS.emitByte(0);
  }
  assert(sizeSoFar == cvsLayout->getSizeInBytes() &&
         "Layout of constant struct may be incorrect!");
}

void ELFWriter::EmitGlobalConstant(const Constant *CV, ELFSection &GblS) {
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypeAllocSize(CV->getType());

  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    if (CVA->isString()) {
      std::string GblStr = CVA->getAsString();
      GblStr.resize(GblStr.size()-1);
      GblS.emitString(GblStr);
    } else { // Not a string.  Print the values in successive locations
      for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
        EmitGlobalConstant(CVA->getOperand(i), GblS);
    }
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    EmitGlobalConstantStruct(CVS, GblS);
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    if (CFP->getType() == Type::DoubleTy)
      GblS.emitWord64(Val);
    else if (CFP->getType() == Type::FloatTy)
      GblS.emitWord32(Val);
    else if (CFP->getType() == Type::X86_FP80Ty) {
      llvm_unreachable("X86_FP80Ty global emission not implemented");
    } else if (CFP->getType() == Type::PPC_FP128Ty)
      llvm_unreachable("PPC_FP128Ty global emission not implemented");
    return;
  } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    if (Size == 4)
      GblS.emitWord32(CI->getZExtValue());
    else if (Size == 8)
      GblS.emitWord64(CI->getZExtValue());
    else
      llvm_unreachable("LargeInt global emission not implemented");
    return;
  } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    const VectorType *PTy = CP->getType();
    for (unsigned I = 0, E = PTy->getNumElements(); I < E; ++I)
      EmitGlobalConstant(CP->getOperand(I), GblS);
    return;
  }
  llvm_unreachable("unknown global constant");
}


bool ELFWriter::runOnMachineFunction(MachineFunction &MF) {
  // Nothing to do here, this is all done through the ElfCE object above.
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the ELF file to 'O'.
bool ELFWriter::doFinalization(Module &M) {
  // Emit .data section placeholder
  getDataSection();

  // Emit .bss section placeholder
  getBSSSection();

  // Build and emit data, bss and "common" sections.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    EmitGlobal(I);
    GblSymLookup[I] = 0;
  }

  // Emit all pending globals
  // TODO: this should be done only for referenced symbols
  for (SetVector<GlobalValue*>::const_iterator I = PendingGlobals.begin(),
       E = PendingGlobals.end(); I != E; ++I) {

    // No need to emit the symbol again
    if (GblSymLookup.find(*I) != GblSymLookup.end())
      continue;

    EmitGlobal(*I);
    GblSymLookup[*I] = 0;
  }

  // Emit non-executable stack note
  if (TAI->getNonexecutableStackDirective())
    getNonExecStackSection();

  // Emit a symbol for each section created until now
  for (std::map<std::string, ELFSection*>::iterator I = SectionLookup.begin(),
       E = SectionLookup.end(); I != E; ++I) {
    ELFSection *ES = I->second;

    // Skip null section
    if (ES->SectionIdx == 0) continue;

    ELFSym SectionSym(0);
    SectionSym.SectionIdx = ES->SectionIdx;
    SectionSym.Size = 0;
    SectionSym.setBind(ELFSym::STB_LOCAL);
    SectionSym.setType(ELFSym::STT_SECTION);
    SectionSym.setVisibility(ELFSym::STV_DEFAULT);

    // Local symbols go in the list front
    SymbolList.push_front(SectionSym);
  }

  // Emit string table
  EmitStringTable();

  // Emit the symbol table now, if non-empty.
  EmitSymbolTable();

  // Emit the relocation sections.
  EmitRelocations();

  // Emit the sections string table.
  EmitSectionTableStringTable();

  // Dump the sections and section table to the .o file.
  OutputSectionsAndSectionTable();

  // We are done with the abstract symbols.
  SectionList.clear();
  NumSections = 0;

  // Release the name mangler object.
  delete Mang; Mang = 0;
  return false;
}

/// EmitRelocations - Emit relocations
void ELFWriter::EmitRelocations() {

  // Create Relocation sections for each section which needs it.
  for (std::list<ELFSection>::iterator I = SectionList.begin(),
       E = SectionList.end(); I != E; ++I) {

    // This section does not have relocations
    if (!I->hasRelocations()) continue;

    // Get the relocation section for section 'I'
    bool HasRelA = TEW->hasRelocationAddend();
    ELFSection &RelSec = getRelocSection(I->getName(), HasRelA,
                                         TEW->getPrefELFAlignment());

    // 'Link' - Section hdr idx of the associated symbol table
    // 'Info' - Section hdr idx of the section to which the relocation applies
    ELFSection &SymTab = getSymbolTableSection();
    RelSec.Link = SymTab.SectionIdx;
    RelSec.Info = I->SectionIdx;
    RelSec.EntSize = TEW->getRelocationEntrySize();

    // Get the relocations from Section
    std::vector<MachineRelocation> Relos = I->getRelocations();
    for (std::vector<MachineRelocation>::iterator MRI = Relos.begin(),
         MRE = Relos.end(); MRI != MRE; ++MRI) {
      MachineRelocation &MR = *MRI;

      // Offset from the start of the section containing the symbol
      unsigned Offset = MR.getMachineCodeOffset();

      // Symbol index in the symbol table
      unsigned SymIdx = 0;

      // Target specific ELF relocation type
      unsigned RelType = TEW->getRelocationType(MR.getRelocationType());

      // Constant addend used to compute the value to be stored
      // into the relocatable field
      int64_t Addend = 0;

      // There are several machine relocations types, and each one of
      // them needs a different approach to retrieve the symbol table index.
      if (MR.isGlobalValue()) {
        const GlobalValue *G = MR.getGlobalValue();
        SymIdx = GblSymLookup[G];
        Addend = TEW->getAddendForRelTy(RelType);
      } else {
        unsigned SectionIdx = MR.getConstantVal();
        // TODO: use a map for this.
        for (std::list<ELFSym>::iterator I = SymbolList.begin(),
             E = SymbolList.end(); I != E; ++I)
          if ((SectionIdx == I->SectionIdx) &&
              (I->getType() == ELFSym::STT_SECTION)) {
            SymIdx = I->SymTabIdx;
            break;
          }
        Addend = (uint64_t)MR.getResultPointer();
      }

      // Get the relocation entry and emit to the relocation section
      ELFRelocation Rel(Offset, SymIdx, RelType, HasRelA, Addend);
      EmitRelocation(RelSec, Rel, HasRelA);
    }
  }
}

/// EmitRelocation - Write relocation 'Rel' to the relocation section 'Rel'
void ELFWriter::EmitRelocation(BinaryObject &RelSec, ELFRelocation &Rel,
                               bool HasRelA) {
  RelSec.emitWord(Rel.getOffset());
  RelSec.emitWord(Rel.getInfo(is64Bit));
  if (HasRelA)
    RelSec.emitWord(Rel.getAddend());
}

/// EmitSymbol - Write symbol 'Sym' to the symbol table 'SymbolTable'
void ELFWriter::EmitSymbol(BinaryObject &SymbolTable, ELFSym &Sym) {
  if (is64Bit) {
    SymbolTable.emitWord32(Sym.NameIdx);
    SymbolTable.emitByte(Sym.Info);
    SymbolTable.emitByte(Sym.Other);
    SymbolTable.emitWord16(Sym.SectionIdx);
    SymbolTable.emitWord64(Sym.Value);
    SymbolTable.emitWord64(Sym.Size);
  } else {
    SymbolTable.emitWord32(Sym.NameIdx);
    SymbolTable.emitWord32(Sym.Value);
    SymbolTable.emitWord32(Sym.Size);
    SymbolTable.emitByte(Sym.Info);
    SymbolTable.emitByte(Sym.Other);
    SymbolTable.emitWord16(Sym.SectionIdx);
  }
}

/// EmitSectionHeader - Write section 'Section' header in 'SHdrTab'
/// Section Header Table
void ELFWriter::EmitSectionHeader(BinaryObject &SHdrTab,
                                  const ELFSection &SHdr) {
  SHdrTab.emitWord32(SHdr.NameIdx);
  SHdrTab.emitWord32(SHdr.Type);
  if (is64Bit) {
    SHdrTab.emitWord64(SHdr.Flags);
    SHdrTab.emitWord(SHdr.Addr);
    SHdrTab.emitWord(SHdr.Offset);
    SHdrTab.emitWord64(SHdr.Size);
    SHdrTab.emitWord32(SHdr.Link);
    SHdrTab.emitWord32(SHdr.Info);
    SHdrTab.emitWord64(SHdr.Align);
    SHdrTab.emitWord64(SHdr.EntSize);
  } else {
    SHdrTab.emitWord32(SHdr.Flags);
    SHdrTab.emitWord(SHdr.Addr);
    SHdrTab.emitWord(SHdr.Offset);
    SHdrTab.emitWord32(SHdr.Size);
    SHdrTab.emitWord32(SHdr.Link);
    SHdrTab.emitWord32(SHdr.Info);
    SHdrTab.emitWord32(SHdr.Align);
    SHdrTab.emitWord32(SHdr.EntSize);
  }
}

/// EmitStringTable - If the current symbol table is non-empty, emit the string
/// table for it
void ELFWriter::EmitStringTable() {
  if (!SymbolList.size()) return;  // Empty symbol table.
  ELFSection &StrTab = getStringTableSection();

  // Set the zero'th symbol to a null byte, as required.
  StrTab.emitByte(0);

  // Walk on the symbol list and write symbol names into the
  // string table.
  unsigned Index = 1;
  for (std::list<ELFSym>::iterator I = SymbolList.begin(),
       E = SymbolList.end(); I != E; ++I) {

    // Use the name mangler to uniquify the LLVM symbol.
    std::string Name;
    if (I->GV) Name.append(Mang->getValueName(I->GV));

    if (Name.empty()) {
      I->NameIdx = 0;
    } else {
      I->NameIdx = Index;
      StrTab.emitString(Name);

      // Keep track of the number of bytes emitted to this section.
      Index += Name.size()+1;
    }
  }
  assert(Index == StrTab.size());
  StrTab.Size = Index;
}

/// EmitSymbolTable - Emit the symbol table itself.
void ELFWriter::EmitSymbolTable() {
  if (!SymbolList.size()) return;  // Empty symbol table.

  unsigned FirstNonLocalSymbol = 1;
  // Now that we have emitted the string table and know the offset into the
  // string table of each symbol, emit the symbol table itself.
  ELFSection &SymTab = getSymbolTableSection();
  SymTab.Align = TEW->getPrefELFAlignment();

  // Section Index of .strtab.
  SymTab.Link = getStringTableSection().SectionIdx;

  // Size of each symtab entry.
  SymTab.EntSize = TEW->getSymTabEntrySize();

  // The first entry in the symtab is the null symbol
  ELFSym NullSym = ELFSym(0);
  EmitSymbol(SymTab, NullSym);

  // Emit all the symbols to the symbol table. Skip the null
  // symbol, cause it's emitted already
  unsigned Index = 1;
  for (std::list<ELFSym>::iterator I = SymbolList.begin(),
       E = SymbolList.end(); I != E; ++I, ++Index) {
    // Keep track of the first non-local symbol
    if (I->getBind() == ELFSym::STB_LOCAL)
      FirstNonLocalSymbol++;

    // Emit symbol to the symbol table
    EmitSymbol(SymTab, *I);

    // Record the symbol table index for each global value
    if (I->GV)
      GblSymLookup[I->GV] = Index;

    // Keep track on the symbol index into the symbol table
    I->SymTabIdx = Index;
  }

  SymTab.Info = FirstNonLocalSymbol;
  SymTab.Size = SymTab.size();
}

/// EmitSectionTableStringTable - This method adds and emits a section for the
/// ELF Section Table string table: the string table that holds all of the
/// section names.
void ELFWriter::EmitSectionTableStringTable() {
  // First step: add the section for the string table to the list of sections:
  ELFSection &SHStrTab = getSectionHeaderStringTableSection();

  // Now that we know which section number is the .shstrtab section, update the
  // e_shstrndx entry in the ELF header.
  ElfHdr.fixWord16(SHStrTab.SectionIdx, ELFHdr_e_shstrndx_Offset);

  // Set the NameIdx of each section in the string table and emit the bytes for
  // the string table.
  unsigned Index = 0;

  for (std::list<ELFSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    // Set the index into the table.  Note if we have lots of entries with
    // common suffixes, we could memoize them here if we cared.
    I->NameIdx = Index;
    SHStrTab.emitString(I->getName());

    // Keep track of the number of bytes emitted to this section.
    Index += I->getName().size()+1;
  }

  // Set the size of .shstrtab now that we know what it is.
  assert(Index == SHStrTab.size());
  SHStrTab.Size = Index;
}

/// OutputSectionsAndSectionTable - Now that we have constructed the file header
/// and all of the sections, emit these to the ostream destination and emit the
/// SectionTable.
void ELFWriter::OutputSectionsAndSectionTable() {
  // Pass #1: Compute the file offset for each section.
  size_t FileOff = ElfHdr.size();   // File header first.

  // Adjust alignment of all section if needed.
  for (std::list<ELFSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {

    // Section idx 0 has 0 offset
    if (!I->SectionIdx)
      continue;

    if (!I->size()) {
      I->Offset = FileOff;
      continue;
    }

    // Update Section size
    if (!I->Size)
      I->Size = I->size();

    // Align FileOff to whatever the alignment restrictions of the section are.
    if (I->Align)
      FileOff = (FileOff+I->Align-1) & ~(I->Align-1);

    I->Offset = FileOff;
    FileOff += I->Size;
  }

  // Align Section Header.
  unsigned TableAlign = TEW->getPrefELFAlignment();
  FileOff = (FileOff+TableAlign-1) & ~(TableAlign-1);

  // Now that we know where all of the sections will be emitted, set the e_shnum
  // entry in the ELF header.
  ElfHdr.fixWord16(NumSections, ELFHdr_e_shnum_Offset);

  // Now that we know the offset in the file of the section table, update the
  // e_shoff address in the ELF header.
  ElfHdr.fixWord(FileOff, ELFHdr_e_shoff_Offset);

  // Now that we know all of the data in the file header, emit it and all of the
  // sections!
  O.write((char *)&ElfHdr.getData()[0], ElfHdr.size());
  FileOff = ElfHdr.size();

  // Section Header Table blob
  BinaryObject SHdrTable(isLittleEndian, is64Bit);

  // Emit all of sections to the file and build the section header table.
  while (!SectionList.empty()) {
    ELFSection &S = *SectionList.begin();
    DOUT << "SectionIdx: " << S.SectionIdx << ", Name: " << S.getName()
         << ", Size: " << S.Size << ", Offset: " << S.Offset
         << ", SectionData Size: " << S.size() << "\n";

    // Align FileOff to whatever the alignment restrictions of the section are.
    if (S.size()) {
      if (S.Align)  {
        for (size_t NewFileOff = (FileOff+S.Align-1) & ~(S.Align-1);
             FileOff != NewFileOff; ++FileOff)
          O << (char)0xAB;
      }
      O.write((char *)&S.getData()[0], S.Size);
      FileOff += S.Size;
    }

    EmitSectionHeader(SHdrTable, S);
    SectionList.pop_front();
  }

  // Align output for the section table.
  for (size_t NewFileOff = (FileOff+TableAlign-1) & ~(TableAlign-1);
       FileOff != NewFileOff; ++FileOff)
    O << (char)0xAB;

  // Emit the section table itself.
  O.write((char *)&SHdrTable.getData()[0], SHdrTable.size());
}
