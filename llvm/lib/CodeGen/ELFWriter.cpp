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

#include "ELFWriter.h"
#include "ELFCodeEmitter.h"
#include "ELF.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <list>
using namespace llvm;

char ELFWriter::ID = 0;
/// AddELFWriter - Concrete function to add the ELF writer to the function pass
/// manager.
MachineCodeEmitter *llvm::AddELFWriter(PassManagerBase &PM,
                                       raw_ostream &O,
                                       TargetMachine &TM) {
  ELFWriter *EW = new ELFWriter(O, TM);
  PM.add(EW);
  return &EW->getMachineCodeEmitter();
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

  // Create the machine code emitter object for this target.
  MCE = new ELFCodeEmitter(*this);

  // Inital number of sections
  NumSections = 0;
}

ELFWriter::~ELFWriter() {
  delete MCE;
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
  getSection("", ELFSection::SHT_NULL, 0);

  // Start up the symbol table.  The first entry in the symtab is the null 
  // entry.
  SymbolList.push_back(ELFSym(0));

  return false;
}

void ELFWriter::EmitGlobal(GlobalVariable *GV) {

  // XXX: put local symbols *before* global ones!
  const Section *S = TAI->SectionForGlobal(GV);
  DOUT << "Section " << S->getName() << " for global " << GV->getName() << "\n";

  // If this is an external global, emit it now.  TODO: Note that it would be
  // better to ignore the symbol here and only add it to the symbol table if
  // referenced.
  if (!GV->hasInitializer()) {
    ELFSym ExternalSym(GV);
    ExternalSym.SetBind(ELFSym::STB_GLOBAL);
    ExternalSym.SetType(ELFSym::STT_NOTYPE);
    ExternalSym.SectionIdx = ELFSection::SHN_UNDEF;
    SymbolList.push_back(ExternalSym);
    return;
  }

  const TargetData *TD = TM.getTargetData();
  unsigned Align = TD->getPreferredAlignment(GV);
  Constant *CV = GV->getInitializer();
  unsigned Size = TD->getTypeAllocSize(CV->getType());

  // If this global has a zero initializer, go to .bss or common section.
  if (CV->isNullValue() || isa<UndefValue>(CV)) {
    // If this global is part of the common block, add it now.  Variables are
    // part of the common block if they are zero initialized and allowed to be
    // merged with other symbols.
    if (GV->hasLinkOnceLinkage() || GV->hasWeakLinkage() ||
        GV->hasCommonLinkage()) {
      ELFSym CommonSym(GV);
      // Value for common symbols is the alignment required.
      CommonSym.Value = Align;
      CommonSym.Size  = Size;
      CommonSym.SetBind(ELFSym::STB_GLOBAL);
      CommonSym.SetType(ELFSym::STT_OBJECT);
      CommonSym.SectionIdx = ELFSection::SHN_COMMON;
      SymbolList.push_back(CommonSym);
      getSection(S->getName(), ELFSection::SHT_NOBITS,
        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC, 1);
      return;
    }

    // Otherwise, this symbol is part of the .bss section.  Emit it now.
    // Handle alignment.  Ensure section is aligned at least as much as required
    // by this symbol.
    ELFSection &BSSSection = getBSSSection();
    BSSSection.Align = std::max(BSSSection.Align, Align);

    // Within the section, emit enough virtual padding to get us to an alignment
    // boundary.
    if (Align)
      BSSSection.Size = (BSSSection.Size + Align - 1) & ~(Align-1);

    ELFSym BSSSym(GV);
    BSSSym.Value = BSSSection.Size;
    BSSSym.Size = Size;
    BSSSym.SetType(ELFSym::STT_OBJECT);

    switch (GV->getLinkage()) {
    default:  // weak/linkonce/common handled above
      assert(0 && "Unexpected linkage type!");
    case GlobalValue::AppendingLinkage:  // FIXME: This should be improved!
    case GlobalValue::ExternalLinkage:
      BSSSym.SetBind(ELFSym::STB_GLOBAL);
      break;
    case GlobalValue::InternalLinkage:
      BSSSym.SetBind(ELFSym::STB_LOCAL);
      break;
    }

    // Set the idx of the .bss section
    BSSSym.SectionIdx = BSSSection.SectionIdx;
    if (!GV->hasPrivateLinkage())
      SymbolList.push_back(BSSSym);

    // Reserve space in the .bss section for this symbol.
    BSSSection.Size += Size;
    return;
  }

  /// Emit the Global symbol to the right ELF section
  ELFSym GblSym(GV);
  GblSym.Size = Size;
  GblSym.SetType(ELFSym::STT_OBJECT);
  GblSym.SetBind(ELFSym::STB_GLOBAL);
  unsigned Flags = S->getFlags();
  unsigned SectType = ELFSection::SHT_PROGBITS;
  unsigned SHdrFlags = ELFSection::SHF_ALLOC;

  if (Flags & SectionFlags::Code)
    SHdrFlags |= ELFSection::SHF_EXECINSTR;
  if (Flags & SectionFlags::Writeable)
    SHdrFlags |= ELFSection::SHF_WRITE;
  if (Flags & SectionFlags::Mergeable)
    SHdrFlags |= ELFSection::SHF_MERGE;
  if (Flags & SectionFlags::TLS)
    SHdrFlags |= ELFSection::SHF_TLS;
  if (Flags & SectionFlags::Strings)
    SHdrFlags |= ELFSection::SHF_STRINGS;

  // Remove tab from section name prefix
  std::string SectionName(S->getName());
  size_t Pos = SectionName.find("\t");
  if (Pos != std::string::npos)
    SectionName.erase(Pos, 1);

  // The section alignment should be bound to the element with
  // the largest alignment
  ELFSection &ElfS = getSection(SectionName, SectType, SHdrFlags);
  GblSym.SectionIdx = ElfS.SectionIdx;
  if (Align > ElfS.Align)
    ElfS.Align = Align;

  // S.Value should contain the symbol index inside the section,
  // and all symbols should start on their required alignment boundary
  GblSym.Value = (ElfS.size() + (Align-1)) & (-Align);
  ElfS.emitAlignment(Align);

  // Emit the constant symbol to its section
  EmitGlobalConstant(CV, ElfS);
  SymbolList.push_back(GblSym);
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
      assert(0 && "X86_FP80Ty global emission not implemented");
    } else if (CFP->getType() == Type::PPC_FP128Ty)
      assert(0 && "PPC_FP128Ty global emission not implemented");
    return;
  } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    if (Size == 4)
      GblS.emitWord32(CI->getZExtValue());
    else if (Size == 8)
      GblS.emitWord64(CI->getZExtValue());
    else
      assert(0 && "LargeInt global emission not implemented");
    return;
  } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    const VectorType *PTy = CP->getType();
    for (unsigned I = 0, E = PTy->getNumElements(); I < E; ++I)
      EmitGlobalConstant(CP->getOperand(I), GblS);
    return;
  }
  assert(0 && "unknown global constant");
}


bool ELFWriter::runOnMachineFunction(MachineFunction &MF) {
  // Nothing to do here, this is all done through the MCE object above.
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the ELF file to 'O'.
bool ELFWriter::doFinalization(Module &M) {
  /// FIXME: This should be removed when moving to ObjectCodeEmiter. Since the
  /// current ELFCodeEmiter uses CurrBuff, ... it doesn't update S.Data
  /// vector size for .text sections, so this is a quick dirty fix
  ELFSection &TS = getTextSection();
  if (TS.Size) {
    BinaryData &BD = TS.getData();
    for (unsigned e=0; e<TS.Size; ++e)
      BD.push_back(BD[e]);
  }

  // Emit .data section placeholder
  getDataSection();

  // Emit .bss section placeholder
  getBSSSection();

  // Build and emit data, bss and "common" sections.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I);

  // Emit non-executable stack note
  if (TAI->getNonexecutableStackDirective())
    getNonExecStackSection();

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

/// EmitSymbolTable - If the current symbol table is non-empty, emit the string
/// table for it and then the symbol table itself.
void ELFWriter::EmitSymbolTable() {
  if (SymbolList.size() == 1) return;  // Only the null entry.

  // FIXME: compact all local symbols to the start of the symtab.
  unsigned FirstNonLocalSymbol = 1;

  ELFSection &StrTab = getStringTableSection();

  // Set the zero'th symbol to a null byte, as required.
  StrTab.emitByte(0);

  unsigned Index = 1;
  for (unsigned i = 1, e = SymbolList.size(); i != e; ++i) {
    // Use the name mangler to uniquify the LLVM symbol.
    std::string Name = Mang->getValueName(SymbolList[i].GV);

    if (Name.empty()) {
      SymbolList[i].NameIdx = 0;
    } else {
      SymbolList[i].NameIdx = Index;
      StrTab.emitString(Name);

      // Keep track of the number of bytes emitted to this section.
      Index += Name.size()+1;
    }
  }
  assert(Index == StrTab.size());
  StrTab.Size = Index;

  // Now that we have emitted the string table and know the offset into the
  // string table of each symbol, emit the symbol table itself.
  ELFSection &SymTab = getSymbolTableSection();
  SymTab.Align = TEW->getSymTabAlignment();
  SymTab.Link  = StrTab.SectionIdx;      // Section Index of .strtab.
  SymTab.Info  = FirstNonLocalSymbol;    // First non-STB_LOCAL symbol.

  // Size of each symtab entry.
  SymTab.EntSize = TEW->getSymTabEntrySize();

  for (unsigned i = 0, e = SymbolList.size(); i != e; ++i)
    EmitSymbol(SymTab, SymbolList[i]);

  SymTab.Size = SymTab.size();
}

/// EmitSectionTableStringTable - This method adds and emits a section for the
/// ELF Section Table string table: the string table that holds all of the
/// section names.
void ELFWriter::EmitSectionTableStringTable() {
  // First step: add the section for the string table to the list of sections:
  ELFSection &SHStrTab = getSection(".shstrtab", ELFSection::SHT_STRTAB, 0);

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
  unsigned TableAlign = is64Bit ? 8 : 4;
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
    if (S.Align) {
      for (size_t NewFileOff = (FileOff+S.Align-1) & ~(S.Align-1);
        FileOff != NewFileOff; ++FileOff)
        O << (char)0xAB;
    }

    if (S.size()) {
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
