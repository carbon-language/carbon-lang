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
// NOTE: This code should eventually be extended to support 64-bit ELF (this
// won't be hard), but we haven't done so yet!
//
//===----------------------------------------------------------------------===//

#include "ELFWriter.h"
#include "ELFCodeEmitter.h"
#include "ELF.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetELFWriterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/OutputBuffer.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
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
  : MachineFunctionPass(&ID), O(o), TM(tm) {
  e_flags = 0;  // e_flags defaults to 0, no flags.
  e_machine = TM.getELFWriterInfo()->getEMachine();

  is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
  isLittleEndian = TM.getTargetData()->isLittleEndian();

  // Create the machine code emitter object for this target.
  MCE = new ELFCodeEmitter(*this);
  NumSections = 0;
}

ELFWriter::~ELFWriter() {
  delete MCE;
}

// doInitialization - Emit the file header and all of the global variables for
// the module to the ELF file.
bool ELFWriter::doInitialization(Module &M) {
  Mang = new Mangler(M);

  // Local alias to shortenify coming code.
  std::vector<unsigned char> &FH = FileHeader;
  OutputBuffer FHOut(FH, is64Bit, isLittleEndian);

  unsigned ElfClass = is64Bit ? ELFCLASS64 : ELFCLASS32;
  unsigned ElfEndian = isLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;

  // ELF Header
  // ----------
  // Fields e_shnum e_shstrndx are only known after all section have
  // been emitted. They locations in the ouput buffer are recorded so
  // to be patched up later.
  //
  // Note
  // ----
  // FHOut.outaddr method behaves differently for ELF32 and ELF64 writing
  // 4 bytes in the former and 8 in the last for *_off and *_addr elf types

  FHOut.outbyte(0x7f); // e_ident[EI_MAG0]
  FHOut.outbyte('E');  // e_ident[EI_MAG1]
  FHOut.outbyte('L');  // e_ident[EI_MAG2]
  FHOut.outbyte('F');  // e_ident[EI_MAG3]

  FHOut.outbyte(ElfClass);   // e_ident[EI_CLASS]
  FHOut.outbyte(ElfEndian);  // e_ident[EI_DATA]
  FHOut.outbyte(EV_CURRENT); // e_ident[EI_VERSION]

  FH.resize(16);  // e_ident[EI_NIDENT-EI_PAD]

  FHOut.outhalf(ET_REL);     // e_type
  FHOut.outhalf(e_machine);  // e_machine = target
  FHOut.outword(EV_CURRENT); // e_version
  FHOut.outaddr(0);          // e_entry = 0 -> no entry point in .o file
  FHOut.outaddr(0);          // e_phoff = 0 -> no program header for .o

  ELFHdr_e_shoff_Offset = FH.size();
  FHOut.outaddr(0);                 // e_shoff = sec hdr table off in bytes
  FHOut.outword(e_flags);           // e_flags = whatever the target wants

  FHOut.outhalf(is64Bit ? 64 : 52); // e_ehsize = ELF header size
  FHOut.outhalf(0);                 // e_phentsize = prog header entry size
  FHOut.outhalf(0);                 // e_phnum     = # prog header entries = 0
  FHOut.outhalf(is64Bit ? 64 : 40); // e_shentsize = sect hdr entry size

  // e_shnum     = # of section header ents
  ELFHdr_e_shnum_Offset = FH.size();
  FHOut.outhalf(0);

  // e_shstrndx  = Section # of '.shstrtab'
  ELFHdr_e_shstrndx_Offset = FH.size();
  FHOut.outhalf(0);

  // Add the null section, which is required to be first in the file.
  getSection("", ELFSection::SHT_NULL, 0);

  // Start up the symbol table.  The first entry in the symtab is the null
  // entry.
  SymbolTable.push_back(ELFSym(0));

  return false;
}

void ELFWriter::EmitGlobal(GlobalVariable *GV) {
  // If this is an external global, emit it now.  TODO: Note that it would be
  // better to ignore the symbol here and only add it to the symbol table if
  // referenced.
  if (!GV->hasInitializer()) {
    ELFSym ExternalSym(GV);
    ExternalSym.SetBind(ELFSym::STB_GLOBAL);
    ExternalSym.SetType(ELFSym::STT_NOTYPE);
    ExternalSym.SectionIdx = ELFSection::SHN_UNDEF;
    SymbolTable.push_back(ExternalSym);
    return;
  }

  unsigned Align = TM.getTargetData()->getPreferredAlignment(GV);
  unsigned Size  =
    TM.getTargetData()->getTypeAllocSize(GV->getType()->getElementType());

  // If this global has a zero initializer, it is part of the .bss or common
  // section.
  if (GV->getInitializer()->isNullValue()) {
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
      // TODO SOMEDAY: add ELF visibility.
      CommonSym.SectionIdx = ELFSection::SHN_COMMON;
      SymbolTable.push_back(CommonSym);
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
      SymbolTable.push_back(BSSSym);

    // Reserve space in the .bss section for this symbol.
    BSSSection.Size += Size;
    return;
  }

  // FIXME: handle .rodata
  //assert(!GV->isConstant() && "unimp");

  // FIXME: handle .data
  //assert(0 && "unimp");
}


bool ELFWriter::runOnMachineFunction(MachineFunction &MF) {
  // Nothing to do here, this is all done through the MCE object above.
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the ELF file to 'O'.
bool ELFWriter::doFinalization(Module &M) {
  // Okay, the ELF header and .text sections have been completed, build the
  // .data, .bss, and "common" sections next.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I);

  // Emit the symbol table now, if non-empty.
  EmitSymbolTable();

  // FIXME: Emit the relocations now.

  // Emit the string table for the sections in the ELF file we have.
  EmitSectionTableStringTable();

  // Emit the sections to the .o file, and emit the section table for the file.
  OutputSectionsAndSectionTable();

  // We are done with the abstract symbols.
  SectionList.clear();
  NumSections = 0;

  // Release the name mangler object.
  delete Mang; Mang = 0;
  return false;
}

/// EmitSymbolTable - If the current symbol table is non-empty, emit the string
/// table for it and then the symbol table itself.
void ELFWriter::EmitSymbolTable() {
  if (SymbolTable.size() == 1) return;  // Only the null entry.

  // FIXME: compact all local symbols to the start of the symtab.
  unsigned FirstNonLocalSymbol = 1;

  ELFSection &StrTab = getSection(".strtab", ELFSection::SHT_STRTAB, 0);
  StrTab.Align = 1;

  DataBuffer &StrTabBuf = StrTab.SectionData;
  OutputBuffer StrTabOut(StrTabBuf, is64Bit, isLittleEndian);

  // Set the zero'th symbol to a null byte, as required.
  StrTabOut.outbyte(0);
  SymbolTable[0].NameIdx = 0;
  unsigned Index = 1;
  for (unsigned i = 1, e = SymbolTable.size(); i != e; ++i) {
    // Use the name mangler to uniquify the LLVM symbol.
    std::string Name = Mang->getValueName(SymbolTable[i].GV);

    if (Name.empty()) {
      SymbolTable[i].NameIdx = 0;
    } else {
      SymbolTable[i].NameIdx = Index;

      // Add the name to the output buffer, including the null terminator.
      StrTabBuf.insert(StrTabBuf.end(), Name.begin(), Name.end());

      // Add a null terminator.
      StrTabBuf.push_back(0);

      // Keep track of the number of bytes emitted to this section.
      Index += Name.size()+1;
    }
  }
  assert(Index == StrTabBuf.size());
  StrTab.Size = Index;

  // Now that we have emitted the string table and know the offset into the
  // string table of each symbol, emit the symbol table itself.
  ELFSection &SymTab = getSection(".symtab", ELFSection::SHT_SYMTAB, 0);
  SymTab.Align = is64Bit ? 8 : 4;
  SymTab.Link = SymTab.SectionIdx;     // Section Index of .strtab.
  SymTab.Info = FirstNonLocalSymbol;   // First non-STB_LOCAL symbol.
  SymTab.EntSize = 16; // Size of each symtab entry. FIXME: wrong for ELF64
  DataBuffer &SymTabBuf = SymTab.SectionData;
  OutputBuffer SymTabOut(SymTabBuf, is64Bit, isLittleEndian);

  if (!is64Bit) {   // 32-bit and 64-bit formats are shuffled a bit.
    for (unsigned i = 0, e = SymbolTable.size(); i != e; ++i) {
      ELFSym &Sym = SymbolTable[i];
      SymTabOut.outword(Sym.NameIdx);
      SymTabOut.outaddr32(Sym.Value);
      SymTabOut.outword(Sym.Size);
      SymTabOut.outbyte(Sym.Info);
      SymTabOut.outbyte(Sym.Other);
      SymTabOut.outhalf(Sym.SectionIdx);
    }
  } else {
    for (unsigned i = 0, e = SymbolTable.size(); i != e; ++i) {
      ELFSym &Sym = SymbolTable[i];
      SymTabOut.outword(Sym.NameIdx);
      SymTabOut.outbyte(Sym.Info);
      SymTabOut.outbyte(Sym.Other);
      SymTabOut.outhalf(Sym.SectionIdx);
      SymTabOut.outaddr64(Sym.Value);
      SymTabOut.outxword(Sym.Size);
    }
  }

  SymTab.Size = SymTabBuf.size();
}

/// EmitSectionTableStringTable - This method adds and emits a section for the
/// ELF Section Table string table: the string table that holds all of the
/// section names.
void ELFWriter::EmitSectionTableStringTable() {
  // First step: add the section for the string table to the list of sections:
  ELFSection &SHStrTab = getSection(".shstrtab", ELFSection::SHT_STRTAB, 0);

  // Now that we know which section number is the .shstrtab section, update the
  // e_shstrndx entry in the ELF header.
  OutputBuffer FHOut(FileHeader, is64Bit, isLittleEndian);
  FHOut.fixhalf(SHStrTab.SectionIdx, ELFHdr_e_shstrndx_Offset);

  // Set the NameIdx of each section in the string table and emit the bytes for
  // the string table.
  unsigned Index = 0;
  DataBuffer &Buf = SHStrTab.SectionData;

  for (std::list<ELFSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    // Set the index into the table.  Note if we have lots of entries with
    // common suffixes, we could memoize them here if we cared.
    I->NameIdx = Index;

    // Add the name to the output buffer, including the null terminator.
    Buf.insert(Buf.end(), I->Name.begin(), I->Name.end());

    // Add a null terminator.
    Buf.push_back(0);

    // Keep track of the number of bytes emitted to this section.
    Index += I->Name.size()+1;
  }

  // Set the size of .shstrtab now that we know what it is.
  assert(Index == Buf.size());
  SHStrTab.Size = Index;
}

/// OutputSectionsAndSectionTable - Now that we have constructed the file header
/// and all of the sections, emit these to the ostream destination and emit the
/// SectionTable.
void ELFWriter::OutputSectionsAndSectionTable() {
  // Pass #1: Compute the file offset for each section.
  size_t FileOff = FileHeader.size();   // File header first.

  // Emit all of the section data in order.
  for (std::list<ELFSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    // Align FileOff to whatever the alignment restrictions of the section are.
    if (I->Align)
      FileOff = (FileOff+I->Align-1) & ~(I->Align-1);
    I->Offset = FileOff;
    FileOff += I->SectionData.size();
  }

  // Align Section Header.
  unsigned TableAlign = is64Bit ? 8 : 4;
  FileOff = (FileOff+TableAlign-1) & ~(TableAlign-1);

  // Now that we know where all of the sections will be emitted, set the e_shnum
  // entry in the ELF header.
  OutputBuffer FHOut(FileHeader, is64Bit, isLittleEndian);
  FHOut.fixhalf(NumSections, ELFHdr_e_shnum_Offset);

  // Now that we know the offset in the file of the section table, update the
  // e_shoff address in the ELF header.
  FHOut.fixaddr(FileOff, ELFHdr_e_shoff_Offset);

  // Now that we know all of the data in the file header, emit it and all of the
  // sections!
  O.write((char*)&FileHeader[0], FileHeader.size());
  FileOff = FileHeader.size();
  DataBuffer().swap(FileHeader);

  DataBuffer Table;
  OutputBuffer TableOut(Table, is64Bit, isLittleEndian);

  // Emit all of the section data and build the section table itself.
  while (!SectionList.empty()) {
    const ELFSection &S = *SectionList.begin();

    // Align FileOff to whatever the alignment restrictions of the section are.
    if (S.Align)
      for (size_t NewFileOff = (FileOff+S.Align-1) & ~(S.Align-1);
           FileOff != NewFileOff; ++FileOff)
        O << (char)0xAB;
    O.write((char*)&S.SectionData[0], S.SectionData.size());
    FileOff += S.SectionData.size();

    TableOut.outword(S.NameIdx);  // sh_name - Symbol table name idx
    TableOut.outword(S.Type);     // sh_type - Section contents & semantics
    TableOut.outword(S.Flags);    // sh_flags - Section flags.
    TableOut.outaddr(S.Addr);     // sh_addr - The mem addr this section is in.
    TableOut.outaddr(S.Offset);   // sh_offset - Offset from the file start.
    TableOut.outword(S.Size);     // sh_size - The section size.
    TableOut.outword(S.Link);     // sh_link - Section header table index link.
    TableOut.outword(S.Info);     // sh_info - Auxillary information.
    TableOut.outword(S.Align);    // sh_addralign - Alignment of section.
    TableOut.outword(S.EntSize);  // sh_entsize - Size of entries in the section

    SectionList.pop_front();
  }

  // Align output for the section table.
  for (size_t NewFileOff = (FileOff+TableAlign-1) & ~(TableAlign-1);
       FileOff != NewFileOff; ++FileOff)
    O << (char)0xAB;

  // Emit the section table itself.
  O.write((char*)&Table[0], Table.size());
}
