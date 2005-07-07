//===-- ELFWriter.cpp - Target-independent ELF Writer code ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

#include "llvm/CodeGen/ELFWriter.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

ELFWriter::ELFWriter(std::ostream &o, TargetMachine &tm) : O(o), TM(tm) {
  e_machine = 0;  // e_machine defaults to 'No Machine'
  e_flags = 0;    // e_flags defaults to 0, no flags.

  is64Bit = TM.getTargetData().getPointerSizeInBits() == 64;  
  isLittleEndian = TM.getTargetData().isLittleEndian();
}

// doInitialization - Emit the file header and all of the global variables for
// the module to the ELF file.
bool ELFWriter::doInitialization(Module &M) {
  outbyte(0x7F);                     // EI_MAG0
  outbyte('E');                      // EI_MAG1
  outbyte('L');                      // EI_MAG2
  outbyte('F');                      // EI_MAG3
  outbyte(is64Bit ? 2 : 1);          // EI_CLASS
  outbyte(isLittleEndian ? 1 : 2);   // EI_DATA
  outbyte(1);                        // EI_VERSION
  for (unsigned i = OutputBuffer.size(); i != 16; ++i)
    outbyte(0);                      // EI_PAD up to 16 bytes.
  
  // This should change for shared objects.
  outhalf(1);                        // e_type = ET_REL
  outhalf(e_machine);                // e_machine = whatever the target wants
  outword(1);                        // e_version = 1
  outaddr(0);                        // e_entry = 0 -> no entry point in .o file
  outaddr(0);                        // e_phoff = 0 -> no program header for .o

  ELFHeader_e_shoff_Offset = OutputBuffer.size();
  outaddr(0);                        // e_shoff
  outword(e_flags);                  // e_flags = whatever the target wants

  assert(!is64Bit && "These sizes need to be adjusted for 64-bit!");
  outhalf(52);                       // e_ehsize = ELF header size
  outhalf(0);                        // e_phentsize = prog header entry size
  outhalf(0);                        // e_phnum     = # prog header entries = 0
  outhalf(40);                       // e_shentsize = sect header entry size

  
  ELFHeader_e_shnum_Offset = OutputBuffer.size();
  outhalf(0);                        // e_shnum     = # of section header ents
  ELFHeader_e_shstrndx_Offset = OutputBuffer.size();
  outhalf(0);                        // e_shstrndx  = Section # of '.shstrtab'

  // Add the null section.
  SectionList.push_back(ELFSection());

  // Start up the symbol table.  The first entry in the symtab is the null
  // entry.
  SymbolTable.push_back(ELFSym(0));



  // FIXME: Should start the .text section.
  return false;
}

void ELFWriter::EmitGlobal(GlobalVariable *GV, ELFSection &DataSection,
                           ELFSection &BSSSection) {
  // If this is an external global, emit it...
  assert(GV->hasInitializer() && "FIXME: unimp");
  
  // If this global has a zero initializer, it is part of the .bss or common
  // section.
  if (GV->getInitializer()->isNullValue()) {
    // If this global is part of the common block, add it now.  Variables are
    // part of the common block if they are zero initialized and allowed to be
    // merged with other symbols.
    if (GV->hasLinkOnceLinkage() || GV->hasWeakLinkage()) {
      ELFSym CommonSym(GV);
      // Value for common symbols is the alignment required.
      const Type *GVType = (const Type*)GV->getType();
      CommonSym.Value = TM.getTargetData().getTypeAlignment(GVType);
      CommonSym.Size  = TM.getTargetData().getTypeSize(GVType);
      CommonSym.SetBind(ELFSym::STB_GLOBAL);
      CommonSym.SetType(ELFSym::STT_OBJECT);
      // TODO SOMEDAY: add ELF visibility.
      CommonSym.SectionIdx = ELFSection::SHN_COMMON;
      SymbolTable.push_back(CommonSym);
      return;
    }

    // FIXME: Implement the .bss section.
    return;
  }

  // FIXME: handle .rodata
  //assert(!GV->isConstant() && "unimp");

  // FIXME: handle .data
  //assert(0 && "unimp");
}


bool ELFWriter::runOnMachineFunction(MachineFunction &MF) {
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the ELF file to 'O'.
bool ELFWriter::doFinalization(Module &M) {
  // Okay, the .text section has now been finalized.
  // FIXME: finalize the .text section.

  // Okay, the ELF header and .text sections have been completed, build the
  // .data, .bss, and "common" sections next.
  ELFSection DataSection(".data", OutputBuffer.size());
  ELFSection BSSSection (".bss");
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I, DataSection, BSSSection);

  // If the .data section is nonempty, add it to our list.
  if (DataSection.Size) {
    DataSection.Align = 4;   // FIXME: Compute!
    // FIXME: Set the right flags and stuff.
    SectionList.push_back(DataSection);
  }

  // If the .bss section is nonempty, add it to our list.
  if (BSSSection.Size) {
    BSSSection.Offset = OutputBuffer.size();
    BSSSection.Align = 4;  // FIXME: Compute!
    // FIXME: Set the right flags and stuff.
    SectionList.push_back(BSSSection);
  }

  // Emit the symbol table now, if non-empty.
  EmitSymbolTable();

  // FIXME: Emit the relocations now.

  // Emit the string table for the sections in the ELF file we have.
  EmitSectionTableStringTable();

  // Emit the .o file section table.
  EmitSectionTable();

  // Emit the .o file to the specified stream.
  O.write((char*)&OutputBuffer[0], OutputBuffer.size());

  // Free the output buffer.
  std::vector<unsigned char>().swap(OutputBuffer);
  return false;
}

/// EmitSymbolTable - If the current symbol table is non-empty, emit the string
/// table for it and then the symbol table itself.
void ELFWriter::EmitSymbolTable() {
  if (SymbolTable.size() == 1) return;  // Only the null entry.

  // FIXME: compact all local symbols to the start of the symtab.
  unsigned FirstNonLocalSymbol = 1;

  SectionList.push_back(ELFSection(".strtab", OutputBuffer.size()));
  ELFSection &StrTab = SectionList.back();
  StrTab.Type = ELFSection::SHT_STRTAB;
  StrTab.Align = 1;

  // Set the zero'th symbol to a null byte, as required.
  outbyte(0);
  SymbolTable[0].NameIdx = 0;
  unsigned Index = 1;
  for (unsigned i = 1, e = SymbolTable.size(); i != e; ++i) {
    // FIXME: USE A MANGLER!!
    const std::string &Name = SymbolTable[i].GV->getName();

    if (Name.empty()) {
      SymbolTable[i].NameIdx = 0;
    } else {
      SymbolTable[i].NameIdx = Index;

      // Add the name to the output buffer, including the null terminator.
      OutputBuffer.insert(OutputBuffer.end(), Name.begin(), Name.end());

      // Add a null terminator.
      OutputBuffer.push_back(0);

      // Keep track of the number of bytes emitted to this section.
      Index += Name.size()+1;
    }
  }

  StrTab.Size = OutputBuffer.size()-StrTab.Offset;

  // Now that we have emitted the string table and know the offset into the
  // string table of each symbol, emit the symbol table itself.
  assert(!is64Bit && "Should this be 8 byte aligned for 64-bit?"
         " (check .Align below also)");
  align(4);

  SectionList.push_back(ELFSection(".symtab", OutputBuffer.size()));
  ELFSection &SymTab = SectionList.back();
  SymTab.Type = ELFSection::SHT_SYMTAB;
  SymTab.Align = 4;   // FIXME: check for ELF64
  SymTab.Link = SectionList.size()-2;  // Section Index of .strtab.
  SymTab.Info = FirstNonLocalSymbol;   // First non-STB_LOCAL symbol.
  SymTab.EntSize = 16; // Size of each symtab entry. FIXME: wrong for ELF64

  assert(!is64Bit && "check this!");
  for (unsigned i = 0, e = SymbolTable.size(); i != e; ++i) {
    ELFSym &Sym = SymbolTable[i];
    outword(Sym.NameIdx);
    outaddr(Sym.Value);
    outword(Sym.Size);
    outbyte(Sym.Info);
    outbyte(Sym.Other);
    outhalf(Sym.SectionIdx);
  }

  SymTab.Size = OutputBuffer.size()-SymTab.Offset;
}

/// EmitSectionTableStringTable - This method adds and emits a section for the
/// ELF Section Table string table: the string table that holds all of the
/// section names.
void ELFWriter::EmitSectionTableStringTable() {
  // First step: add the section for the string table to the list of sections:
  SectionList.push_back(ELFSection(".shstrtab", OutputBuffer.size()));
  SectionList.back().Type = ELFSection::SHT_STRTAB;

  // Now that we know which section number is the .shstrtab section, update the
  // e_shstrndx entry in the ELF header.
  fixhalf(SectionList.size()-1, ELFHeader_e_shstrndx_Offset);

  // Set the NameIdx of each section in the string table and emit the bytes for
  // the string table.
  unsigned Index = 0;

  for (unsigned i = 0, e = SectionList.size(); i != e; ++i) {
    // Set the index into the table.  Note if we have lots of entries with
    // common suffixes, we could memoize them here if we cared.
    SectionList[i].NameIdx = Index;

    // Add the name to the output buffer, including the null terminator.
    OutputBuffer.insert(OutputBuffer.end(), SectionList[i].Name.begin(),
                        SectionList[i].Name.end());
    // Add a null terminator.
    OutputBuffer.push_back(0);

    // Keep track of the number of bytes emitted to this section.
    Index += SectionList[i].Name.size()+1;
  }

  // Set the size of .shstrtab now that we know what it is.
  SectionList.back().Size = Index;
}

/// EmitSectionTable - Now that we have emitted the entire contents of the file
/// (all of the sections), emit the section table which informs the reader where
/// the boundaries are.
void ELFWriter::EmitSectionTable() {
  // Now that all of the sections have been emitted, set the e_shnum entry in
  // the ELF header.
  fixhalf(SectionList.size(), ELFHeader_e_shnum_Offset);
  
  // Now that we know the offset in the file of the section table (which we emit
  // next), update the e_shoff address in the ELF header.
  fixaddr(OutputBuffer.size(), ELFHeader_e_shoff_Offset);
  
  // Emit all of the section table entries.
  for (unsigned i = 0, e = SectionList.size(); i != e; ++i) {
    const ELFSection &S = SectionList[i];
    outword(S.NameIdx);  // sh_name - Symbol table name idx
    outword(S.Type);     // sh_type - Section contents & semantics
    outword(S.Flags);    // sh_flags - Section flags.
    outaddr(S.Addr);     // sh_addr - The mem address this section appears in.
    outaddr(S.Offset);   // sh_offset - The offset from the start of the file.
    outword(S.Size);     // sh_size - The section size.
    outword(S.Link);     // sh_link - Section header table index link.
    outword(S.Info);     // sh_info - Auxillary information.
    outword(S.Align);    // sh_addralign - Alignment of section.
    outword(S.EntSize);  // sh_entsize - Size of each entry in the section.
  }

  // Release the memory allocated for the section list.
  std::vector<ELFSection>().swap(SectionList);
}
