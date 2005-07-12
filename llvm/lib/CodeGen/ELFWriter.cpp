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
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                       ELFCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {
  /// ELFCodeEmitter - This class is used by the ELFWriter to emit the code for
  /// functions to the ELF file.
  class ELFCodeEmitter : public MachineCodeEmitter {
    ELFWriter &EW;
    std::vector<unsigned char> &OutputBuffer;
    size_t FnStart;
  public:
    ELFCodeEmitter(ELFWriter &ew) : EW(ew), OutputBuffer(EW.OutputBuffer) {}

    void startFunction(MachineFunction &F);
    void finishFunction(MachineFunction &F);

    void emitConstantPool(MachineConstantPool *MCP) {
      if (MCP->isEmpty()) return;
      assert(0 && "unimp");
    }
    virtual void emitByte(unsigned char B) {
      OutputBuffer.push_back(B);
    }
    virtual void emitWordAt(unsigned W, unsigned *Ptr) {
      assert(0 && "ni");
    }
    virtual void emitWord(unsigned W) {
      assert(0 && "ni");
    }
    virtual uint64_t getCurrentPCValue() {
      return OutputBuffer.size();
    }
    virtual uint64_t getCurrentPCOffset() {
      return OutputBuffer.size()-FnStart;
    }
    void addRelocation(const MachineRelocation &MR) {
      assert(0 && "relo not handled yet!");
    }
    virtual uint64_t getConstantPoolEntryAddress(unsigned Index) {
      assert(0 && "CP not implementated yet!");
      return 0;
    }

    /// JIT SPECIFIC FUNCTIONS - DO NOT IMPLEMENT THESE HERE!
    void startFunctionStub(unsigned StubSize) {
      assert(0 && "JIT specific function called!");
      abort();
    }
    void *finishFunctionStub(const Function *F) {
      assert(0 && "JIT specific function called!");
      abort();
      return 0;
    }
  };
}

/// startFunction - This callback is invoked when a new machine function is
/// about to be emitted.
void ELFCodeEmitter::startFunction(MachineFunction &F) {
  // Align the output buffer to the appropriate alignment.
  unsigned Align = 16;   // FIXME: GENERICIZE!!
  ELFWriter::ELFSection &TextSection = EW.SectionList.back();
  
  // Upgrade the section alignment if required.
  if (TextSection.Align < Align) TextSection.Align = Align;
  
  // Add padding zeros to the end of the buffer to make sure that the
  // function will start on the correct byte alignment within the section.
  size_t SectionOff = OutputBuffer.size()-TextSection.Offset;
  if (SectionOff & (Align-1)) {
    // Add padding to get alignment to the correct place.
    size_t Pad = Align-(SectionOff & (Align-1));
    OutputBuffer.resize(OutputBuffer.size()+Pad);
  }
  
  FnStart = OutputBuffer.size();
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.
void ELFCodeEmitter::finishFunction(MachineFunction &F) {
  // We now know the size of the function, add a symbol to represent it.
  ELFWriter::ELFSym FnSym(F.getFunction());
  
  // Figure out the binding (linkage) of the symbol.
  switch (F.getFunction()->getLinkage()) {
  default:
    // appending linkage is illegal for functions.
    assert(0 && "Unknown linkage type!");
  case GlobalValue::ExternalLinkage:
    FnSym.SetBind(ELFWriter::ELFSym::STB_GLOBAL);
    break;
  case GlobalValue::LinkOnceLinkage:
  case GlobalValue::WeakLinkage:
    FnSym.SetBind(ELFWriter::ELFSym::STB_WEAK);
    break;
  case GlobalValue::InternalLinkage:
    FnSym.SetBind(ELFWriter::ELFSym::STB_LOCAL);
    break;
  }
  
  FnSym.SetType(ELFWriter::ELFSym::STT_FUNC);
  FnSym.SectionIdx = EW.SectionList.size()-1;  // .text section.
  // Value = Offset from start of .text
  FnSym.Value = FnStart - EW.SectionList.back().Offset;
  FnSym.Size = OutputBuffer.size()-FnStart;
  
  // Finally, add it to the symtab.
  EW.SymbolTable.push_back(FnSym);
}

//===----------------------------------------------------------------------===//
//                          ELFWriter Implementation
//===----------------------------------------------------------------------===//

ELFWriter::ELFWriter(std::ostream &o, TargetMachine &tm) : O(o), TM(tm) {
  e_machine = 0;  // e_machine defaults to 'No Machine'
  e_flags = 0;    // e_flags defaults to 0, no flags.

  is64Bit = TM.getTargetData().getPointerSizeInBits() == 64;  
  isLittleEndian = TM.getTargetData().isLittleEndian();

  // Create the machine code emitter object for this target.
  MCE = new ELFCodeEmitter(*this);
}

ELFWriter::~ELFWriter() {
  delete MCE;
}

// doInitialization - Emit the file header and all of the global variables for
// the module to the ELF file.
bool ELFWriter::doInitialization(Module &M) {
  Mang = new Mangler(M);

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

  SectionList.push_back(ELFSection(".text", OutputBuffer.size()));

  return false;
}

void ELFWriter::EmitGlobal(GlobalVariable *GV, ELFSection &DataSection,
                           ELFSection &BSSSection) {
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
  
  const Type *GVType = (const Type*)GV->getType();
  unsigned Align = TM.getTargetData().getTypeAlignment(GVType);
  unsigned Size  = TM.getTargetData().getTypeSize(GVType);

  // If this global has a zero initializer, it is part of the .bss or common
  // section.
  if (GV->getInitializer()->isNullValue()) {
    // If this global is part of the common block, add it now.  Variables are
    // part of the common block if they are zero initialized and allowed to be
    // merged with other symbols.
    if (GV->hasLinkOnceLinkage() || GV->hasWeakLinkage()) {
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
    default:  // weak/linkonce handled above
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
    BSSSym.SectionIdx = &BSSSection-&SectionList[0];
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
  // Okay, the .text section has now been finalized.  If it contains nothing, do
  // not emit it.
  uint64_t TextSize = OutputBuffer.size() - SectionList.back().Offset;
  if (TextSize == 0) {
    SectionList.pop_back();
  } else {
    ELFSection &Text = SectionList.back();
    Text.Size = TextSize;
    Text.Type = ELFSection::SHT_PROGBITS;
    Text.Flags = ELFSection::SHF_EXECINSTR | ELFSection::SHF_ALLOC;
  }

  // Okay, the ELF header and .text sections have been completed, build the
  // .data, .bss, and "common" sections next.
  SectionList.push_back(ELFSection(".data", OutputBuffer.size()));
  SectionList.push_back(ELFSection(".bss"));
  ELFSection &DataSection = *(SectionList.end()-2);
  ELFSection &BSSSection = SectionList.back();
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I, DataSection, BSSSection);

  // Finish up the data section.
  DataSection.Type  = ELFSection::SHT_PROGBITS;
  DataSection.Flags = ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC;

  // The BSS Section logically starts at the end of the Data Section (adjusted
  // to the required alignment of the BSSSection).
  BSSSection.Offset = DataSection.Offset+DataSection.Size;
  BSSSection.Type   = ELFSection::SHT_NOBITS; 
  BSSSection.Flags  = ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC;
  if (BSSSection.Align)
    BSSSection.Offset = (BSSSection.Offset+BSSSection.Align-1) &
                        ~(BSSSection.Align-1);

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

  SectionList.push_back(ELFSection(".strtab", OutputBuffer.size()));
  ELFSection &StrTab = SectionList.back();
  StrTab.Type = ELFSection::SHT_STRTAB;
  StrTab.Align = 1;

  // Set the zero'th symbol to a null byte, as required.
  outbyte(0);
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
