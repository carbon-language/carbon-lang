//===-- MachOWriter.cpp - Target-independent Mach-O Writer code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent Mach-O writer.  This file writes
// out the Mach-O file in the following order:
//
//  #1 FatHeader (universal-only)
//  #2 FatArch (universal-only, 1 per universal arch)
//  Per arch:
//    #3 Header
//    #4 Load Commands
//    #5 Sections
//    #6 Relocations
//    #7 Symbols
//    #8 Strings
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/CodeGen/MachOWriter.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetJITInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                       MachOCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {
  /// MachOCodeEmitter - This class is used by the MachOWriter to emit the code 
  /// for functions to the Mach-O file.
  class MachOCodeEmitter : public MachineCodeEmitter {
    MachOWriter &MOW;
    
    /// MOS - The current section we're writing to
    MachOWriter::MachOSection *MOS;

    /// Relocations - These are the relocations that the function needs, as
    /// emitted.
    std::vector<MachineRelocation> Relocations;

    /// MBBLocations - This vector is a mapping from MBB ID's to their address.
    /// It is filled in by the StartMachineBasicBlock callback and queried by
    /// the getMachineBasicBlockAddress callback.
    std::vector<intptr_t> MBBLocations;
    
  public:
    MachOCodeEmitter(MachOWriter &mow) : MOW(mow) {}

    void startFunction(MachineFunction &F);
    bool finishFunction(MachineFunction &F);

    void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }
    
    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
      if (MBBLocations.size() <= (unsigned)MBB->getNumber())
        MBBLocations.resize((MBB->getNumber()+1)*2);
      MBBLocations[MBB->getNumber()] = getCurrentPCValue();
    }

    virtual intptr_t getConstantPoolEntryAddress(unsigned Index) const {
      assert(0 && "CP not implementated yet!");
      return 0;
    }
    virtual intptr_t getJumpTableEntryAddress(unsigned Index) const {
      assert(0 && "JT not implementated yet!");
      return 0;
    }

    virtual intptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(MBBLocations.size() > (unsigned)MBB->getNumber() && 
             MBBLocations[MBB->getNumber()] && "MBB not emitted!");
      return MBBLocations[MBB->getNumber()];
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
void MachOCodeEmitter::startFunction(MachineFunction &F) {
  // Align the output buffer to the appropriate alignment, power of 2.
  // FIXME: GENERICIZE!!
  unsigned Align = 4;

  // Get the Mach-O Section that this function belongs in.
  MOS = &MOW.getTextSection();
  
   // FIXME: better memory management
  MOS->SectionData.reserve(4096);
  BufferBegin = &(MOS->SectionData[0]);
  BufferEnd = BufferBegin + MOS->SectionData.capacity();
  CurBufferPtr = BufferBegin + MOS->size;

  // Upgrade the section alignment if required.
  if (MOS->align < Align) MOS->align = Align;

  // Make sure we only relocate to this function's MBBs.
  MBBLocations.clear();
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.
bool MachOCodeEmitter::finishFunction(MachineFunction &F) {
  MOS->size += CurBufferPtr - BufferBegin;
  
  // Get a symbol for the function to add to the symbol table
  MachOWriter::MachOSym FnSym(F.getFunction(), MOS->Index);
  
  // Figure out the binding (linkage) of the symbol.
  switch (F.getFunction()->getLinkage()) {
  default:
    // appending linkage is illegal for functions.
    assert(0 && "Unknown linkage type!");
  case GlobalValue::ExternalLinkage:
    FnSym.n_type = MachOWriter::MachOSym::N_SECT | MachOWriter::MachOSym::N_EXT;
    break;
  case GlobalValue::InternalLinkage:
    FnSym.n_type = MachOWriter::MachOSym::N_SECT;
    break;
  }
  
  // Resolve the function's relocations either to concrete pointers in the case
  // of branches from one block to another, or to target relocation entries.
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    if (MR.isBasicBlock()) {
      void *MBBAddr = (void *)getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setResultPointer(MBBAddr);
      MOW.TM.getJITInfo()->relocate(BufferBegin, &MR, 1, 0);
      // FIXME: we basically want the JITInfo relocate() function to rewrite
      //        this guy right now, so we just write the correct displacement
      //        to the file.
    } else {
      // isString | isGV | isCPI | isJTI
      // FIXME: do something smart here.  We won't be able to relocate these
      //        until the sections are all layed out, but we still need to
      //        record them.  Maybe emit TargetRelocations and then resolve
      //        those at file writing time?
      std::cerr << "whee!\n";
    }
  }
  Relocations.clear();
  
  // Finally, add it to the symtab.
  MOW.SymbolTable.push_back(FnSym);
  return false;
}

//===----------------------------------------------------------------------===//
//                          MachOWriter Implementation
//===----------------------------------------------------------------------===//

MachOWriter::MachOWriter(std::ostream &o, TargetMachine &tm) : O(o), TM(tm) {
  // FIXME: set cpu type and cpu subtype somehow from TM
  is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
  isLittleEndian = TM.getTargetData()->isLittleEndian();

  // Create the machine code emitter object for this target.
  MCE = new MachOCodeEmitter(*this);
}

MachOWriter::~MachOWriter() {
  delete MCE;
}

void MachOWriter::EmitGlobal(GlobalVariable *GV) {
  // FIXME: do something smart here.
}


bool MachOWriter::runOnMachineFunction(MachineFunction &MF) {
  // Nothing to do here, this is all done through the MCE object.
  return false;
}

bool MachOWriter::doInitialization(Module &M) {
  // Set the magic value, now that we know the pointer size and endianness
  Header.setMagic(isLittleEndian, is64Bit);

  // Set the file type
  // FIXME: this only works for object files, we do not support the creation
  //        of dynamic libraries or executables at this time.
  Header.filetype = MachOHeader::MH_OBJECT;

  Mang = new Mangler(M);
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the Mach-O file to 'O'.
bool MachOWriter::doFinalization(Module &M) {
  // Okay, the.text section has been completed, build the .data, .bss, and 
  // "common" sections next.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I);
  
  // Emit the header and load commands.
  EmitHeaderAndLoadCommands();

  // Emit the text and data sections.
  EmitSections();

  // Emit the relocation entry data for each section.
  // FIXME: presumably this should be a virtual method, since different targets
  //        have different relocation types.
  EmitRelocations();

  // Emit the symbol table.
  // FIXME: we don't handle debug info yet, we should probably do that.
  EmitSymbolTable();

  // Emit the string table for the sections we have.
  EmitStringTable();

  // We are done with the abstract symbols.
  SectionList.clear();
  SymbolTable.clear();
  DynamicSymbolTable.clear();

  // Release the name mangler object.
  delete Mang; Mang = 0;
  return false;
}

void MachOWriter::EmitHeaderAndLoadCommands() {
  // Step #0: Fill in the segment load command size, since we need it to figure
  //          out the rest of the header fields
  MachOSegment SEG("", is64Bit);
  SEG.nsects  = SectionList.size();
  SEG.cmdsize = SEG.cmdSize(is64Bit) + 
                SEG.nsects * SectionList.begin()->cmdSize(is64Bit);
  
  // Step #1: calculate the number of load commands.  We always have at least
  //          one, for the LC_SEGMENT load command, plus two for the normal
  //          and dynamic symbol tables, if there are any symbols.
  Header.ncmds = SymbolTable.empty() ? 1 : 3;
  
  // Step #2: calculate the size of the load commands
  Header.sizeofcmds = SEG.cmdsize;
  if (!SymbolTable.empty())
    Header.sizeofcmds += SymTab.cmdsize + DySymTab.cmdsize;
    
  // Step #3: write the header to the file
  // Local alias to shortenify coming code.
  DataBuffer &FH = Header.HeaderData;
  outword(FH, Header.magic);
  outword(FH, Header.cputype);
  outword(FH, Header.cpusubtype);
  outword(FH, Header.filetype);
  outword(FH, Header.ncmds);
  outword(FH, Header.sizeofcmds);
  outword(FH, Header.flags);
  if (is64Bit)
    outword(FH, Header.reserved);
  
  // Step #4: Finish filling in the segment load command and write it out
  for (std::list<MachOSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    SEG.filesize += I->size;
  SEG.vmsize = SEG.filesize;
  SEG.fileoff = Header.cmdSize(is64Bit) + Header.sizeofcmds;
  
  outword(FH, SEG.cmd);
  outword(FH, SEG.cmdsize);
  outstring(FH, SEG.segname, 16);
  outaddr(FH, SEG.vmaddr);
  outaddr(FH, SEG.vmsize);
  outaddr(FH, SEG.fileoff);
  outaddr(FH, SEG.filesize);
  outword(FH, SEG.maxprot);
  outword(FH, SEG.initprot);
  outword(FH, SEG.nsects);
  outword(FH, SEG.flags);
  
  // Step #5: Write out the section commands for each section
  for (std::list<MachOSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    I->offset = SEG.fileoff;  // FIXME: separate offset
    outstring(FH, I->sectname, 16);
    outstring(FH, I->segname, 16);
    outaddr(FH, I->addr);
    outaddr(FH, I->size);
    outword(FH, I->offset);
    outword(FH, I->align);
    outword(FH, I->reloff);
    outword(FH, I->nreloc);
    outword(FH, I->flags);
    outword(FH, I->reserved1);
    outword(FH, I->reserved2);
    if (is64Bit)
      outword(FH, I->reserved3);
  }
  
  // Step #6: Emit LC_SYMTAB/LC_DYSYMTAB load commands
  // FIXME: We'll need to scan over the symbol table and possibly do the sort
  // here so that we can set the proper indices in the dysymtab load command for
  // the index and number of external symbols defined in this module.
  // FIXME: We'll also need to scan over all the symbols so that we can 
  // calculate the size of the string table.
  // FIXME: add size of relocs
  SymTab.symoff  = SEG.fileoff + SEG.filesize;
  SymTab.nsyms   = SymbolTable.size();
  SymTab.stroff  = SymTab.symoff + SymTab.nsyms * MachOSym::entrySize();
  SymTab.strsize = 10;
  outword(FH, SymTab.cmd);
  outword(FH, SymTab.cmdsize);
  outword(FH, SymTab.symoff);
  outword(FH, SymTab.nsyms);
  outword(FH, SymTab.stroff);
  outword(FH, SymTab.strsize);

  // FIXME: set DySymTab fields appropriately
  outword(FH, DySymTab.cmd);
  outword(FH, DySymTab.cmdsize);
  outword(FH, DySymTab.ilocalsym);
  outword(FH, DySymTab.nlocalsym);
  outword(FH, DySymTab.iextdefsym);
  outword(FH, DySymTab.nextdefsym);
  outword(FH, DySymTab.iundefsym);
  outword(FH, DySymTab.nundefsym);
  outword(FH, DySymTab.tocoff);
  outword(FH, DySymTab.ntoc);
  outword(FH, DySymTab.modtaboff);
  outword(FH, DySymTab.nmodtab);
  outword(FH, DySymTab.extrefsymoff);
  outword(FH, DySymTab.nextrefsyms);
  outword(FH, DySymTab.indirectsymoff);
  outword(FH, DySymTab.nindirectsyms);
  outword(FH, DySymTab.extreloff);
  outword(FH, DySymTab.nextrel);
  outword(FH, DySymTab.locreloff);
  outword(FH, DySymTab.nlocrel);
  
  O.write((char*)&FH[0], FH.size());
}

/// EmitSections - Now that we have constructed the file header and load
/// commands, emit the data for each section to the file.
void MachOWriter::EmitSections() {
  for (std::list<MachOSection>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    O.write((char*)&I->SectionData[0], I->size);
  }
}

void MachOWriter::EmitRelocations() {
  // FIXME: this should probably be a pure virtual function, since the
  // relocation types and layout of the relocations themselves are target
  // specific.
}

/// EmitSymbolTable - Sort the symbols we encountered and assign them each a 
/// string table index so that they appear in the correct order in the output 
/// file.
void MachOWriter::EmitSymbolTable() {
  // The order of the symbol table is:
  // local symbols
  // defined external symbols (sorted by name)
  // undefined external symbols (sorted by name)
  DataBuffer ST;
  
  // FIXME: enforce the above ordering, presumably by sorting by name, 
  // then partitioning twice.
  unsigned stringIndex;
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I) {
    // FIXME: remove when we actually calculate these correctly
    I->n_strx = 1;
    StringTable.push_back(Mang->getValueName(I->GV));
    // Emit nlist to buffer
    outword(ST, I->n_strx);
    outbyte(ST, I->n_type);
    outbyte(ST, I->n_sect);
    outhalf(ST, I->n_desc);
    outaddr(ST, I->n_value);
  }
  
  O.write((char*)&ST[0], ST.size());
}

/// EmitStringTable - This method adds and emits a section for the Mach-O 
/// string table.
void MachOWriter::EmitStringTable() {
  // The order of the string table is:
  // strings for external symbols
  // strings for local symbols
  // This is the symbol table, but backwards.  This allows us to avoid a sorting
  // the symbol table again; all we have to do is use a reverse iterator.
  DataBuffer ST;

  // Write out a leading zero byte when emitting string table, for n_strx == 0
  // which means an empty string.
  outbyte(ST, 0);

  for (std::vector<std::string>::iterator I = StringTable.begin(),
         E = StringTable.end(); I != E; ++I) {
    // FIXME: do not arbitrarily cap symbols to 16 characters
    // FIXME: do something more efficient than outstring
    outstring(ST, *I, 16);
  }
  O.write((char*)&ST[0], ST.size());
}
