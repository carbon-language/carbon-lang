//===-- MachOWriter.cpp - Target-independent Mach-O Writer code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "MachO.h"
#include "MachOWriter.h"
#include "MachOCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachOWriterInfo.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/OutputBuffer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// AddMachOWriter - Concrete function to add the Mach-O writer to the function
/// pass manager.
ObjectCodeEmitter *AddMachOWriter(PassManagerBase &PM,
                                         raw_ostream &O,
                                         TargetMachine &TM) {
  MachOWriter *MOW = new MachOWriter(O, TM);
  PM.add(MOW);
  return MOW->getObjectCodeEmitter();
}

//===----------------------------------------------------------------------===//
//                          MachOWriter Implementation
//===----------------------------------------------------------------------===//

char MachOWriter::ID = 0;

MachOWriter::MachOWriter(raw_ostream &o, TargetMachine &tm)
  : MachineFunctionPass(&ID), O(o), TM(tm) {
  is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
  isLittleEndian = TM.getTargetData()->isLittleEndian();

  MAI = TM.getMCAsmInfo();

  // Create the machine code emitter object for this target.
  MachOCE = new MachOCodeEmitter(*this, *getTextSection(true));
}

MachOWriter::~MachOWriter() {
  delete MachOCE;
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

bool MachOWriter::runOnMachineFunction(MachineFunction &MF) {
  return false;
}

/// doFinalization - Now that the module has been completely processed, emit
/// the Mach-O file to 'O'.
bool MachOWriter::doFinalization(Module &M) {
  // FIXME: we don't handle debug info yet, we should probably do that.
  // Okay, the.text section has been completed, build the .data, .bss, and
  // "common" sections next.

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobal(I);

  // Emit the header and load commands.
  EmitHeaderAndLoadCommands();

  // Emit the various sections and their relocation info.
  EmitSections();
  EmitRelocations();

  // Write the symbol table and the string table to the end of the file.
  O.write((char*)&SymT[0], SymT.size());
  O.write((char*)&StrT[0], StrT.size());

  // We are done with the abstract symbols.
  SectionList.clear();
  SymbolTable.clear();
  DynamicSymbolTable.clear();

  // Release the name mangler object.
  delete Mang; Mang = 0;
  return false;
}

// getConstSection - Get constant section for Constant 'C'
MachOSection *MachOWriter::getConstSection(Constant *C) {
  const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
  if (CVA && CVA->isCString())
    return getSection("__TEXT", "__cstring", 
                      MachOSection::S_CSTRING_LITERALS);

  const Type *Ty = C->getType();
  if (Ty->isPrimitiveType() || Ty->isInteger()) {
    unsigned Size = TM.getTargetData()->getTypeAllocSize(Ty);
    switch(Size) {
    default: break; // Fall through to __TEXT,__const
    case 4:
      return getSection("__TEXT", "__literal4",
                        MachOSection::S_4BYTE_LITERALS);
    case 8:
      return getSection("__TEXT", "__literal8",
                        MachOSection::S_8BYTE_LITERALS);
    case 16:
      return getSection("__TEXT", "__literal16",
                        MachOSection::S_16BYTE_LITERALS);
    }
  }
  return getSection("__TEXT", "__const");
}

// getJumpTableSection - Select the Jump Table section
MachOSection *MachOWriter::getJumpTableSection() {
  if (TM.getRelocationModel() == Reloc::PIC_)
    return getTextSection(false);
  else
    return getSection("__TEXT", "__const");
}

// getSection - Return the section with the specified name, creating a new
// section if one does not already exist.
MachOSection *MachOWriter::getSection(const std::string &seg,
                                      const std::string &sect,
                                      unsigned Flags /* = 0 */ ) {
  MachOSection *MOS = SectionLookup[seg+sect];
  if (MOS) return MOS;

  MOS = new MachOSection(seg, sect);
  SectionList.push_back(MOS);
  MOS->Index = SectionList.size();
  MOS->flags = MachOSection::S_REGULAR | Flags;
  SectionLookup[seg+sect] = MOS;
  return MOS;
}

// getTextSection - Return text section with different flags for code/data
MachOSection *MachOWriter::getTextSection(bool isCode /* = true */ ) {
  if (isCode)
    return getSection("__TEXT", "__text",
                      MachOSection::S_ATTR_PURE_INSTRUCTIONS |
                      MachOSection::S_ATTR_SOME_INSTRUCTIONS);
  else
    return getSection("__TEXT", "__text");
}

MachOSection *MachOWriter::getBSSSection() {
  return getSection("__DATA", "__bss", MachOSection::S_ZEROFILL);
}

// GetJTRelocation - Get a relocation a new BB relocation based
// on target information.
MachineRelocation MachOWriter::GetJTRelocation(unsigned Offset,
                                               MachineBasicBlock *MBB) const {
  return TM.getMachOWriterInfo()->GetJTRelocation(Offset, MBB);
}

// GetTargetRelocation - Returns the number of relocations.
unsigned MachOWriter::GetTargetRelocation(MachineRelocation &MR,
                             unsigned FromIdx, unsigned ToAddr,
                             unsigned ToIndex, OutputBuffer &RelocOut,
                             OutputBuffer &SecOut, bool Scattered,
                             bool Extern) {
  return TM.getMachOWriterInfo()->GetTargetRelocation(MR, FromIdx, ToAddr,
                                                      ToIndex, RelocOut,
                                                      SecOut, Scattered,
                                                      Extern);
}

void MachOWriter::AddSymbolToSection(MachOSection *Sec, GlobalVariable *GV) {
  const Type *Ty = GV->getType()->getElementType();
  unsigned Size = TM.getTargetData()->getTypeAllocSize(Ty);
  unsigned Align = TM.getTargetData()->getPreferredAlignment(GV);

  // Reserve space in the .bss section for this symbol while maintaining the
  // desired section alignment, which must be at least as much as required by
  // this symbol.
  OutputBuffer SecDataOut(Sec->getData(), is64Bit, isLittleEndian);

  if (Align) {
    Align = Log2_32(Align);
    Sec->align = std::max(unsigned(Sec->align), Align);

    Sec->emitAlignment(Sec->align);
  }
  // Globals without external linkage apparently do not go in the symbol table.
  if (!GV->hasLocalLinkage()) {
    MachOSym Sym(GV, Mang->getMangledName(GV), Sec->Index, MAI);
    Sym.n_value = Sec->size();
    SymbolTable.push_back(Sym);
  }

  // Record the offset of the symbol, and then allocate space for it.
  // FIXME: remove when we have unified size + output buffer

  // Now that we know what section the GlovalVariable is going to be emitted
  // into, update our mappings.
  // FIXME: We may also need to update this when outputting non-GlobalVariable
  // GlobalValues such as functions.

  GVSection[GV] = Sec;
  GVOffset[GV] = Sec->size();

  // Allocate space in the section for the global.
  for (unsigned i = 0; i < Size; ++i)
    SecDataOut.outbyte(0);
}

void MachOWriter::EmitGlobal(GlobalVariable *GV) {
  const Type *Ty = GV->getType()->getElementType();
  unsigned Size = TM.getTargetData()->getTypeAllocSize(Ty);
  bool NoInit = !GV->hasInitializer();

  // If this global has a zero initializer, it is part of the .bss or common
  // section.
  if (NoInit || GV->getInitializer()->isNullValue()) {
    // If this global is part of the common block, add it now.  Variables are
    // part of the common block if they are zero initialized and allowed to be
    // merged with other symbols.
    if (NoInit || GV->hasLinkOnceLinkage() || GV->hasWeakLinkage() ||
        GV->hasCommonLinkage()) {
      MachOSym ExtOrCommonSym(GV, Mang->getMangledName(GV),
                              MachOSym::NO_SECT, MAI);
      // For undefined (N_UNDF) external (N_EXT) types, n_value is the size in
      // bytes of the symbol.
      ExtOrCommonSym.n_value = Size;
      SymbolTable.push_back(ExtOrCommonSym);
      // Remember that we've seen this symbol
      GVOffset[GV] = Size;
      return;
    }
    // Otherwise, this symbol is part of the .bss section.
    MachOSection *BSS = getBSSSection();
    AddSymbolToSection(BSS, GV);
    return;
  }

  // Scalar read-only data goes in a literal section if the scalar is 4, 8, or
  // 16 bytes, or a cstring.  Other read only data goes into a regular const
  // section.  Read-write data goes in the data section.
  MachOSection *Sec = GV->isConstant() ? getConstSection(GV->getInitializer()) :
                                         getDataSection();
  AddSymbolToSection(Sec, GV);
  InitMem(GV->getInitializer(), GVOffset[GV], TM.getTargetData(), Sec);
}



void MachOWriter::EmitHeaderAndLoadCommands() {
  // Step #0: Fill in the segment load command size, since we need it to figure
  //          out the rest of the header fields

  MachOSegment SEG("", is64Bit);
  SEG.nsects  = SectionList.size();
  SEG.cmdsize = SEG.cmdSize(is64Bit) +
                SEG.nsects * SectionList[0]->cmdSize(is64Bit);

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
  std::vector<unsigned char> &FH = Header.HeaderData;
  OutputBuffer FHOut(FH, is64Bit, isLittleEndian);

  FHOut.outword(Header.magic);
  FHOut.outword(TM.getMachOWriterInfo()->getCPUType());
  FHOut.outword(TM.getMachOWriterInfo()->getCPUSubType());
  FHOut.outword(Header.filetype);
  FHOut.outword(Header.ncmds);
  FHOut.outword(Header.sizeofcmds);
  FHOut.outword(Header.flags);
  if (is64Bit)
    FHOut.outword(Header.reserved);

  // Step #4: Finish filling in the segment load command and write it out
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    SEG.filesize += (*I)->size();

  SEG.vmsize = SEG.filesize;
  SEG.fileoff = Header.cmdSize(is64Bit) + Header.sizeofcmds;

  FHOut.outword(SEG.cmd);
  FHOut.outword(SEG.cmdsize);
  FHOut.outstring(SEG.segname, 16);
  FHOut.outaddr(SEG.vmaddr);
  FHOut.outaddr(SEG.vmsize);
  FHOut.outaddr(SEG.fileoff);
  FHOut.outaddr(SEG.filesize);
  FHOut.outword(SEG.maxprot);
  FHOut.outword(SEG.initprot);
  FHOut.outword(SEG.nsects);
  FHOut.outword(SEG.flags);

  // Step #5: Finish filling in the fields of the MachOSections
  uint64_t currentAddr = 0;
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    MachOSection *MOS = *I;
    MOS->addr = currentAddr;
    MOS->offset = currentAddr + SEG.fileoff;
    // FIXME: do we need to do something with alignment here?
    currentAddr += MOS->size();
  }

  // Step #6: Emit the symbol table to temporary buffers, so that we know the
  // size of the string table when we write the next load command.  This also
  // sorts and assigns indices to each of the symbols, which is necessary for
  // emitting relocations to externally-defined objects.
  BufferSymbolAndStringTable();

  // Step #7: Calculate the number of relocations for each section and write out
  // the section commands for each section
  currentAddr += SEG.fileoff;
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I) {
    MachOSection *MOS = *I;

    // Convert the relocations to target-specific relocations, and fill in the
    // relocation offset for this section.
    CalculateRelocations(*MOS);
    MOS->reloff = MOS->nreloc ? currentAddr : 0;
    currentAddr += MOS->nreloc * 8;

    // write the finalized section command to the output buffer
    FHOut.outstring(MOS->sectname, 16);
    FHOut.outstring(MOS->segname, 16);
    FHOut.outaddr(MOS->addr);
    FHOut.outaddr(MOS->size());
    FHOut.outword(MOS->offset);
    FHOut.outword(MOS->align);
    FHOut.outword(MOS->reloff);
    FHOut.outword(MOS->nreloc);
    FHOut.outword(MOS->flags);
    FHOut.outword(MOS->reserved1);
    FHOut.outword(MOS->reserved2);
    if (is64Bit)
      FHOut.outword(MOS->reserved3);
  }

  // Step #8: Emit LC_SYMTAB/LC_DYSYMTAB load commands
  SymTab.symoff  = currentAddr;
  SymTab.nsyms   = SymbolTable.size();
  SymTab.stroff  = SymTab.symoff + SymT.size();
  SymTab.strsize = StrT.size();
  FHOut.outword(SymTab.cmd);
  FHOut.outword(SymTab.cmdsize);
  FHOut.outword(SymTab.symoff);
  FHOut.outword(SymTab.nsyms);
  FHOut.outword(SymTab.stroff);
  FHOut.outword(SymTab.strsize);

  // FIXME: set DySymTab fields appropriately
  // We should probably just update these in BufferSymbolAndStringTable since
  // thats where we're partitioning up the different kinds of symbols.
  FHOut.outword(DySymTab.cmd);
  FHOut.outword(DySymTab.cmdsize);
  FHOut.outword(DySymTab.ilocalsym);
  FHOut.outword(DySymTab.nlocalsym);
  FHOut.outword(DySymTab.iextdefsym);
  FHOut.outword(DySymTab.nextdefsym);
  FHOut.outword(DySymTab.iundefsym);
  FHOut.outword(DySymTab.nundefsym);
  FHOut.outword(DySymTab.tocoff);
  FHOut.outword(DySymTab.ntoc);
  FHOut.outword(DySymTab.modtaboff);
  FHOut.outword(DySymTab.nmodtab);
  FHOut.outword(DySymTab.extrefsymoff);
  FHOut.outword(DySymTab.nextrefsyms);
  FHOut.outword(DySymTab.indirectsymoff);
  FHOut.outword(DySymTab.nindirectsyms);
  FHOut.outword(DySymTab.extreloff);
  FHOut.outword(DySymTab.nextrel);
  FHOut.outword(DySymTab.locreloff);
  FHOut.outword(DySymTab.nlocrel);

  O.write((char*)&FH[0], FH.size());
}

/// EmitSections - Now that we have constructed the file header and load
/// commands, emit the data for each section to the file.
void MachOWriter::EmitSections() {
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    // Emit the contents of each section
    if ((*I)->size())
      O.write((char*)&(*I)->getData()[0], (*I)->size());
}

/// EmitRelocations - emit relocation data from buffer.
void MachOWriter::EmitRelocations() {
  for (std::vector<MachOSection*>::iterator I = SectionList.begin(),
         E = SectionList.end(); I != E; ++I)
    // Emit the relocation entry data for each section.
    if ((*I)->RelocBuffer.size())
      O.write((char*)&(*I)->RelocBuffer[0], (*I)->RelocBuffer.size());
}

/// BufferSymbolAndStringTable - Sort the symbols we encountered and assign them
/// each a string table index so that they appear in the correct order in the
/// output file.
void MachOWriter::BufferSymbolAndStringTable() {
  // The order of the symbol table is:
  // 1. local symbols
  // 2. defined external symbols (sorted by name)
  // 3. undefined external symbols (sorted by name)

  // Before sorting the symbols, check the PendingGlobals for any undefined
  // globals that need to be put in the symbol table.
  for (std::vector<GlobalValue*>::iterator I = PendingGlobals.begin(),
         E = PendingGlobals.end(); I != E; ++I) {
    if (GVOffset[*I] == 0 && GVSection[*I] == 0) {
      MachOSym UndfSym(*I, Mang->getMangledName(*I), MachOSym::NO_SECT, MAI);
      SymbolTable.push_back(UndfSym);
      GVOffset[*I] = -1;
    }
  }

  // Sort the symbols by name, so that when we partition the symbols by scope
  // of definition, we won't have to sort by name within each partition.
  std::sort(SymbolTable.begin(), SymbolTable.end(), MachOSym::SymCmp());

  // Parition the symbol table entries so that all local symbols come before
  // all symbols with external linkage. { 1 | 2 3 }
  std::partition(SymbolTable.begin(), SymbolTable.end(),
                 MachOSym::PartitionByLocal);

  // Advance iterator to beginning of external symbols and partition so that
  // all external symbols defined in this module come before all external
  // symbols defined elsewhere. { 1 | 2 | 3 }
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I) {
    if (!MachOSym::PartitionByLocal(*I)) {
      std::partition(I, E, MachOSym::PartitionByDefined);
      break;
    }
  }

  // Calculate the starting index for each of the local, extern defined, and
  // undefined symbols, as well as the number of each to put in the LC_DYSYMTAB
  // load command.
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I) {
    if (MachOSym::PartitionByLocal(*I)) {
      ++DySymTab.nlocalsym;
      ++DySymTab.iextdefsym;
      ++DySymTab.iundefsym;
    } else if (MachOSym::PartitionByDefined(*I)) {
      ++DySymTab.nextdefsym;
      ++DySymTab.iundefsym;
    } else {
      ++DySymTab.nundefsym;
    }
  }

  // Write out a leading zero byte when emitting string table, for n_strx == 0
  // which means an empty string.
  OutputBuffer StrTOut(StrT, is64Bit, isLittleEndian);
  StrTOut.outbyte(0);

  // The order of the string table is:
  // 1. strings for external symbols
  // 2. strings for local symbols
  // Since this is the opposite order from the symbol table, which we have just
  // sorted, we can walk the symbol table backwards to output the string table.
  for (std::vector<MachOSym>::reverse_iterator I = SymbolTable.rbegin(),
        E = SymbolTable.rend(); I != E; ++I) {
    if (I->GVName == "") {
      I->n_strx = 0;
    } else {
      I->n_strx = StrT.size();
      StrTOut.outstring(I->GVName, I->GVName.length()+1);
    }
  }

  OutputBuffer SymTOut(SymT, is64Bit, isLittleEndian);

  unsigned index = 0;
  for (std::vector<MachOSym>::iterator I = SymbolTable.begin(),
         E = SymbolTable.end(); I != E; ++I, ++index) {
    // Add the section base address to the section offset in the n_value field
    // to calculate the full address.
    // FIXME: handle symbols where the n_value field is not the address
    GlobalValue *GV = const_cast<GlobalValue*>(I->GV);
    if (GV && GVSection[GV])
      I->n_value += GVSection[GV]->addr;
    if (GV && (GVOffset[GV] == -1))
      GVOffset[GV] = index;

    // Emit nlist to buffer
    SymTOut.outword(I->n_strx);
    SymTOut.outbyte(I->n_type);
    SymTOut.outbyte(I->n_sect);
    SymTOut.outhalf(I->n_desc);
    SymTOut.outaddr(I->n_value);
  }
}

/// CalculateRelocations - For each MachineRelocation in the current section,
/// calculate the index of the section containing the object to be relocated,
/// and the offset into that section.  From this information, create the
/// appropriate target-specific MachORelocation type and add buffer it to be
/// written out after we are finished writing out sections.
void MachOWriter::CalculateRelocations(MachOSection &MOS) {
  std::vector<MachineRelocation> Relocations =  MOS.getRelocations();
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    unsigned TargetSection = MR.getConstantVal();
    unsigned TargetAddr = 0;
    unsigned TargetIndex = 0;

    // This is a scattered relocation entry if it points to a global value with
    // a non-zero offset.
    bool Scattered = false;
    bool Extern = false;

    // Since we may not have seen the GlobalValue we were interested in yet at
    // the time we emitted the relocation for it, fix it up now so that it
    // points to the offset into the correct section.
    if (MR.isGlobalValue()) {
      GlobalValue *GV = MR.getGlobalValue();
      MachOSection *MOSPtr = GVSection[GV];
      intptr_t Offset = GVOffset[GV];

      // If we have never seen the global before, it must be to a symbol
      // defined in another module (N_UNDF).
      if (!MOSPtr) {
        // FIXME: need to append stub suffix
        Extern = true;
        TargetAddr = 0;
        TargetIndex = GVOffset[GV];
      } else {
        Scattered = TargetSection != 0;
        TargetSection = MOSPtr->Index;
      }
      MR.setResultPointer((void*)Offset);
    }

    // If the symbol is locally defined, pass in the address of the section and
    // the section index to the code which will generate the target relocation.
    if (!Extern) {
        MachOSection &To = *SectionList[TargetSection - 1];
        TargetAddr = To.addr;
        TargetIndex = To.Index;
    }

    OutputBuffer RelocOut(MOS.RelocBuffer, is64Bit, isLittleEndian);
    OutputBuffer SecOut(MOS.getData(), is64Bit, isLittleEndian);

    MOS.nreloc += GetTargetRelocation(MR, MOS.Index, TargetAddr, TargetIndex,
                                      RelocOut, SecOut, Scattered, Extern);
  }
}

// InitMem - Write the value of a Constant to the specified memory location,
// converting it into bytes and relocations.
void MachOWriter::InitMem(const Constant *C, uintptr_t Offset,
                          const TargetData *TD, MachOSection* mos) {
  typedef std::pair<const Constant*, intptr_t> CPair;
  std::vector<CPair> WorkList;
  uint8_t *Addr = &mos->getData()[0];

  WorkList.push_back(CPair(C,(intptr_t)Addr + Offset));

  intptr_t ScatteredOffset = 0;

  while (!WorkList.empty()) {
    const Constant *PC = WorkList.back().first;
    intptr_t PA = WorkList.back().second;
    WorkList.pop_back();

    if (isa<UndefValue>(PC)) {
      continue;
    } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(PC)) {
      unsigned ElementSize =
        TD->getTypeAllocSize(CP->getType()->getElementType());
      for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i)
        WorkList.push_back(CPair(CP->getOperand(i), PA+i*ElementSize));
    } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(PC)) {
      //
      // FIXME: Handle ConstantExpression.  See EE::getConstantValue()
      //
      switch (CE->getOpcode()) {
      case Instruction::GetElementPtr: {
        SmallVector<Value*, 8> Indices(CE->op_begin()+1, CE->op_end());
        ScatteredOffset = TD->getIndexedOffset(CE->getOperand(0)->getType(),
                                               &Indices[0], Indices.size());
        WorkList.push_back(CPair(CE->getOperand(0), PA));
        break;
      }
      case Instruction::Add:
      default:
        errs() << "ConstantExpr not handled as global var init: " << *CE <<"\n";
        llvm_unreachable(0);
      }
    } else if (PC->getType()->isSingleValueType()) {
      unsigned char *ptr = (unsigned char *)PA;
      switch (PC->getType()->getTypeID()) {
      case Type::IntegerTyID: {
        unsigned NumBits = cast<IntegerType>(PC->getType())->getBitWidth();
        uint64_t val = cast<ConstantInt>(PC)->getZExtValue();
        if (NumBits <= 8)
          ptr[0] = val;
        else if (NumBits <= 16) {
          if (TD->isBigEndian())
            val = ByteSwap_16(val);
          ptr[0] = val;
          ptr[1] = val >> 8;
        } else if (NumBits <= 32) {
          if (TD->isBigEndian())
            val = ByteSwap_32(val);
          ptr[0] = val;
          ptr[1] = val >> 8;
          ptr[2] = val >> 16;
          ptr[3] = val >> 24;
        } else if (NumBits <= 64) {
          if (TD->isBigEndian())
            val = ByteSwap_64(val);
          ptr[0] = val;
          ptr[1] = val >> 8;
          ptr[2] = val >> 16;
          ptr[3] = val >> 24;
          ptr[4] = val >> 32;
          ptr[5] = val >> 40;
          ptr[6] = val >> 48;
          ptr[7] = val >> 56;
        } else {
          llvm_unreachable("Not implemented: bit widths > 64");
        }
        break;
      }
      case Type::FloatTyID: {
        uint32_t val = cast<ConstantFP>(PC)->getValueAPF().bitcastToAPInt().
                        getZExtValue();
        if (TD->isBigEndian())
          val = ByteSwap_32(val);
        ptr[0] = val;
        ptr[1] = val >> 8;
        ptr[2] = val >> 16;
        ptr[3] = val >> 24;
        break;
      }
      case Type::DoubleTyID: {
        uint64_t val = cast<ConstantFP>(PC)->getValueAPF().bitcastToAPInt().
                         getZExtValue();
        if (TD->isBigEndian())
          val = ByteSwap_64(val);
        ptr[0] = val;
        ptr[1] = val >> 8;
        ptr[2] = val >> 16;
        ptr[3] = val >> 24;
        ptr[4] = val >> 32;
        ptr[5] = val >> 40;
        ptr[6] = val >> 48;
        ptr[7] = val >> 56;
        break;
      }
      case Type::PointerTyID:
        if (isa<ConstantPointerNull>(PC))
          memset(ptr, 0, TD->getPointerSize());
        else if (const GlobalValue* GV = dyn_cast<GlobalValue>(PC)) {
          // FIXME: what about function stubs?
          mos->addRelocation(MachineRelocation::getGV(PA-(intptr_t)Addr,
                                                 MachineRelocation::VANILLA,
                                                 const_cast<GlobalValue*>(GV),
                                                 ScatteredOffset));
          ScatteredOffset = 0;
        } else
          llvm_unreachable("Unknown constant pointer type!");
        break;
      default:
        std::string msg;
        raw_string_ostream Msg(msg);
        Msg << "ERROR: Constant unimp for type: " << *PC->getType();
        llvm_report_error(Msg.str());
      }
    } else if (isa<ConstantAggregateZero>(PC)) {
      memset((void*)PA, 0, (size_t)TD->getTypeAllocSize(PC->getType()));
    } else if (const ConstantArray *CPA = dyn_cast<ConstantArray>(PC)) {
      unsigned ElementSize =
        TD->getTypeAllocSize(CPA->getType()->getElementType());
      for (unsigned i = 0, e = CPA->getNumOperands(); i != e; ++i)
        WorkList.push_back(CPair(CPA->getOperand(i), PA+i*ElementSize));
    } else if (const ConstantStruct *CPS = dyn_cast<ConstantStruct>(PC)) {
      const StructLayout *SL =
        TD->getStructLayout(cast<StructType>(CPS->getType()));
      for (unsigned i = 0, e = CPS->getNumOperands(); i != e; ++i)
        WorkList.push_back(CPair(CPS->getOperand(i),
                                 PA+SL->getElementOffset(i)));
    } else {
      errs() << "Bad Type: " << *PC->getType() << "\n";
      llvm_unreachable("Unknown constant type to initialize memory with!");
    }
  }
}

//===----------------------------------------------------------------------===//
//                          MachOSym Implementation
//===----------------------------------------------------------------------===//

MachOSym::MachOSym(const GlobalValue *gv, std::string name, uint8_t sect,
                   const MCAsmInfo *MAI) :
  GV(gv), n_strx(0), n_type(sect == NO_SECT ? N_UNDF : N_SECT), n_sect(sect),
  n_desc(0), n_value(0) {

  // FIXME: This is completely broken, it should use the mangler interface.
  switch (GV->getLinkage()) {
  default:
    llvm_unreachable("Unexpected linkage type!");
    break;
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::CommonLinkage:
    assert(!isa<Function>(gv) && "Unexpected linkage type for Function!");
  case GlobalValue::ExternalLinkage:
    GVName = MAI->getGlobalPrefix() + name;
    n_type |= GV->hasHiddenVisibility() ? N_PEXT : N_EXT;
    break;
  case GlobalValue::PrivateLinkage:
    GVName = MAI->getPrivateGlobalPrefix() + name;
    break;
  case GlobalValue::LinkerPrivateLinkage:
    GVName = MAI->getLinkerPrivateGlobalPrefix() + name;
    break;
  case GlobalValue::InternalLinkage:
    GVName = MAI->getGlobalPrefix() + name;
    break;
  }
}

} // end namespace llvm
