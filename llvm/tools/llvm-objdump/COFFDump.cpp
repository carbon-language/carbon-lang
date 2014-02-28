//===-- COFFDump.cpp - COFF-specific dumper ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the COFF-specific dumper for llvm-objdump.
/// It outputs the Win64 EH data structures as plain text.
/// The encoding of the unwind codes is described in MSDN:
/// http://msdn.microsoft.com/en-us/library/ck9asaa9.aspx
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Win64EH.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>

using namespace llvm;
using namespace object;
using namespace llvm::Win64EH;

// Returns the name of the unwind code.
static StringRef getUnwindCodeTypeName(uint8_t Code) {
  switch(Code) {
  default: llvm_unreachable("Invalid unwind code");
  case UOP_PushNonVol: return "UOP_PushNonVol";
  case UOP_AllocLarge: return "UOP_AllocLarge";
  case UOP_AllocSmall: return "UOP_AllocSmall";
  case UOP_SetFPReg: return "UOP_SetFPReg";
  case UOP_SaveNonVol: return "UOP_SaveNonVol";
  case UOP_SaveNonVolBig: return "UOP_SaveNonVolBig";
  case UOP_SaveXMM128: return "UOP_SaveXMM128";
  case UOP_SaveXMM128Big: return "UOP_SaveXMM128Big";
  case UOP_PushMachFrame: return "UOP_PushMachFrame";
  }
}

// Returns the name of a referenced register.
static StringRef getUnwindRegisterName(uint8_t Reg) {
  switch(Reg) {
  default: llvm_unreachable("Invalid register");
  case 0: return "RAX";
  case 1: return "RCX";
  case 2: return "RDX";
  case 3: return "RBX";
  case 4: return "RSP";
  case 5: return "RBP";
  case 6: return "RSI";
  case 7: return "RDI";
  case 8: return "R8";
  case 9: return "R9";
  case 10: return "R10";
  case 11: return "R11";
  case 12: return "R12";
  case 13: return "R13";
  case 14: return "R14";
  case 15: return "R15";
  }
}

// Calculates the number of array slots required for the unwind code.
static unsigned getNumUsedSlots(const UnwindCode &UnwindCode) {
  switch (UnwindCode.getUnwindOp()) {
  default: llvm_unreachable("Invalid unwind code");
  case UOP_PushNonVol:
  case UOP_AllocSmall:
  case UOP_SetFPReg:
  case UOP_PushMachFrame:
    return 1;
  case UOP_SaveNonVol:
  case UOP_SaveXMM128:
    return 2;
  case UOP_SaveNonVolBig:
  case UOP_SaveXMM128Big:
    return 3;
  case UOP_AllocLarge:
    return (UnwindCode.getOpInfo() == 0) ? 2 : 3;
  }
}

// Prints one unwind code. Because an unwind code can occupy up to 3 slots in
// the unwind codes array, this function requires that the correct number of
// slots is provided.
static void printUnwindCode(ArrayRef<UnwindCode> UCs) {
  assert(UCs.size() >= getNumUsedSlots(UCs[0]));
  outs() <<  format("    0x%02x: ", unsigned(UCs[0].u.CodeOffset))
         << getUnwindCodeTypeName(UCs[0].getUnwindOp());
  switch (UCs[0].getUnwindOp()) {
  case UOP_PushNonVol:
    outs() << " " << getUnwindRegisterName(UCs[0].getOpInfo());
    break;
  case UOP_AllocLarge:
    if (UCs[0].getOpInfo() == 0) {
      outs() << " " << UCs[1].FrameOffset;
    } else {
      outs() << " " << UCs[1].FrameOffset
                       + (static_cast<uint32_t>(UCs[2].FrameOffset) << 16);
    }
    break;
  case UOP_AllocSmall:
    outs() << " " << ((UCs[0].getOpInfo() + 1) * 8);
    break;
  case UOP_SetFPReg:
    outs() << " ";
    break;
  case UOP_SaveNonVol:
    outs() << " " << getUnwindRegisterName(UCs[0].getOpInfo())
           << format(" [0x%04x]", 8 * UCs[1].FrameOffset);
    break;
  case UOP_SaveNonVolBig:
    outs() << " " << getUnwindRegisterName(UCs[0].getOpInfo())
           << format(" [0x%08x]", UCs[1].FrameOffset
                    + (static_cast<uint32_t>(UCs[2].FrameOffset) << 16));
    break;
  case UOP_SaveXMM128:
    outs() << " XMM" << static_cast<uint32_t>(UCs[0].getOpInfo())
           << format(" [0x%04x]", 16 * UCs[1].FrameOffset);
    break;
  case UOP_SaveXMM128Big:
    outs() << " XMM" << UCs[0].getOpInfo()
           << format(" [0x%08x]", UCs[1].FrameOffset
                           + (static_cast<uint32_t>(UCs[2].FrameOffset) << 16));
    break;
  case UOP_PushMachFrame:
    outs() << " " << (UCs[0].getOpInfo() ? "w/o" : "w")
           << " error code";
    break;
  }
  outs() << "\n";
}

static void printAllUnwindCodes(ArrayRef<UnwindCode> UCs) {
  for (const UnwindCode *I = UCs.begin(), *E = UCs.end(); I < E; ) {
    unsigned UsedSlots = getNumUsedSlots(*I);
    if (UsedSlots > UCs.size()) {
      outs() << "Unwind data corrupted: Encountered unwind op "
             << getUnwindCodeTypeName((*I).getUnwindOp())
             << " which requires " << UsedSlots
             << " slots, but only " << UCs.size()
             << " remaining in buffer";
      return ;
    }
    printUnwindCode(ArrayRef<UnwindCode>(I, E));
    I += UsedSlots;
  }
}

// Given a symbol sym this functions returns the address and section of it.
static error_code resolveSectionAndAddress(const COFFObjectFile *Obj,
                                           const SymbolRef &Sym,
                                           const coff_section *&ResolvedSection,
                                           uint64_t &ResolvedAddr) {
  if (error_code EC = Sym.getAddress(ResolvedAddr))
    return EC;
  section_iterator iter(Obj->section_begin());
  if (error_code EC = Sym.getSection(iter))
    return EC;
  ResolvedSection = Obj->getCOFFSection(iter);
  return object_error::success;
}

// Given a vector of relocations for a section and an offset into this section
// the function returns the symbol used for the relocation at the offset.
static error_code resolveSymbol(const std::vector<RelocationRef> &Rels,
                                uint64_t Offset, SymbolRef &Sym) {
  for (std::vector<RelocationRef>::const_iterator I = Rels.begin(),
                                                  E = Rels.end();
                                                  I != E; ++I) {
    uint64_t Ofs;
    if (error_code EC = I->getOffset(Ofs))
      return EC;
    if (Ofs == Offset) {
      Sym = *I->getSymbol();
      break;
    }
  }
  return object_error::success;
}

// Given a vector of relocations for a section and an offset into this section
// the function resolves the symbol used for the relocation at the offset and
// returns the section content and the address inside the content pointed to
// by the symbol.
static error_code getSectionContents(const COFFObjectFile *Obj,
                                     const std::vector<RelocationRef> &Rels,
                                     uint64_t Offset,
                                     ArrayRef<uint8_t> &Contents,
                                     uint64_t &Addr) {
  SymbolRef Sym;
  if (error_code ec = resolveSymbol(Rels, Offset, Sym)) return ec;
  const coff_section *Section;
  if (error_code EC = resolveSectionAndAddress(Obj, Sym, Section, Addr))
    return EC;
  if (error_code EC = Obj->getSectionContents(Section, Contents))
    return EC;
  return object_error::success;
}

// Given a vector of relocations for a section and an offset into this section
// the function returns the name of the symbol used for the relocation at the
// offset.
static error_code resolveSymbolName(const std::vector<RelocationRef> &Rels,
                                    uint64_t Offset, StringRef &Name) {
  SymbolRef Sym;
  if (error_code EC = resolveSymbol(Rels, Offset, Sym))
    return EC;
  if (error_code EC = Sym.getName(Name))
    return EC;
  return object_error::success;
}

static void printCOFFSymbolAddress(llvm::raw_ostream &Out,
                                   const std::vector<RelocationRef> &Rels,
                                   uint64_t Offset, uint32_t Disp) {
  StringRef Sym;
  if (error(resolveSymbolName(Rels, Offset, Sym)))
    return;
  Out << Sym;
  if (Disp > 0)
    Out << format(" + 0x%04x", Disp);
}

static void
printSEHTable(const COFFObjectFile *Obj, uint32_t TableVA, int Count) {
  if (Count == 0)
    return;

  const pe32_header *PE32Header;
  if (error(Obj->getPE32Header(PE32Header)))
    return;
  uint32_t ImageBase = PE32Header->ImageBase;
  uintptr_t IntPtr = 0;
  if (error(Obj->getVaPtr(TableVA, IntPtr)))
    return;
  const support::ulittle32_t *P = (const support::ulittle32_t *)IntPtr;
  outs() << "SEH Table:";
  for (int I = 0; I < Count; ++I)
    outs() << format(" 0x%x", P[I] + ImageBase);
  outs() << "\n\n";
}

static void printLoadConfiguration(const COFFObjectFile *Obj) {
  // Skip if it's not executable.
  const pe32_header *PE32Header;
  if (error(Obj->getPE32Header(PE32Header)))
    return;
  if (!PE32Header)
    return;

  const coff_file_header *Header;
  if (error(Obj->getCOFFHeader(Header)))
    return;
  // Currently only x86 is supported
  if (Header->Machine != COFF::IMAGE_FILE_MACHINE_I386)
    return;

  const data_directory *DataDir;
  if (error(Obj->getDataDirectory(COFF::LOAD_CONFIG_TABLE, DataDir)))
    return;
  uintptr_t IntPtr = 0;
  if (DataDir->RelativeVirtualAddress == 0)
    return;
  if (error(Obj->getRvaPtr(DataDir->RelativeVirtualAddress, IntPtr)))
    return;

  const coff_load_configuration32 *LoadConf =
      reinterpret_cast<const coff_load_configuration32 *>(IntPtr);

  outs() << "Load configuration:"
         << "\n  Timestamp: " << LoadConf->TimeDateStamp
         << "\n  Major Version: " << LoadConf->MajorVersion
         << "\n  Minor Version: " << LoadConf->MinorVersion
         << "\n  GlobalFlags Clear: " << LoadConf->GlobalFlagsClear
         << "\n  GlobalFlags Set: " << LoadConf->GlobalFlagsSet
         << "\n  Critical Section Default Timeout: " << LoadConf->CriticalSectionDefaultTimeout
         << "\n  Decommit Free Block Threshold: " << LoadConf->DeCommitFreeBlockThreshold
         << "\n  Decommit Total Free Threshold: " << LoadConf->DeCommitTotalFreeThreshold
         << "\n  Lock Prefix Table: " << LoadConf->LockPrefixTable
         << "\n  Maximum Allocation Size: " << LoadConf->MaximumAllocationSize
         << "\n  Virtual Memory Threshold: " << LoadConf->VirtualMemoryThreshold
         << "\n  Process Affinity Mask: " << LoadConf->ProcessAffinityMask
         << "\n  Process Heap Flags: " << LoadConf->ProcessHeapFlags
         << "\n  CSD Version: " << LoadConf->CSDVersion
         << "\n  Security Cookie: " << LoadConf->SecurityCookie
         << "\n  SEH Table: " << LoadConf->SEHandlerTable
         << "\n  SEH Count: " << LoadConf->SEHandlerCount
         << "\n\n";
  printSEHTable(Obj, LoadConf->SEHandlerTable, LoadConf->SEHandlerCount);
  outs() << "\n";
}

// Prints import tables. The import table is a table containing the list of
// DLL name and symbol names which will be linked by the loader.
static void printImportTables(const COFFObjectFile *Obj) {
  import_directory_iterator I = Obj->import_directory_begin();
  import_directory_iterator E = Obj->import_directory_end();
  if (I == E)
    return;
  outs() << "The Import Tables:\n";
  for (; I != E; I = ++I) {
    const import_directory_table_entry *Dir;
    StringRef Name;
    if (I->getImportTableEntry(Dir)) return;
    if (I->getName(Name)) return;

    outs() << format("  lookup %08x time %08x fwd %08x name %08x addr %08x\n\n",
                     static_cast<uint32_t>(Dir->ImportLookupTableRVA),
                     static_cast<uint32_t>(Dir->TimeDateStamp),
                     static_cast<uint32_t>(Dir->ForwarderChain),
                     static_cast<uint32_t>(Dir->NameRVA),
                     static_cast<uint32_t>(Dir->ImportAddressTableRVA));
    outs() << "    DLL Name: " << Name << "\n";
    outs() << "    Hint/Ord  Name\n";
    const import_lookup_table_entry32 *entry;
    if (I->getImportLookupEntry(entry))
      return;
    for (; entry->data; ++entry) {
      if (entry->isOrdinal()) {
        outs() << format("      % 6d\n", entry->getOrdinal());
        continue;
      }
      uint16_t Hint;
      StringRef Name;
      if (Obj->getHintName(entry->getHintNameRVA(), Hint, Name))
        return;
      outs() << format("      % 6d  ", Hint) << Name << "\n";
    }
    outs() << "\n";
  }
}

// Prints export tables. The export table is a table containing the list of
// exported symbol from the DLL.
static void printExportTable(const COFFObjectFile *Obj) {
  outs() << "Export Table:\n";
  export_directory_iterator I = Obj->export_directory_begin();
  export_directory_iterator E = Obj->export_directory_end();
  if (I == E)
    return;
  StringRef DllName;
  uint32_t OrdinalBase;
  if (I->getDllName(DllName))
    return;
  if (I->getOrdinalBase(OrdinalBase))
    return;
  outs() << " DLL name: " << DllName << "\n";
  outs() << " Ordinal base: " << OrdinalBase << "\n";
  outs() << " Ordinal      RVA  Name\n";
  for (; I != E; I = ++I) {
    uint32_t Ordinal;
    if (I->getOrdinal(Ordinal))
      return;
    uint32_t RVA;
    if (I->getExportRVA(RVA))
      return;
    outs() << format("    % 4d %# 8x", Ordinal, RVA);

    StringRef Name;
    if (I->getSymbolName(Name))
      continue;
    if (!Name.empty())
      outs() << "  " << Name;
    outs() << "\n";
  }
}

void llvm::printCOFFUnwindInfo(const COFFObjectFile *Obj) {
  const coff_file_header *Header;
  if (error(Obj->getCOFFHeader(Header))) return;

  if (Header->Machine != COFF::IMAGE_FILE_MACHINE_AMD64) {
    errs() << "Unsupported image machine type "
              "(currently only AMD64 is supported).\n";
    return;
  }

  const coff_section *Pdata = 0;

  for (section_iterator SI = Obj->section_begin(), SE = Obj->section_end();
       SI != SE; ++SI) {
    StringRef Name;
    if (error(SI->getName(Name))) continue;

    if (Name != ".pdata") continue;

    Pdata = Obj->getCOFFSection(SI);
    std::vector<RelocationRef> Rels;
    for (relocation_iterator RI = SI->relocation_begin(),
                             RE = SI->relocation_end();
         RI != RE; ++RI)
      Rels.push_back(*RI);

    // Sort relocations by address.
    std::sort(Rels.begin(), Rels.end(), RelocAddressLess);

    ArrayRef<uint8_t> Contents;
    if (error(Obj->getSectionContents(Pdata, Contents))) continue;
    if (Contents.empty()) continue;

    ArrayRef<RuntimeFunction> RFs(
                  reinterpret_cast<const RuntimeFunction *>(Contents.data()),
                                  Contents.size() / sizeof(RuntimeFunction));
    for (const RuntimeFunction *I = RFs.begin(), *E = RFs.end(); I < E; ++I) {
      const uint64_t SectionOffset = std::distance(RFs.begin(), I)
                                     * sizeof(RuntimeFunction);

      outs() << "Function Table:\n";

      outs() << "  Start Address: ";
      printCOFFSymbolAddress(outs(), Rels, SectionOffset +
                             /*offsetof(RuntimeFunction, StartAddress)*/ 0,
                             I->StartAddress);
      outs() << "\n";

      outs() << "  End Address: ";
      printCOFFSymbolAddress(outs(), Rels, SectionOffset +
                             /*offsetof(RuntimeFunction, EndAddress)*/ 4,
                             I->EndAddress);
      outs() << "\n";

      outs() << "  Unwind Info Address: ";
      printCOFFSymbolAddress(outs(), Rels, SectionOffset +
                             /*offsetof(RuntimeFunction, UnwindInfoOffset)*/ 8,
                             I->UnwindInfoOffset);
      outs() << "\n";

      ArrayRef<uint8_t> XContents;
      uint64_t UnwindInfoOffset = 0;
      if (error(getSectionContents(Obj, Rels, SectionOffset +
                              /*offsetof(RuntimeFunction, UnwindInfoOffset)*/ 8,
                                   XContents, UnwindInfoOffset))) continue;
      if (XContents.empty()) continue;

      UnwindInfoOffset += I->UnwindInfoOffset;
      if (UnwindInfoOffset > XContents.size()) continue;

      const Win64EH::UnwindInfo *UI =
                            reinterpret_cast<const Win64EH::UnwindInfo *>
                              (XContents.data() + UnwindInfoOffset);

      // The casts to int are required in order to output the value as number.
      // Without the casts the value would be interpreted as char data (which
      // results in garbage output).
      outs() << "  Version: " << static_cast<int>(UI->getVersion()) << "\n";
      outs() << "  Flags: " << static_cast<int>(UI->getFlags());
      if (UI->getFlags()) {
          if (UI->getFlags() & UNW_ExceptionHandler)
            outs() << " UNW_ExceptionHandler";
          if (UI->getFlags() & UNW_TerminateHandler)
            outs() << " UNW_TerminateHandler";
          if (UI->getFlags() & UNW_ChainInfo)
            outs() << " UNW_ChainInfo";
      }
      outs() << "\n";
      outs() << "  Size of prolog: "
             << static_cast<int>(UI->PrologSize) << "\n";
      outs() << "  Number of Codes: "
             << static_cast<int>(UI->NumCodes) << "\n";
      // Maybe this should move to output of UOP_SetFPReg?
      if (UI->getFrameRegister()) {
        outs() << "  Frame register: "
                << getUnwindRegisterName(UI->getFrameRegister())
                << "\n";
        outs() << "  Frame offset: "
                << 16 * UI->getFrameOffset()
                << "\n";
      } else {
        outs() << "  No frame pointer used\n";
      }
      if (UI->getFlags() & (UNW_ExceptionHandler | UNW_TerminateHandler)) {
        // FIXME: Output exception handler data
      } else if (UI->getFlags() & UNW_ChainInfo) {
        // FIXME: Output chained unwind info
      }

      if (UI->NumCodes)
        outs() << "  Unwind Codes:\n";

      printAllUnwindCodes(ArrayRef<UnwindCode>(&UI->UnwindCodes[0],
                          UI->NumCodes));

      outs() << "\n\n";
      outs().flush();
    }
  }
}

void llvm::printCOFFFileHeader(const object::ObjectFile *Obj) {
  const COFFObjectFile *file = dyn_cast<const COFFObjectFile>(Obj);
  printLoadConfiguration(file);
  printImportTables(file);
  printExportTable(file);
}
