//===-- macho-dump.cpp - Mach Object Dumping Tool -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC/Mach-O LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachO.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
ShowSectionData("dump-section-data", cl::desc("Dump the contents of sections"),
                cl::init(false));

///

static const char *ProgramName;

static void Message(const char *Type, const Twine &Msg) {
  errs() << ProgramName << ": " << Type << ": " << Msg << "\n";
}

static int Error(const Twine &Msg) {
  Message("error", Msg);
  return 1;
}

static void Warning(const Twine &Msg) {
  Message("warning", Msg);
}

///

static void DumpSegmentCommandData(StringRef Name,
                                   uint64_t VMAddr, uint64_t VMSize,
                                   uint64_t FileOffset, uint64_t FileSize,
                                   uint32_t MaxProt, uint32_t InitProt,
                                   uint32_t NumSections, uint32_t Flags) {
  outs() << "  ('segment_name', '";
  outs().write_escaped(Name, /*UseHexEscapes=*/true) << "')\n";
  outs() << "  ('vm_addr', " << VMAddr << ")\n";
  outs() << "  ('vm_size', " << VMSize << ")\n";
  outs() << "  ('file_offset', " << FileOffset << ")\n";
  outs() << "  ('file_size', " << FileSize << ")\n";
  outs() << "  ('maxprot', " << MaxProt << ")\n";
  outs() << "  ('initprot', " << InitProt << ")\n";
  outs() << "  ('num_sections', " << NumSections << ")\n";
  outs() << "  ('flags', " << Flags << ")\n";
}

static int DumpSectionData(const MachOObjectFile &Obj, unsigned Index,
                           StringRef Name,
                           StringRef SegmentName, uint64_t Address,
                           uint64_t Size, uint32_t Offset,
                           uint32_t Align, uint32_t RelocationTableOffset,
                           uint32_t NumRelocationTableEntries,
                           uint32_t Flags, uint32_t Reserved1,
                           uint32_t Reserved2, uint64_t Reserved3 = ~0ULL) {
  outs() << "    # Section " << Index << "\n";
  outs() << "   (('section_name', '";
  outs().write_escaped(Name, /*UseHexEscapes=*/true) << "')\n";
  outs() << "    ('segment_name', '";
  outs().write_escaped(SegmentName, /*UseHexEscapes=*/true) << "')\n";
  outs() << "    ('address', " << Address << ")\n";
  outs() << "    ('size', " << Size << ")\n";
  outs() << "    ('offset', " << Offset << ")\n";
  outs() << "    ('alignment', " << Align << ")\n";
  outs() << "    ('reloc_offset', " << RelocationTableOffset << ")\n";
  outs() << "    ('num_reloc', " << NumRelocationTableEntries << ")\n";
  outs() << "    ('flags', " << format("0x%x", Flags) << ")\n";
  outs() << "    ('reserved1', " << Reserved1 << ")\n";
  outs() << "    ('reserved2', " << Reserved2 << ")\n";
  if (Reserved3 != ~0ULL)
    outs() << "    ('reserved3', " << Reserved3 << ")\n";
  outs() << "   ),\n";

  // Dump the relocation entries.
  outs() << "  ('_relocations', [\n";
  unsigned RelNum = 0;
  for (relocation_iterator I = Obj.section_rel_begin(Index),
                           E = Obj.section_rel_end(Index);
       I != E; ++I, ++RelNum) {
    MachO::any_relocation_info RE = Obj.getRelocation(I->getRawDataRefImpl());
    outs() << "    # Relocation " << RelNum << "\n";
    outs() << "    (('word-0', " << format("0x%x", RE.r_word0) << "),\n";
    outs() << "     ('word-1', " << format("0x%x", RE.r_word1) << ")),\n";
  }
  outs() << "  ])\n";

  // Dump the section data, if requested.
  if (ShowSectionData) {
    outs() << "  ('_section_data', '";
    StringRef Data = Obj.getData().substr(Offset, Size);
    for (unsigned i = 0; i != Data.size(); ++i) {
      if (i && (i % 4) == 0)
        outs() << ' ';
      outs() << hexdigit((Data[i] >> 4) & 0xF, /*LowerCase=*/true);
      outs() << hexdigit((Data[i] >> 0) & 0xF, /*LowerCase=*/true);
    }
    outs() << "')\n";
  }

  return 0;
}

static int DumpSegmentCommand(const MachOObjectFile &Obj,
                              const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::segment_command SLC = Obj.getSegmentLoadCommand(LCI);

  DumpSegmentCommandData(StringRef(SLC.segname, 16), SLC.vmaddr,
                         SLC.vmsize, SLC.fileoff, SLC.filesize,
                         SLC.maxprot, SLC.initprot, SLC.nsects, SLC.flags);

  // Dump the sections.
  outs() << "  ('sections', [\n";
  for (unsigned i = 0; i != SLC.nsects; ++i) {
    MachO::section Sect = Obj.getSection(LCI, i);
    DumpSectionData(Obj, i, StringRef(Sect.sectname, 16),
                    StringRef(Sect.segname, 16), Sect.addr,
                    Sect.size, Sect.offset, Sect.align,
                    Sect.reloff, Sect.nreloc, Sect.flags,
                    Sect.reserved1, Sect.reserved2);
  }
  outs() << "  ])\n";

  return 0;
}

static int DumpSegment64Command(const MachOObjectFile &Obj,
                                const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::segment_command_64 SLC = Obj.getSegment64LoadCommand(LCI);
  DumpSegmentCommandData(StringRef(SLC.segname, 16), SLC.vmaddr,
                         SLC.vmsize, SLC.fileoff, SLC.filesize,
                         SLC.maxprot, SLC.initprot, SLC.nsects, SLC.flags);

  // Dump the sections.
  outs() << "  ('sections', [\n";
  for (unsigned i = 0; i != SLC.nsects; ++i) {
    MachO::section_64 Sect = Obj.getSection64(LCI, i);

    DumpSectionData(Obj, i, StringRef(Sect.sectname, 16),
                    StringRef(Sect.segname, 16), Sect.addr,
                    Sect.size, Sect.offset, Sect.align,
                    Sect.reloff, Sect.nreloc, Sect.flags,
                    Sect.reserved1, Sect.reserved2,
                    Sect.reserved3);
  }
  outs() << "  ])\n";

  return 0;
}

static void DumpSymbolTableEntryData(const MachOObjectFile &Obj,
                                     unsigned Index, uint32_t StringIndex,
                                     uint8_t Type, uint8_t SectionIndex,
                                     uint16_t Flags, uint64_t Value,
                                     StringRef StringTable) {
  const char *Name = &StringTable.data()[StringIndex];
  outs() << "    # Symbol " << Index << "\n";
  outs() << "   (('n_strx', " << StringIndex << ")\n";
  outs() << "    ('n_type', " << format("0x%x", Type) << ")\n";
  outs() << "    ('n_sect', " << uint32_t(SectionIndex) << ")\n";
  outs() << "    ('n_desc', " << Flags << ")\n";
  outs() << "    ('n_value', " << Value << ")\n";
  outs() << "    ('_string', '" << Name << "')\n";
  outs() << "   ),\n";
}

static int DumpSymtabCommand(const MachOObjectFile &Obj) {
  MachO::symtab_command SLC = Obj.getSymtabLoadCommand();

  outs() << "  ('symoff', " << SLC.symoff << ")\n";
  outs() << "  ('nsyms', " << SLC.nsyms << ")\n";
  outs() << "  ('stroff', " << SLC.stroff << ")\n";
  outs() << "  ('strsize', " << SLC.strsize << ")\n";

  // Dump the string data.
  outs() << "  ('_string_data', '";
  StringRef StringTable = Obj.getStringTableData();
  outs().write_escaped(StringTable,
                       /*UseHexEscapes=*/true) << "')\n";

  // Dump the symbol table.
  outs() << "  ('_symbols', [\n";
  unsigned SymNum = 0;
  for (const SymbolRef &Symbol : Obj.symbols()) {
    DataRefImpl DRI = Symbol.getRawDataRefImpl();
    if (Obj.is64Bit()) {
      MachO::nlist_64 STE = Obj.getSymbol64TableEntry(DRI);
      DumpSymbolTableEntryData(Obj, SymNum, STE.n_strx, STE.n_type,
                               STE.n_sect, STE.n_desc, STE.n_value,
                               StringTable);
    } else {
      MachO::nlist STE = Obj.getSymbolTableEntry(DRI);
      DumpSymbolTableEntryData(Obj, SymNum, STE.n_strx, STE.n_type,
                               STE.n_sect, STE.n_desc, STE.n_value,
                               StringTable);
    }
    SymNum++;
  }
  outs() << "  ])\n";

  return 0;
}

static int DumpDysymtabCommand(const MachOObjectFile &Obj) {
  MachO::dysymtab_command DLC = Obj.getDysymtabLoadCommand();

  outs() << "  ('ilocalsym', " << DLC.ilocalsym << ")\n";
  outs() << "  ('nlocalsym', " << DLC.nlocalsym << ")\n";
  outs() << "  ('iextdefsym', " << DLC.iextdefsym << ")\n";
  outs() << "  ('nextdefsym', " << DLC.nextdefsym << ")\n";
  outs() << "  ('iundefsym', " << DLC.iundefsym << ")\n";
  outs() << "  ('nundefsym', " << DLC.nundefsym << ")\n";
  outs() << "  ('tocoff', " << DLC.tocoff << ")\n";
  outs() << "  ('ntoc', " << DLC.ntoc << ")\n";
  outs() << "  ('modtaboff', " << DLC.modtaboff << ")\n";
  outs() << "  ('nmodtab', " << DLC.nmodtab << ")\n";
  outs() << "  ('extrefsymoff', " << DLC.extrefsymoff << ")\n";
  outs() << "  ('nextrefsyms', " << DLC.nextrefsyms << ")\n";
  outs() << "  ('indirectsymoff', " << DLC.indirectsymoff << ")\n";
  outs() << "  ('nindirectsyms', " << DLC.nindirectsyms << ")\n";
  outs() << "  ('extreloff', " << DLC.extreloff << ")\n";
  outs() << "  ('nextrel', " << DLC.nextrel << ")\n";
  outs() << "  ('locreloff', " << DLC.locreloff << ")\n";
  outs() << "  ('nlocrel', " << DLC.nlocrel << ")\n";

  // Dump the indirect symbol table.
  outs() << "  ('_indirect_symbols', [\n";
  for (unsigned i = 0; i != DLC.nindirectsyms; ++i) {
    uint32_t ISTE = Obj.getIndirectSymbolTableEntry(DLC, i);
    outs() << "    # Indirect Symbol " << i << "\n";
    outs() << "    (('symbol_index', " << format("0x%x", ISTE) << "),),\n";
  }
  outs() << "  ])\n";

  return 0;
}

static int
DumpLinkeditDataCommand(const MachOObjectFile &Obj,
                        const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::linkedit_data_command LLC = Obj.getLinkeditDataLoadCommand(LCI);
  outs() << "  ('dataoff', " << LLC.dataoff << ")\n"
         << "  ('datasize', " << LLC.datasize << ")\n"
         << "  ('_addresses', [\n";

  SmallVector<uint64_t, 8> Addresses;
  Obj.ReadULEB128s(LLC.dataoff, Addresses);
  for (unsigned i = 0, e = Addresses.size(); i != e; ++i)
    outs() << "    # Address " << i << '\n'
           << "    ('address', " << format("0x%x", Addresses[i]) << "),\n";

  outs() << "  ])\n";

  return 0;
}

static int
DumpDataInCodeDataCommand(const MachOObjectFile &Obj,
                          const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::linkedit_data_command LLC = Obj.getLinkeditDataLoadCommand(LCI);
  outs() << "  ('dataoff', " << LLC.dataoff << ")\n"
         << "  ('datasize', " << LLC.datasize << ")\n"
         << "  ('_data_regions', [\n";

  unsigned NumRegions = LLC.datasize / sizeof(MachO::data_in_code_entry);
  for (unsigned i = 0; i < NumRegions; ++i) {
    MachO::data_in_code_entry DICE= Obj.getDataInCodeTableEntry(LLC.dataoff, i);
    outs() << "    # DICE " << i << "\n"
           << "    ('offset', " << DICE.offset << ")\n"
           << "    ('length', " << DICE.length << ")\n"
           << "    ('kind', " << DICE.kind << ")\n";
  }

  outs() <<"  ])\n";

  return 0;
}

static int
DumpLinkerOptionsCommand(const MachOObjectFile &Obj,
                         const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::linker_option_command LOLC = Obj.getLinkerOptionLoadCommand(LCI);
  outs() << "  ('count', " << LOLC.count << ")\n"
         << "  ('_strings', [\n";

  uint64_t DataSize = LOLC.cmdsize - sizeof(MachO::linker_option_command);
  const char *P = LCI.Ptr + sizeof(MachO::linker_option_command);
  StringRef Data(P, DataSize);
  for (unsigned i = 0; i != LOLC.count; ++i) {
    std::pair<StringRef,StringRef> Split = Data.split('\0');
    outs() << "\t\"";
    outs().write_escaped(Split.first);
    outs() << "\",\n";
    Data = Split.second;
  }
  outs() <<"  ])\n";

  return 0;
}

static int
DumpVersionMin(const MachOObjectFile &Obj,
               const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::version_min_command VMLC = Obj.getVersionMinLoadCommand(LCI);
  outs() << "  ('version, " << VMLC.version << ")\n"
         << "  ('sdk, " << VMLC.sdk << ")\n";
  return 0;
}

static int
DumpDylibID(const MachOObjectFile &Obj,
            const MachOObjectFile::LoadCommandInfo &LCI) {
  MachO::dylib_command DLLC = Obj.getDylibIDLoadCommand(LCI);
  outs() << "  ('install_name', '" << LCI.Ptr + DLLC.dylib.name << "')\n"
         << "  ('timestamp, " << DLLC.dylib.timestamp << ")\n"
         << "  ('cur_version, " << DLLC.dylib.current_version << ")\n"
         << "  ('compat_version, " << DLLC.dylib.compatibility_version << ")\n";
  return 0;
}

static int DumpLoadCommand(const MachOObjectFile &Obj,
                           const MachOObjectFile::LoadCommandInfo &LCI) {
  switch (LCI.C.cmd) {
  case MachO::LC_SEGMENT:
    return DumpSegmentCommand(Obj, LCI);
  case MachO::LC_SEGMENT_64:
    return DumpSegment64Command(Obj, LCI);
  case MachO::LC_SYMTAB:
    return DumpSymtabCommand(Obj);
  case MachO::LC_DYSYMTAB:
    return DumpDysymtabCommand(Obj);
  case MachO::LC_CODE_SIGNATURE:
  case MachO::LC_SEGMENT_SPLIT_INFO:
  case MachO::LC_FUNCTION_STARTS:
    return DumpLinkeditDataCommand(Obj, LCI);
  case MachO::LC_DATA_IN_CODE:
    return DumpDataInCodeDataCommand(Obj, LCI);
  case MachO::LC_LINKER_OPTION:
    return DumpLinkerOptionsCommand(Obj, LCI);
  case MachO::LC_VERSION_MIN_IPHONEOS:
  case MachO::LC_VERSION_MIN_MACOSX:
    return DumpVersionMin(Obj, LCI);
  case MachO::LC_ID_DYLIB:
    return DumpDylibID(Obj, LCI);
  default:
    Warning("unknown load command: " + Twine(LCI.C.cmd));
    return 0;
  }
}

static int DumpLoadCommand(const MachOObjectFile &Obj, unsigned Index,
                           const MachOObjectFile::LoadCommandInfo &LCI) {
  outs() << "  # Load Command " << Index << "\n"
         << " (('command', " << LCI.C.cmd << ")\n"
         << "  ('size', " << LCI.C.cmdsize << ")\n";
  int Res = DumpLoadCommand(Obj, LCI);
  outs() << " ),\n";
  return Res;
}

static void printHeader(const MachOObjectFile *Obj,
                        const MachO::mach_header &Header) {
  outs() << "('cputype', " << Header.cputype << ")\n";
  outs() << "('cpusubtype', " << Header.cpusubtype << ")\n";
  outs() << "('filetype', " << Header.filetype << ")\n";
  outs() << "('num_load_commands', " << Header.ncmds << ")\n";
  outs() << "('load_commands_size', " << Header.sizeofcmds << ")\n";
  outs() << "('flag', " << Header.flags << ")\n";

  // Print extended header if 64-bit.
  if (Obj->is64Bit()) {
    const MachO::mach_header_64 *Header64 =
      reinterpret_cast<const MachO::mach_header_64 *>(&Header);
    outs() << "('reserved', " << Header64->reserved << ")\n";
  }
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm Mach-O dumping tool\n");

  ErrorOr<OwningBinary<Binary>> BinaryOrErr = createBinary(InputFile);
  if (std::error_code EC = BinaryOrErr.getError())
    return Error("unable to read input: '" + EC.message() + "'");
  Binary &Binary = *BinaryOrErr.get().getBinary();

  const MachOObjectFile *InputObject = dyn_cast<MachOObjectFile>(&Binary);
  if (!InputObject)
    return Error("Not a MachO object");

  // Print the header
  MachO::mach_header_64 Header64;
  MachO::mach_header *Header = reinterpret_cast<MachO::mach_header*>(&Header64);
  if (InputObject->is64Bit())
    Header64 = InputObject->getHeader64();
  else
    *Header = InputObject->getHeader();
  printHeader(InputObject, *Header);

  // Print the load commands.
  int Res = 0;
  unsigned Index = 0;
  outs() << "('load_commands', [\n";
  for (const auto &Load : InputObject->load_commands()) {
    if (DumpLoadCommand(*InputObject, Index++, Load))
      break;
  }
  outs() << "])\n";

  return Res;
}
