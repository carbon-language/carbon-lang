//===- llvm-readobj/ELF.cpp - ELF Specific Dumper -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-readobj.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"

namespace llvm {
using namespace object;
using namespace ELF;

const char *getTypeString(uint64_t Type) {
  switch (Type) {
  case DT_BIND_NOW:
    return "(BIND_NOW)";
  case DT_DEBUG:
    return "(DEBUG)";
  case DT_FINI:
    return "(FINI)";
  case DT_FINI_ARRAY:
    return "(FINI_ARRAY)";
  case DT_FINI_ARRAYSZ:
    return "(FINI_ARRAYSZ)";
  case DT_FLAGS:
    return "(FLAGS)";
  case DT_HASH:
    return "(HASH)";
  case DT_INIT:
    return "(INIT)";
  case DT_INIT_ARRAY:
    return "(INIT_ARRAY)";
  case DT_INIT_ARRAYSZ:
    return "(INIT_ARRAYSZ)";
  case DT_PREINIT_ARRAY:
    return "(PREINIT_ARRAY)";
  case DT_PREINIT_ARRAYSZ:
    return "(PREINIT_ARRAYSZ)";
  case DT_JMPREL:
    return "(JMPREL)";
  case DT_NEEDED:
    return "(NEEDED)";
  case DT_NULL:
    return "(NULL)";
  case DT_PLTGOT:
    return "(PLTGOT)";
  case DT_PLTREL:
    return "(PLTREL)";
  case DT_PLTRELSZ:
    return "(PLTRELSZ)";
  case DT_REL:
    return "(REL)";
  case DT_RELA:
    return "(RELA)";
  case DT_RELENT:
    return "(RELENT)";
  case DT_RELSZ:
    return "(RELSZ)";
  case DT_RELAENT:
    return "(RELAENT)";
  case DT_RELASZ:
    return "(RELASZ)";
  case DT_RPATH:
    return "(RPATH)";
  case DT_RUNPATH:
    return "(RUNPATH)";
  case DT_SONAME:
    return "(SONAME)";
  case DT_STRSZ:
    return "(STRSZ)";
  case DT_STRTAB:
    return "(STRTAB)";
  case DT_SYMBOLIC:
    return "(SYMBOLIC)";
  case DT_SYMENT:
    return "(SYMENT)";
  case DT_SYMTAB:
    return "(SYMTAB)";
  case DT_TEXTREL:
    return "(TEXTREL)";
  default:
    return "unknown";
  }
}

template <class ELFT>
void printValue(const ELFObjectFile<ELFT> *O, uint64_t Type, uint64_t Value,
                bool Is64, raw_ostream &OS) {
  switch (Type) {
  case DT_PLTREL:
    if (Value == DT_REL) {
      OS << "REL";
      break;
    } else if (Value == DT_RELA) {
      OS << "RELA";
      break;
    }
  // Fallthrough.
  case DT_PLTGOT:
  case DT_HASH:
  case DT_STRTAB:
  case DT_SYMTAB:
  case DT_RELA:
  case DT_INIT:
  case DT_FINI:
  case DT_REL:
  case DT_JMPREL:
  case DT_INIT_ARRAY:
  case DT_FINI_ARRAY:
  case DT_PREINIT_ARRAY:
  case DT_DEBUG:
  case DT_NULL:
    OS << format("0x%" PRIx64, Value);
    break;
  case DT_PLTRELSZ:
  case DT_RELASZ:
  case DT_RELAENT:
  case DT_STRSZ:
  case DT_SYMENT:
  case DT_RELSZ:
  case DT_RELENT:
  case DT_INIT_ARRAYSZ:
  case DT_FINI_ARRAYSZ:
  case DT_PREINIT_ARRAYSZ:
    OS << Value << " (bytes)";
    break;
  case DT_NEEDED:
    OS << "Shared library: ["
       << O->getString(O->getDynamicStringTableSectionHeader(), Value) << "]";
    break;
  case DT_SONAME:
    OS << "Library soname: ["
       << O->getString(O->getDynamicStringTableSectionHeader(), Value) << "]";
    break;
  }
}

template <class ELFT>
ErrorOr<void> dumpDynamicTable(const ELFObjectFile<ELFT> *O, raw_ostream &OS) {
  typedef ELFObjectFile<ELFT> ELFO;
  typedef typename ELFO::Elf_Dyn_iterator EDI;
  EDI Start = O->begin_dynamic_table(),
      End = O->end_dynamic_table(true);

  if (Start == End)
    return error_code::success();

  ptrdiff_t Total = std::distance(Start, End);
  OS << "Dynamic section contains " << Total << " entries\n";

  bool Is64 = O->getBytesInAddress() == 8;

  OS << "  Tag" << (Is64 ? "                " : "        ") << "Type"
     << "                 " << "Name/Value\n";
  for (; Start != End; ++Start) {
    OS << " "
       << format(Is64 ? "0x%016" PRIx64 : "0x%08" PRIx64, Start->getTag())
       << " " << format("%-21s", getTypeString(Start->getTag()));
    printValue(O, Start->getTag(), Start->getVal(), Is64, OS);
    OS << "\n";
  }

  OS << "  Total: " << Total << "\n\n";
  return error_code::success();
}

ErrorOr<void> dumpELFDynamicTable(ObjectFile *O, raw_ostream &OS) {
  // Little-endian 32-bit
  if (const ELFObjectFile<ELFType<support::little, 4, false> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::little, 4, false> > >(O))
    return dumpDynamicTable(ELFObj, OS);

  // Big-endian 32-bit
  if (const ELFObjectFile<ELFType<support::big, 4, false> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::big, 4, false> > >(O))
    return dumpDynamicTable(ELFObj, OS);

  // Little-endian 64-bit
  if (const ELFObjectFile<ELFType<support::little, 8, true> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::little, 8, true> > >(O))
    return dumpDynamicTable(ELFObj, OS);

  // Big-endian 64-bit
  if (const ELFObjectFile<ELFType<support::big, 8, true> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::big, 8, true> > >(O))
    return dumpDynamicTable(ELFObj, OS);
  return error_code(object_error::invalid_file_type);
}
} // end namespace llvm
