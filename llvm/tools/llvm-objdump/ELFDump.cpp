//===-- ELFDump.cpp - ELF-specific dumper -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the ELF-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

template<class ELFT>
void printProgramHeaders(
    const ELFObjectFile<ELFT> *o) {
  typedef ELFObjectFile<ELFT> ELFO;
  outs() << "Program Header:\n";
  for (typename ELFO::Elf_Phdr_Iter pi = o->begin_program_headers(),
                                    pe = o->end_program_headers();
                                    pi != pe; ++pi) {
    switch (pi->p_type) {
    case ELF::PT_LOAD:
      outs() << "    LOAD ";
      break;
    case ELF::PT_GNU_STACK:
      outs() << "   STACK ";
      break;
    case ELF::PT_GNU_EH_FRAME:
      outs() << "EH_FRAME ";
      break;
    case ELF::PT_INTERP:
      outs() << "  INTERP ";
      break;
    case ELF::PT_DYNAMIC:
      outs() << " DYNAMIC ";
      break;
    case ELF::PT_PHDR:
      outs() << "    PHDR ";
      break;
    case ELF::PT_TLS:
      outs() << "    TLS ";
      break;
    default:
      outs() << " UNKNOWN ";
    }

    const char *Fmt = ELFT::Is64Bits ? "0x%016" PRIx64 " " : "0x%08" PRIx64 " ";

    outs() << "off    "
           << format(Fmt, (uint64_t)pi->p_offset)
           << "vaddr "
           << format(Fmt, (uint64_t)pi->p_vaddr)
           << "paddr "
           << format(Fmt, (uint64_t)pi->p_paddr)
           << format("align 2**%u\n", countTrailingZeros<uint64_t>(pi->p_align))
           << "         filesz "
           << format(Fmt, (uint64_t)pi->p_filesz)
           << "memsz "
           << format(Fmt, (uint64_t)pi->p_memsz)
           << "flags "
           << ((pi->p_flags & ELF::PF_R) ? "r" : "-")
           << ((pi->p_flags & ELF::PF_W) ? "w" : "-")
           << ((pi->p_flags & ELF::PF_X) ? "x" : "-")
           << "\n";
  }
  outs() << "\n";
}

void llvm::printELFFileHeader(const object::ObjectFile *Obj) {
  // Little-endian 32-bit
  if (const ELF32LEObjectFile *ELFObj = dyn_cast<ELF32LEObjectFile>(Obj))
    printProgramHeaders(ELFObj);

  // Big-endian 32-bit
  if (const ELF32BEObjectFile *ELFObj = dyn_cast<ELF32BEObjectFile>(Obj))
    printProgramHeaders(ELFObj);

  // Little-endian 64-bit
  if (const ELF64LEObjectFile *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj))
    printProgramHeaders(ELFObj);

  // Big-endian 64-bit
  if (const ELF64BEObjectFile *ELFObj = dyn_cast<ELF64BEObjectFile>(Obj))
    printProgramHeaders(ELFObj);
}
