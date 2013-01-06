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

template<endianness target_endianness, std::size_t max_alignment, bool is64Bits>
void printProgramHeaders(
    const ELFObjectFile<target_endianness, max_alignment, is64Bits> *o) {
  typedef ELFObjectFile<target_endianness, max_alignment, is64Bits> ELFO;
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
    default:
      outs() << " UNKNOWN ";
    }

    const char *Fmt = is64Bits ? "0x%016" PRIx64 " " : "0x%08" PRIx64 " ";

    outs() << "off    "
           << format(Fmt, (uint64_t)pi->p_offset)
           << "vaddr "
           << format(Fmt, (uint64_t)pi->p_vaddr)
           << "paddr "
           << format(Fmt, (uint64_t)pi->p_paddr)
           << format("align 2**%u\n", CountTrailingZeros_64(pi->p_align))
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
  if (const ELFObjectFile<support::little, 4, false> *ELFObj =
          dyn_cast<ELFObjectFile<support::little, 4, false> >(Obj))
    printProgramHeaders(ELFObj);

  // Big-endian 32-bit
  if (const ELFObjectFile<support::big, 4, false> *ELFObj =
          dyn_cast<ELFObjectFile<support::big, 4, false> >(Obj))
    printProgramHeaders(ELFObj);

  // Little-endian 64-bit
  if (const ELFObjectFile<support::little, 8, true> *ELFObj =
          dyn_cast<ELFObjectFile<support::little, 8, true> >(Obj))
    printProgramHeaders(ELFObj);

  // Big-endian 64-bit
  if (const ELFObjectFile<support::big, 8, true> *ELFObj =
          dyn_cast<ELFObjectFile<support::big, 8, true> >(Obj))
    printProgramHeaders(ELFObj);
}
