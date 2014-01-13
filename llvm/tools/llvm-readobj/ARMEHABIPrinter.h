//===--- ARMEHABIPrinter.h - ARM EHABI Unwind Information Printer ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_READOBJ_ARMEHABI_PRINTER_H
#define LLVM_READOBJ_ARMEHABI_PRINTER_H

#include "Error.h"
#include "StreamWriter.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/ARMEHABI.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/type_traits.h"

namespace llvm {
namespace ARM {
namespace EHABI {
template <typename ET>
class PrinterContext {
  StreamWriter &SW;
  const object::ELFFile<ET> *ELF;

  typedef typename object::ELFFile<ET>::Elf_Sym Elf_Sym;
  typedef typename object::ELFFile<ET>::Elf_Shdr Elf_Shdr;

  typedef typename object::ELFFile<ET>::Elf_Rel_Iter Elf_Rel_iterator;
  typedef typename object::ELFFile<ET>::Elf_Sym_Iter Elf_Sym_iterator;
  typedef typename object::ELFFile<ET>::Elf_Shdr_Iter Elf_Shdr_iterator;

  static const size_t IndexTableEntrySize;

  static uint64_t PREL31(uint32_t Address, uint32_t Place) {
    uint64_t Location = Address & 0x7fffffff;
    if (Location & 0x04000000)
      Location |= (uint64_t) ~0x7fffffff;
    return Location + Place;
  }

  ErrorOr<StringRef> FunctionAtAddress(unsigned Section, uint64_t Address) const;
  const Elf_Shdr *FindExceptionTable(unsigned IndexTableIndex,
                                     off_t IndexTableOffset) const;

  void PrintIndexTable(unsigned SectionIndex, const Elf_Shdr *IT) const;
  void PrintExceptionTable(const Elf_Shdr *IT, const Elf_Shdr *EHT,
                           uint64_t TableEntryOffset) const;
  void PrintOpcodes(const uint8_t *Entry, size_t Length, off_t Offset) const;

public:
  PrinterContext(StreamWriter &Writer, const object::ELFFile<ET> *File)
    : SW(Writer), ELF(File) {}

  void PrintUnwindInformation() const;
};

template <typename ET>
const size_t PrinterContext<ET>::IndexTableEntrySize = 8;

template <typename ET>
ErrorOr<StringRef> PrinterContext<ET>::FunctionAtAddress(unsigned Section,
                                                         uint64_t Address) const {
  for (Elf_Sym_iterator SI = ELF->begin_symbols(), SE = ELF->end_symbols();
       SI != SE; ++SI)
    if (SI->st_shndx == Section && SI->st_value == Address &&
        SI->getType() == ELF::STT_FUNC)
      return ELF->getSymbolName(SI);
  return readobj_error::unknown_symbol;
}

template <typename ET>
const typename object::ELFFile<ET>::Elf_Shdr *
PrinterContext<ET>::FindExceptionTable(unsigned IndexSectionIndex,
                                       off_t IndexTableOffset) const {
  /// Iterate through the sections, searching for the relocation section
  /// associated with the unwind index table section specified by
  /// IndexSectionIndex.  Iterate the associated section searching for the
  /// relocation associated with the index table entry specified by
  /// IndexTableOffset.  The symbol is the section symbol for the exception
  /// handling table.  Use this symbol to recover the actual exception handling
  /// table.

  for (Elf_Shdr_iterator SI = ELF->begin_sections(), SE = ELF->end_sections();
       SI != SE; ++SI) {
    if (SI->sh_type == ELF::SHT_REL && SI->sh_info == IndexSectionIndex) {
      for (Elf_Rel_iterator RI = ELF->begin_rel(&*SI), RE = ELF->end_rel(&*SI);
           RI != RE; ++RI) {
        if (RI->r_offset == IndexTableOffset) {
          typename object::ELFFile<ET>::Elf_Rela RelA;
          RelA.r_offset = RI->r_offset;
          RelA.r_info = RI->r_info;
          RelA.r_addend = 0;

          std::pair<const Elf_Shdr *, const Elf_Sym *> Symbol =
            ELF->getRelocationSymbol(&(*SI), &RelA);

          return ELF->getSection(Symbol.second);
        }
      }
    }
  }
  return NULL;
}

template <typename ET>
void PrinterContext<ET>::PrintExceptionTable(const Elf_Shdr *IT,
                                             const Elf_Shdr *EHT,
                                             uint64_t TableEntryOffset) const {
  ErrorOr<ArrayRef<uint8_t> > Contents = ELF->getSectionContents(EHT);
  if (!Contents)
    return;

  /// ARM EHABI Section 6.2 - The generic model
  ///
  /// An exception-handling table entry for the generic model is laid out as:
  ///
  ///  3 3
  ///  1 0                            0
  /// +-+------------------------------+
  /// |0|  personality routine offset  |
  /// +-+------------------------------+
  /// |  personality routine data ...  |
  ///
  ///
  /// ARM EHABI Section 6.3 - The ARM-defined compact model
  ///
  /// An exception-handling table entry for the compact model looks like:
  ///
  ///  3 3 2 2  2 2
  ///  1 0 8 7  4 3                     0
  /// +-+---+----+-----------------------+
  /// |1| 0 | Ix | data for pers routine |
  /// +-+---+----+-----------------------+
  /// |  more personality routine data   |

  const support::ulittle32_t Word =
    *reinterpret_cast<const support::ulittle32_t *>(Contents->data() + TableEntryOffset);

  if (Word & 0x80000000) {
    SW.printString("Model", StringRef("Compact"));

    unsigned PersonalityIndex = (Word & 0x0f000000) >> 24;
    SW.printNumber("PersonalityIndex", PersonalityIndex);

    switch (PersonalityIndex) {
    case AEABI_UNWIND_CPP_PR0:
      llvm_unreachable("Personality 0 should be compact inline!");
      break;
    case AEABI_UNWIND_CPP_PR1:
    case AEABI_UNWIND_CPP_PR2:
      unsigned AdditionalWords = (Word & 0x00ff0000) >> 16;
      PrintOpcodes(Contents->data() + TableEntryOffset, 2 + 4 * AdditionalWords,
                   2);
      break;
    }
  } else {
    SW.printString("Model", StringRef("Generic"));

    uint64_t Address = PREL31(Word, EHT->sh_addr);
    SW.printHex("PersonalityRoutineAddress", Address);
    if (ErrorOr<StringRef> Name = FunctionAtAddress(EHT->sh_link, Address))
      SW.printString("PersonalityRoutineName", *Name);
  }
}

template <typename ET>
void PrinterContext<ET>::PrintOpcodes(const uint8_t *Entry,
                                      size_t Length, off_t Offset) const {
  ListScope OCC(SW, "Opcodes");
  for (unsigned OCI = Offset; OCI < Length + Offset; OCI++)
    SW.printHex("Opcode", Entry[OCI ^ 0x3]);
}

template <typename ET>
void PrinterContext<ET>::PrintIndexTable(unsigned SectionIndex,
                                         const Elf_Shdr *IT) const {
  ErrorOr<ArrayRef<uint8_t> > Contents = ELF->getSectionContents(IT);
  if (!Contents)
    return;

  /// ARM EHABI Section 5 - Index Table Entries
  /// * The first word contains a PREL31 offset to the start of a function with
  ///   bit 31 clear
  /// * The second word contains one of:
  ///   - The PREL31 offset of the start of the table entry for the function,
  ///     with bit 31 clear
  ///   - The exception-handling table entry itself with bit 31 set
  ///   - The special bit pattern EXIDX_CANTUNWIND, indicating that associated
  ///     frames cannot be unwound

  const support::ulittle32_t *Data =
    reinterpret_cast<const support::ulittle32_t *>(Contents->data());
  const unsigned Entries = IT->sh_size / IndexTableEntrySize;

  ListScope E(SW, "Entries");
  for (unsigned Entry = 0; Entry < Entries; ++Entry) {
    DictScope E(SW, "Entry");

    const support::ulittle32_t Word0 =
      Data[Entry * (IndexTableEntrySize / sizeof(*Data)) + 0];
    const support::ulittle32_t Word1 =
      Data[Entry * (IndexTableEntrySize / sizeof(*Data)) + 1];

    if (Word0 & 0x80000000) {
      errs() << "corrupt unwind data in section " << SectionIndex << "\n";
      continue;
    }

    const uint64_t Offset = PREL31(Word0, IT->sh_addr);
    SW.printHex("FunctionAddress", Offset);
    if (ErrorOr<StringRef> Name = FunctionAtAddress(IT->sh_link, Offset))
      SW.printString("FunctionName", *Name);

    if (Word1 == EXIDX_CANTUNWIND) {
      SW.printString("Model", StringRef("CantUnwind"));
      continue;
    }

    if (Word1 & 0x80000000) {
      SW.printString("Model", StringRef("Compact (Inline)"));

      unsigned PersonalityIndex = (Word1 & 0x0f000000) >> 24;
      SW.printNumber("PersonalityIndex", PersonalityIndex);

      PrintOpcodes(Contents->data() + Entry * IndexTableEntrySize + 4, 3, 1);
    } else {
      const Elf_Shdr *EHT =
        FindExceptionTable(SectionIndex, Entry * IndexTableEntrySize + 4);

      if (ErrorOr<StringRef> Name = ELF->getSectionName(EHT))
        SW.printString("ExceptionHandlingTable", *Name);

      uint64_t TableEntryOffset = PREL31(Word1, IT->sh_addr);
      SW.printHex("TableEntryOffset", TableEntryOffset);

      PrintExceptionTable(IT, EHT, TableEntryOffset);
    }
  }
}

template <typename ET>
void PrinterContext<ET>::PrintUnwindInformation() const {
  DictScope UI(SW, "UnwindInformation");

  int SectionIndex = 0;
  for (Elf_Shdr_iterator SI = ELF->begin_sections(), SE = ELF->end_sections();
       SI != SE; ++SI, ++SectionIndex) {
    if (SI->sh_type == ELF::SHT_ARM_EXIDX) {
      const Elf_Shdr *IT = &(*SI);

      DictScope UIT(SW, "UnwindIndexTable");

      SW.printNumber("SectionIndex", SectionIndex);
      if (ErrorOr<StringRef> SectionName = ELF->getSectionName(IT))
        SW.printString("SectionName", *SectionName);
      SW.printHex("SectionOffset", IT->sh_offset);

      PrintIndexTable(SectionIndex, IT);
    }
  }
}
}
}
}

#endif

