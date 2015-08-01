//===- lib/ReaderWriter/ELF/Mips/MipsDynamicTable.h -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_DYNAMIC_TABLE_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_DYNAMIC_TABLE_H

#include "TargetLayout.h"
#include "SectionChunks.h"

namespace lld {
namespace elf {

template <class ELFT> class MipsTargetLayout;

template <class ELFT> class MipsDynamicTable : public DynamicTable<ELFT> {
public:
  MipsDynamicTable(const ELFLinkingContext &ctx, MipsTargetLayout<ELFT> &layout)
      : DynamicTable<ELFT>(ctx, layout, ".dynamic",
                           TargetLayout<ELFT>::ORDER_DYNAMIC),
        _targetLayout(layout) {}

  void createDefaultEntries() override {
    DynamicTable<ELFT>::createDefaultEntries();

    // Version id for the Runtime Linker Interface.
    this->addEntry(DT_MIPS_RLD_VERSION, 1);

    // The .rld_map section address.
    if (this->_ctx.getOutputELFType() == ET_EXEC) {
      _dt_rldmap = this->addEntry(DT_MIPS_RLD_MAP, 0);
      _dt_rldmaprel = this->addEntry(DT_MIPS_RLD_MAP_REL, 0);
    }

    // MIPS flags.
    this->addEntry(DT_MIPS_FLAGS, RHF_NOTPOT);

    // The base address of the segment.
    _dt_baseaddr = this->addEntry(DT_MIPS_BASE_ADDRESS, 0);

    // Number of local global offset table entries.
    _dt_localgot = this->addEntry(DT_MIPS_LOCAL_GOTNO, 0);

    // Number of entries in the .dynsym section.
    _dt_symtabno = this->addEntry(DT_MIPS_SYMTABNO, 0);

    // The index of the first dynamic symbol table entry that corresponds
    // to an entry in the global offset table.
    _dt_gotsym = this->addEntry(DT_MIPS_GOTSYM, 0);

    // Address of the .got section.
    _dt_pltgot = this->addEntry(DT_PLTGOT, 0);
  }

  void doPreFlight() override {
    DynamicTable<ELFT>::doPreFlight();

    if (_targetLayout.findOutputSection(".MIPS.options")) {
      _dt_options = this->addEntry(DT_MIPS_OPTIONS, 0);
    }
  }

  void updateDynamicTable() override {
    DynamicTable<ELFT>::updateDynamicTable();

    // Assign the minimum segment address to the DT_MIPS_BASE_ADDRESS tag.
    auto baseAddr = std::numeric_limits<uint64_t>::max();
    for (auto si : _targetLayout.segments())
      if (si->segmentType() != llvm::ELF::PT_NULL)
        baseAddr = std::min(baseAddr, si->virtualAddr());
    this->_entries[_dt_baseaddr].d_un.d_val = baseAddr;

    auto &got = _targetLayout.getGOTSection();

    this->_entries[_dt_symtabno].d_un.d_val = this->getSymbolTable()->size();
    this->_entries[_dt_gotsym].d_un.d_val =
        this->getSymbolTable()->size() - got.getGlobalCount();
    this->_entries[_dt_localgot].d_un.d_val = got.getLocalCount();
    this->_entries[_dt_pltgot].d_un.d_ptr = got.virtualAddr();

    if (const auto *sec = _targetLayout.findOutputSection(".MIPS.options"))
      this->_entries[_dt_options].d_un.d_ptr = sec->virtualAddr();

    if (const auto *sec = _targetLayout.findOutputSection(".rld_map")) {
      this->_entries[_dt_rldmap].d_un.d_ptr = sec->virtualAddr();
      this->_entries[_dt_rldmaprel].d_un.d_ptr =
          sec->virtualAddr() -
          (this->virtualAddr() +
           _dt_rldmaprel * sizeof(typename DynamicTable<ELFT>::Elf_Dyn));
    }
  }

  int64_t getGotPltTag() override { return DT_MIPS_PLTGOT; }

protected:
  /// \brief Adjust the symbol's value for microMIPS code.
  uint64_t getAtomVirtualAddress(const AtomLayout *al) const override {
    if (const auto *da = dyn_cast<DefinedAtom>(al->_atom))
      if (da->codeModel() == DefinedAtom::codeMipsMicro ||
          da->codeModel() == DefinedAtom::codeMipsMicroPIC)
        return al->_virtualAddr | 1;
    return al->_virtualAddr;
  }

private:
  std::size_t _dt_symtabno;
  std::size_t _dt_localgot;
  std::size_t _dt_gotsym;
  std::size_t _dt_pltgot;
  std::size_t _dt_baseaddr;
  std::size_t _dt_options;
  std::size_t _dt_rldmap;
  std::size_t _dt_rldmaprel;
  MipsTargetLayout<ELFT> &_targetLayout;
};

} // end namespace elf
} // end namespace lld

#endif
