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

#include "DefaultLayout.h"
#include "SectionChunks.h"

namespace lld {
namespace elf {

class MipsLinkingContext;
template <class ELFType> class MipsTargetLayout;

template <class MipsELFType>
class MipsDynamicTable : public DynamicTable<MipsELFType> {
public:
  MipsDynamicTable(MipsLinkingContext &context,
                   MipsTargetLayout<MipsELFType> &layout)
      : DynamicTable<MipsELFType>(context, layout, ".dynamic",
                                  DefaultLayout<MipsELFType>::ORDER_DYNAMIC),
        _mipsTargetLayout(layout) {}

  virtual void createDefaultEntries() {
    DynamicTable<MipsELFType>::createDefaultEntries();

    typename DynamicTable<MipsELFType>::Elf_Dyn dyn;

    // Version id for the Runtime Linker Interface.
    dyn.d_un.d_val = 1;
    dyn.d_tag = DT_MIPS_RLD_VERSION;
    this->addEntry(dyn);

    // MIPS flags.
    dyn.d_un.d_val = RHF_NOTPOT;
    dyn.d_tag = DT_MIPS_FLAGS;
    this->addEntry(dyn);

    // The base address of the segment.
    dyn.d_un.d_ptr = 0;
    dyn.d_tag = DT_MIPS_BASE_ADDRESS;
    _dt_baseaddr = this->addEntry(dyn);

    // Number of local global offset table entries.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_MIPS_LOCAL_GOTNO;
    _dt_localgot = this->addEntry(dyn);

    // Number of entries in the .dynsym section.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_MIPS_SYMTABNO;
    _dt_symtabno = this->addEntry(dyn);

    // The index of the first dynamic symbol table entry that corresponds
    // to an entry in the global offset table.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_MIPS_GOTSYM;
    _dt_gotsym = this->addEntry(dyn);

    // Address of the .got section.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_PLTGOT;
    _dt_pltgot = this->addEntry(dyn);
  }

  virtual void updateDynamicTable() {
    DynamicTable<MipsELFType>::updateDynamicTable();

    // Assign the minimum segment address to the DT_MIPS_BASE_ADDRESS tag.
    auto baseAddr = std::numeric_limits<uint64_t>::max();
    for (auto si : _mipsTargetLayout.segments())
      if (si->segmentType() != llvm::ELF::PT_NULL)
        baseAddr = std::min(baseAddr, si->virtualAddr());
    this->_entries[_dt_baseaddr].d_un.d_val = baseAddr;

    auto &got = _mipsTargetLayout.getGOTSection();

    this->_entries[_dt_symtabno].d_un.d_val = this->getSymbolTable()->size();
    this->_entries[_dt_gotsym].d_un.d_val =
       this-> getSymbolTable()->size() - got.getGlobalCount();
    this->_entries[_dt_localgot].d_un.d_val = got.getLocalCount();
    this->_entries[_dt_pltgot].d_un.d_ptr =
        _mipsTargetLayout.findOutputSection(".got")->virtualAddr();
  }

  virtual int64_t getGotPltTag() { return DT_MIPS_PLTGOT; }

private:
  std::size_t _dt_symtabno;
  std::size_t _dt_localgot;
  std::size_t _dt_gotsym;
  std::size_t _dt_pltgot;
  std::size_t _dt_baseaddr;
  MipsTargetLayout<MipsELFType> &_mipsTargetLayout;
};

} // end namespace elf
} // end namespace lld

#endif
