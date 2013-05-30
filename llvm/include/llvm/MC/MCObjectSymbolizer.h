//===-- llvm/MC/MCObjectSymbolizer.h --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCObjectSymbolizer class, an MCSymbolizer that is
// backed by an object::ObjectFile.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTSYMBOLIZER_H
#define LLVM_MC_MCOBJECTSYMBOLIZER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCSymbolizer.h"
#include "llvm/Object/ObjectFile.h"
#include <vector>

namespace llvm {

class MCExpr;
class MCInst;
class MCRelocationInfo;
class raw_ostream;

/// \brief An ObjectFile-backed symbolizer.
class MCObjectSymbolizer : public MCSymbolizer {
protected:
  const object::ObjectFile *Obj;

  typedef DenseMap<uint64_t, object::RelocationRef> AddrToRelocMap;
  typedef std::vector<object::SectionRef> SortedSectionList;
  SortedSectionList SortedSections;

  // Map a load address to the first relocation that applies there. As far as I
  // know, if there are several relocations at the exact same address, they are
  // related and the others can be determined from the first that was found in
  // the relocation table. For instance, on x86-64 mach-o, a SUBTRACTOR
  // relocation (referencing the minuend symbol) is followed by an UNSIGNED
  // relocation (referencing the subtrahend symbol).
  AddrToRelocMap AddrToReloc;

  // Helpers around SortedSections.
  SortedSectionList::const_iterator findSectionContaining(uint64_t Addr) const;
  void insertSection(object::SectionRef Section);


  MCObjectSymbolizer(MCContext &Ctx, OwningPtr<MCRelocationInfo> &RelInfo,
                     const object::ObjectFile *Obj);

public:
  /// \name Overridden MCSymbolizer methods:
  /// @{
  bool tryAddingSymbolicOperand(MCInst &MI, raw_ostream &cStream,
                                int64_t Value,
                                uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t InstSize);

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                       int64_t Value, uint64_t Address);
  /// @}

  /// \brief Create an object symbolizer for \p Obj.
  static MCObjectSymbolizer *
    createObjectSymbolizer(MCContext &Ctx, OwningPtr<MCRelocationInfo> &RelInfo,
                           const object::ObjectFile *Obj);
};

}

#endif
