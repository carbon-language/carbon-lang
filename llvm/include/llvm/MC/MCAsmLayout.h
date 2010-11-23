//===- MCAsmLayout.h - Assembly Layout Object -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMLAYOUT_H
#define LLVM_MC_MCASMLAYOUT_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {
class MCAssembler;
class MCFragment;
class MCSectionData;
class MCSymbolData;

/// Encapsulates the layout of an assembly file at a particular point in time.
///
/// Assembly may requiring compute multiple layouts for a particular assembly
/// file as part of the relaxation process. This class encapsulates the layout
/// at a single point in time in such a way that it is always possible to
/// efficiently compute the exact addresses of any symbol in the assembly file,
/// even during the relaxation process.
class MCAsmLayout {
public:
  typedef llvm::SmallVectorImpl<MCSectionData*>::const_iterator const_iterator;
  typedef llvm::SmallVectorImpl<MCSectionData*>::iterator iterator;

private:
  MCAssembler &Assembler;

  /// List of sections in layout order.
  llvm::SmallVector<MCSectionData*, 16> SectionOrder;

  /// The last fragment which was layed out, or 0 if nothing has been layed
  /// out. Fragments are always layed out in order, so all fragments with a
  /// lower ordinal will be up to date.
  mutable MCFragment *LastValidFragment;

  /// \brief Make sure that the layout for the given fragment is valid, lazily
  /// computing it if necessary.
  void EnsureValid(const MCFragment *F) const;

  bool isSectionUpToDate(const MCSectionData *SD) const;
  bool isFragmentUpToDate(const MCFragment *F) const;

public:
  MCAsmLayout(MCAssembler &_Assembler);

  /// Get the assembler object this is a layout for.
  MCAssembler &getAssembler() const { return Assembler; }

  /// \brief Invalidate all following fragments because a fragment has been resized. The
  /// fragments size should have already been updated.
  void Invalidate(MCFragment *F);

  /// \brief Update the layout, replacing Src with Dst. The contents
  /// of Src and Dst are not modified, and must be copied by the caller.
  /// Src will be removed from the layout, but not deleted.
  void ReplaceFragment(MCFragment *Src, MCFragment *Dst);

  /// \brief Update the layout to coalesce Src into Dst. The contents
  /// of Src and Dst are not modified, and must be coalesced by the caller.
  /// Src will be removed from the layout, but not deleted.
  void CoalesceFragments(MCFragment *Src, MCFragment *Dst);

  /// \brief Perform a full layout.
  void LayoutFile();

  /// \brief Perform layout for a single fragment, assuming that the previous
  /// fragment has already been layed out correctly, and the parent section has
  /// been initialized.
  void LayoutFragment(MCFragment *Fragment);

  /// \brief Performs initial layout for a single section, assuming that the
  /// previous section (including its fragments) has already been layed out
  /// correctly.
  void LayoutSection(MCSectionData *SD);

  /// @name Section Access (in layout order)
  /// @{

  llvm::SmallVectorImpl<MCSectionData*> &getSectionOrder() {
    return SectionOrder;
  }
  const llvm::SmallVectorImpl<MCSectionData*> &getSectionOrder() const {
    return SectionOrder;
  }

  /// @}
  /// @name Fragment Layout Data
  /// @{

  /// \brief Get the effective size of the given fragment, as computed in the
  /// current layout.
  uint64_t getFragmentEffectiveSize(const MCFragment *F) const;

  /// \brief Get the offset of the given fragment inside its containing section.
  uint64_t getFragmentOffset(const MCFragment *F) const;

  /// @}
  /// @name Section Layout Data
  /// @{

  /// \brief Get the computed address of the given section.
  uint64_t getSectionAddress(const MCSectionData *SD) const;

  /// @}
  /// @name Utility Functions
  /// @{

  /// \brief Get the address of the given fragment, as computed in the current
  /// layout.
  uint64_t getFragmentAddress(const MCFragment *F) const;

  /// \brief Get the address space size of the given section, as it effects
  /// layout. This may differ from the size reported by \see getSectionSize() by
  /// not including section tail padding.
  uint64_t getSectionAddressSize(const MCSectionData *SD) const;

  /// \brief Get the data size of the given section, as emitted to the object
  /// file. This may include additional padding, or be 0 for virtual sections.
  uint64_t getSectionFileSize(const MCSectionData *SD) const;

  /// \brief Get the logical data size of the given section.
  uint64_t getSectionSize(const MCSectionData *SD) const;

  /// \brief Get the address of the given symbol, as computed in the current
  /// layout.
  uint64_t getSymbolAddress(const MCSymbolData *SD) const;

  /// @}
};

} // end namespace llvm

#endif
