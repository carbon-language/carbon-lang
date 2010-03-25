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
private:
  MCAssembler &Assembler;

public:
  MCAsmLayout(MCAssembler &_Assembler) : Assembler(_Assembler) {}

  /// Get the assembler object this is a layout for.
  MCAssembler &getAssembler() const { return Assembler; }

  /// \brief Update the layout because a fragment has been resized. The
  /// fragments size should have already been updated, the \arg SlideAmount is
  /// the delta from the old size.
  void UpdateForSlide(MCFragment *F, int SlideAmount);

  /// @name Fragment Layout Data
  /// @{

  /// \brief Get the effective size of the given fragment, as computed in the
  /// current layout.
  uint64_t getFragmentEffectiveSize(const MCFragment *F) const;

  /// \brief Set the effective size of the given fragment.
  void setFragmentEffectiveSize(MCFragment *F, uint64_t Value);

  /// \brief Get the offset of the given fragment inside its containing section.
  uint64_t getFragmentOffset(const MCFragment *F) const;

  /// \brief Set the offset of the given fragment inside its containing section.
  void setFragmentOffset(MCFragment *F, uint64_t Value);

  /// @}
  /// @name Section Layout Data
  /// @{

  /// \brief Get the computed address of the given section.
  uint64_t getSectionAddress(const MCSectionData *SD) const;

  /// \brief Set the computed address of the given section.
  void setSectionAddress(MCSectionData *SD, uint64_t Value);

  /// \brief Get the data size of the given section, as emitted to the object
  /// file. This may include additional padding, or be 0 for virtual sections.
  uint64_t getSectionFileSize(const MCSectionData *SD) const;

  /// \brief Set the data size of the given section.
  void setSectionFileSize(MCSectionData *SD, uint64_t Value);

  /// \brief Get the actual data size of the given section.
  uint64_t getSectionSize(const MCSectionData *SD) const;

  /// \brief Set the actual data size of the given section.
  void setSectionSize(MCSectionData *SD, uint64_t Value);

  /// @}
  /// @name Utility Functions
  /// @{

  /// \brief Get the address of the given fragment, as computed in the current
  /// layout.
  uint64_t getFragmentAddress(const MCFragment *F) const;

  /// \brief Get the address of the given symbol, as computed in the current
  /// layout.
  uint64_t getSymbolAddress(const MCSymbolData *SD) const;

  /// @}
};

} // end namespace llvm

#endif
