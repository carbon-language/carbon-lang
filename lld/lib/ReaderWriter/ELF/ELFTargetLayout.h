//===- lib/ReaderWriter/ELF/ELFTargetLayout.h -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_LAYOUT_H
#define LLD_READER_WRITER_ELF_TARGET_LAYOUT_H

#include "DefaultELFLayout.h"

#include "lld/Core/LLVM.h"

namespace lld {
namespace elf {
/// \brief The target can override certain functions in the DefaultELFLayout
/// class so that the order, the name of the section and the segment type could
/// be changed in the final layout
template <class ELFT> class ELFTargetLayout : public DefaultELFLayout<ELFT> {
public:
  ELFTargetLayout(ELFTargetInfo &targetInfo, DefaultELFLayout<ELFT> &layout)
      : _targetInfo(targetInfo), _layout(layout) {
  }

  /// isTargetSection provides a way to determine if the section that
  /// we are processing has been registered by the target and the target
  /// wants to handle them. 
  /// For example: the Writer may be processing a section but the target
  /// might want to override the functionality on how atoms are inserted
  /// into the section. Such sections are set the K_TargetSection flag in
  /// the SectionKind after they are created
  virtual bool isTargetSection(const StringRef name, const int32_t contentType,
                               const int32_t contentPermissions) = 0;

  /// The target may want to override the sectionName to a different
  /// section Name in the output
  virtual StringRef sectionName(const StringRef name, const int32_t contentType,
                                const int32_t contentPermissions) = 0;

  /// The target may want to override the section order that has been 
  /// set by the DefaultLayout
  virtual ELFLayout::SectionOrder getSectionOrder(
      const StringRef name, int32_t contentType,
      int32_t contentPermissions) = 0;

  /// The target can set the segment type for a Section
  virtual ELFLayout::SegmentType segmentType(Section<ELFT> *section) const = 0;

  /// Returns true/false depending on whether the section has a Output
  //  segment or not
  bool hasOutputSegment(Section<ELFT> *section) = 0;

  /// Returns the target Section for a section name and content Type
  Section<ELFT> *getSection(const StringRef name,
                            DefinedAtom::ContentPermissions permissions) = 0;

private:
  const ELFTargetInfo &_targetInfo;
  const DefaultELFLayout<ELFT> &_layout;
};
} // end namespace elf
} // end namespace lld

#endif
