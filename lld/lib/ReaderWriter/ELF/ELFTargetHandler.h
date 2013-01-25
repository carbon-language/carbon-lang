//===- lib/ReaderWriter/ELF/ELFTargetHandler.h -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_TARGETHANDLER_H
#define LLD_READER_WRITER_ELF_TARGETHANDLER_H

#include "lld/Core/LinkerOptions.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"
#include "DefaultELFLayout.h"
#include "AtomsELF.h"

#include <memory>
#include <vector>

/// \brief All ELF targets would want to override the way the ELF file gets
/// processed by the linker. This class serves as an interface which would be
/// used to derive the needed functionality of a particular target/platform.

/// \brief The target registers a set of handlers for overriding target specific
/// attributes for a DefinedAtom. The Reader uses this class to query for the
/// type of atom and its permissions 

namespace lld {

template <class ELFT> class ELFDefinedAtom;

namespace elf {

template <class ELFT> class ELFTargetAtomHandler {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  virtual DefinedAtom::ContentType contentType(
      const lld::ELFDefinedAtom<ELFT> *atom) const {
    return atom->contentType();
  }

  virtual DefinedAtom::ContentType contentType(const Elf_Sym *sym) const {
    return DefinedAtom::typeZeroFill;
  }

  virtual DefinedAtom::ContentPermissions contentPermissions(
      const lld::ELFDefinedAtom<ELFT> *atom) const {
    return atom->permissions();
  }
};

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

private:
  const ELFTargetInfo &_targetInfo;
  const DefaultELFLayout<ELFT> &_layout;
};

/// \brief An interface to override functions that are provided by the 
/// the default ELF Layout
template <class ELFT> class ELFTargetHandler : public ELFTargetHandlerBase {

public:

  ELFTargetHandler(ELFTargetInfo &targetInfo) : _targetInfo(targetInfo) {}

  /// If the target overrides ELF header information, this API would
  /// return true, so that the target can set all fields specific to
  /// that target
  virtual bool doesOverrideELFHeader() = 0;

  /// Set the ELF Header information 
  virtual void setELFHeaderInfo(ELFHeader<ELFT> *elfHeader) = 0;

  /// ELFTargetLayout 
  virtual ELFTargetLayout<ELFT> &targetLayout() = 0;

  /// ELFTargetAtomHandler
  virtual ELFTargetAtomHandler<ELFT> &targetAtomHandler() = 0;

  /// Create a set of Default target sections that a target might needj
  virtual void createDefaultSections() = 0;

  /// \brief Add a section to the current Layout
  virtual void addSection(Section<ELFT> *section) = 0;

  /// \brief add new symbol file 
  virtual void addFiles(InputFiles &) = 0;

  /// \brief Finalize the symbol values
  virtual void finalizeSymbolValues() = 0;

  /// \brief allocate Commons, some architectures may move small common
  /// symbols over to small data, this would also be used 
  virtual void allocateCommons() = 0;

protected:
  const ELFTargetInfo &_targetInfo;
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_TARGETHANDLER_H
