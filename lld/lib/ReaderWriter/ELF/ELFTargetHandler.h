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
#include <unordered_map>

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

  /// Returns the target Section for a section name and content Type
  Section<ELFT> *getSection(const StringRef name,
                            DefinedAtom::ContentPermissions permissions) = 0;

private:
  const ELFTargetInfo &_targetInfo;
  const DefaultELFLayout<ELFT> &_layout;
};

/// \brief An interface to override functions that are provided by the 
/// the default ELF Layout
template <class ELFT> class ELFTargetHandler : public ELFTargetHandlerBase {

public:
  ELFTargetHandler(ELFTargetInfo &targetInfo) : _targetInfo(targetInfo) {}

  /// Register a Target, so that the target backend may choose on how to merge
  /// individual atoms within the section, this is a way to control output order
  /// of atoms that is determined by the target
  void registerTargetSection(StringRef name,
                             DefinedAtom::ContentPermissions perm) {
    const TargetSectionKey targetSection(name, perm);
    if (_registeredTargetSections.find(targetSection) ==
        _registeredTargetSections.end())
      _registeredTargetSections.insert(std::make_pair(targetSection, true));
  }

  /// Check if the section is registered given the section name and its
  /// contentType, if they are registered the target would need to 
  /// create a section so that atoms insert, atom virtual address assignment
  /// could be overridden and controlled by the Target
  bool isSectionRegisteredByTarget(StringRef name,
                                   DefinedAtom::ContentPermissions perm) {
    const TargetSectionKey targetSection(name, perm);
    if (_registeredTargetSections.find(targetSection) ==
        _registeredTargetSections.end())
      return false;
    return true;
  }

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

private:
  struct TargetSectionKey {
    TargetSectionKey(StringRef name, DefinedAtom::ContentPermissions perm)
        : _name(name), _perm(perm) {
    }

    // Data members
    const StringRef _name;
    DefinedAtom::ContentPermissions _perm;
  };

  struct TargetSectionKeyHash {
    int64_t operator()(const TargetSectionKey &k) const {
      return llvm::hash_combine(k._name, k._perm);
    }
  };

  struct TargetSectionKeyEq {
    bool operator()(const TargetSectionKey &lhs,
                    const TargetSectionKey &rhs) const {
      return ((lhs._name == rhs._name) && (lhs._perm == rhs._perm));
    }
  };

  typedef std::unordered_map<TargetSectionKey, bool, TargetSectionKeyHash,
                             TargetSectionKeyEq> RegisteredTargetSectionMapT;
  typedef typename RegisteredTargetSectionMapT::iterator RegisteredTargetSectionMapIterT;

protected:
  const ELFTargetInfo &_targetInfo;
  RegisteredTargetSectionMapT _registeredTargetSections;
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_TARGETHANDLER_H
