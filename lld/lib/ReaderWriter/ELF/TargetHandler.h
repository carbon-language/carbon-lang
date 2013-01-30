//===- lib/ReaderWriter/ELF/TargetHandler.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief These interfaces provide target specific hooks to change the linker's
/// behaivor.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_TARGET_HANDLER_H

#include "Layout.h"

#include "lld/Core/InputFiles.h"
#include "lld/Core/LinkerOptions.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FileOutputBuffer.h"

#include <memory>
#include <vector>
#include <unordered_map>

namespace lld {
namespace elf {
template <class ELFT> class ELFDefinedAtom;
class ELFWriter;
template <class ELFT> class Header;
template <class ELFT> class Section;
template <class ELFT> class TargetLayout;

/// \brief The target registers a set of handlers for overriding target specific
/// attributes for a DefinedAtom. The Reader uses this class to query for the
/// type of atom and its permissions 
template <class ELFT> class TargetAtomHandler {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  virtual DefinedAtom::ContentType contentType(
      const ELFDefinedAtom<ELFT> *atom) const {
    return atom->contentType();
  }

  virtual DefinedAtom::ContentType contentType(const Elf_Sym *sym) const {
    return DefinedAtom::typeZeroFill;
  }

  virtual DefinedAtom::ContentPermissions contentPermissions(
      const ELFDefinedAtom<ELFT> *atom) const {
    return atom->permissions();
  }
};

template <class ELFT> class TargetRelocationHandler {
public:
  virtual ErrorOr<void> applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                        const AtomLayout &,
                                        const Reference &)const = 0;
};

/// \brief An interface to override functions that are provided by the 
/// the default ELF Layout
template <class ELFT> class TargetHandler : public TargetHandlerBase {

public:
  TargetHandler(ELFTargetInfo &targetInfo) : _targetInfo(targetInfo) {}

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
  virtual bool doesOverrideHeader() = 0;

  /// Set the ELF Header information 
  virtual void setHeaderInfo(Header<ELFT> *Header) = 0;

  /// TargetLayout 
  virtual TargetLayout<ELFT> &targetLayout() = 0;

  /// TargetAtomHandler
  virtual TargetAtomHandler<ELFT> &targetAtomHandler() = 0;

  virtual const TargetRelocationHandler<ELFT> &getRelocationHandler() const = 0;

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
} // end namespace elf
} // end namespace lld

#endif
