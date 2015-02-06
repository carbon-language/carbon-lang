//===--------- lib/ReaderWriter/ELF/ARM/ARMRelocationPass.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the relocation processing pass for ARM. This includes
///   GOT and PLT entries, TLS, COPY, and ifunc.
///
/// This also includes additional behavior that gnu-ld and gold implement but
/// which is not specified anywhere.
///
//===----------------------------------------------------------------------===//

#include "ARMRelocationPass.h"

#include "lld/Core/Simple.h"

#include "llvm/ADT/DenseMap.h"

#include "Atoms.h"
#include "ARMLinkingContext.h"
#include "llvm/Support/Debug.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;

namespace {
class ELFPassFile : public SimpleFile {
public:
  ELFPassFile(const ELFLinkingContext &eti) : SimpleFile("ELFPassFile") {
    setOrdinal(eti.getNextOrdinalAndIncrement());
  }

  llvm::BumpPtrAllocator _alloc;
};

/// \brief CRTP base for handling relocations.
template <class Derived> class ARMRelocationPass : public Pass {
  /// \brief Handle a specific reference.
  void handleReference(const DefinedAtom &atom, const Reference &ref) {
    DEBUG_WITH_TYPE(
        "ARM", llvm::dbgs() << "\t" << LLVM_FUNCTION_NAME << "()"
            << ":   Name of Defined Atom: " << atom.name().str();
        llvm::dbgs() << "   kindValue: " << ref.kindValue() << "\n");
    if (ref.kindNamespace() != Reference::KindNamespace::ELF)
      return;
    assert(ref.kindArch() == Reference::KindArch::ARM);
  }

protected:
public:
  ARMRelocationPass(const ELFLinkingContext &ctx)
      : _file(ctx), _ctx(ctx) {}

  /// \brief Do the pass.
  ///
  /// The goal here is to first process each reference individually. Each call
  /// to handleReference may modify the reference itself and/or create new
  /// atoms which must be stored in one of the maps below.
  ///
  /// After all references are handled, the atoms created during that are all
  /// added to mf.
  void perform(std::unique_ptr<MutableFile> &mf) override {
    ScopedTask task(getDefaultDomain(), "ARM GOT/PLT Pass");
    DEBUG_WITH_TYPE(
        "ARM", llvm::dbgs() << "Undefined Atoms" << "\n";
        for (const auto &atom
             : mf->undefined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Shared Library Atoms" << "\n";
        for (const auto &atom
             : mf->sharedLibrary()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Absolute Atoms" << "\n";
        for (const auto &atom
             : mf->absolute()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        }

        llvm::dbgs() << "Defined Atoms" << "\n";
        for (const auto &atom
             : mf->defined()) {
          llvm::dbgs() << " Name of Atom: " << atom->name().str() << "\n";
        });

    // Process all references.
    for (const auto &atom : mf->defined()) {
      for (const auto &ref : *atom) {
        handleReference(*atom, *ref);
      }
    }
  }

protected:
  /// \brief Owner of all the Atoms created by this pass.
  ELFPassFile _file;
  const ELFLinkingContext &_ctx;
};

/// This implements the static relocation model. Meaning GOT and PLT entries are
/// not created for references that can be directly resolved. These are
/// converted to a direct relocation. For entries that do require a GOT or PLT
/// entry, that entry is statically bound.
///
/// TLS always assumes module 1 and attempts to remove indirection.
class ARMStaticRelocationPass final
    : public ARMRelocationPass<ARMStaticRelocationPass> {
public:
  ARMStaticRelocationPass(const elf::ARMLinkingContext &ctx)
      : ARMRelocationPass(ctx) {}
};

} // end of anon namespace

std::unique_ptr<Pass>
lld::elf::createARMRelocationPass(const ARMLinkingContext &ctx) {
  switch (ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    if (ctx.isDynamic())
      llvm_unreachable("Unhandled output file type");
    return llvm::make_unique<ARMStaticRelocationPass>(ctx);
  default:
    llvm_unreachable("Unhandled output file type");
  }
}
