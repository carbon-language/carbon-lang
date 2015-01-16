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
#include "lld/Core/Atom.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/STDExtras.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>
#include <vector>

namespace lld {
namespace elf {
template <class ELFT> class DynamicTable;
template <class ELFT> class DynamicSymbolTable;
template <class ELFT> class ELFDefinedAtom;
template <class ELFT> class ELFReference;
class ELFWriter;
template <class ELFT> class ELFHeader;
template <class ELFT> class Section;
template <class ELFT> class TargetLayout;

class TargetRelocationHandler {
public:
  /// Constructor
  TargetRelocationHandler(ELFLinkingContext &targetInfo) : _ctx(targetInfo) {}
  virtual ~TargetRelocationHandler() {}

  virtual std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                          const lld::AtomLayout &,
                                          const Reference &) const = 0;

protected:
  void unhandledReferenceType(const Atom &atom, const Reference &ref) const {
    llvm::errs() << "Unhandled reference type in file " << atom.file().path()
                 << ": reference from " << atom.name() << "+"
                 << ref.offsetInAtom() << " to " << ref.target()->name() << "+"
                 << ref.addend() << " of type ";

    StringRef kindValStr;
    if (!_ctx.registry().referenceKindToString(
            ref.kindNamespace(), ref.kindArch(), ref.kindValue(), kindValStr)) {
      kindValStr = "unknown";
    }

    llvm::errs() << ref.kindValue() << " (" << kindValStr << ")\n";
    llvm::report_fatal_error("unhandled reference type");
  }

private:
  ELFLinkingContext &_ctx;
};

/// \brief TargetHandler contains all the information responsible to handle a
/// a particular target on ELF. A target might wish to override implementation
/// of creating atoms and how the atoms are written to the output file.
template <class ELFT> class TargetHandler : public TargetHandlerBase {
public:
  /// The layout determined completely by the Target.
  virtual TargetLayout<ELFT> &getTargetLayout() = 0;

  /// Determine how relocations need to be applied.
  virtual const TargetRelocationHandler &getRelocationHandler() const = 0;

  /// How does the target deal with reading input files.
  virtual std::unique_ptr<Reader> getObjReader(bool) = 0;

  /// How does the target deal with reading dynamic libraries.
  virtual std::unique_ptr<Reader> getDSOReader(bool) = 0;

  /// How does the target deal with writing ELF output.
  virtual std::unique_ptr<Writer> getWriter() = 0;
};
} // end namespace elf
} // end namespace lld

#endif
