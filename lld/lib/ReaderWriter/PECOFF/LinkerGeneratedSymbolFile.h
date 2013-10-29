//===- lib/ReaderWriter/PECOFF/LinkerGeneratedSymbolFile.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "GroupedSectionsPass.h"
#include "IdataPass.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Path.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/PassManager.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Simple.h"
#include "lld/ReaderWriter/Writer.h"

namespace lld {
namespace coff {

namespace {

// A virtual file containing absolute symbol __ImageBase. __ImageBase (or
// ___ImageBase on x86) is a linker-generated symbol whose address is the same
// as the image base address.
//
// This is the only member file of LinkerGeneratedSymbolFile.
class MemberFile : public SimpleFile {
public:
  MemberFile(const PECOFFLinkingContext &ctx)
      : SimpleFile(ctx, "Member of the Linker Internal File"),
        _imageBaseAtom(*this, ctx.decorateSymbol("__ImageBase"),
                       Atom::scopeGlobal, ctx.getBaseAddress()) {
    addAtom(_imageBaseAtom);
  };

  bool contains(StringRef name) const {
    return _imageBaseAtom.name() == name;
  }

private:
  COFFAbsoluteAtom _imageBaseAtom;
};

} // anonymous namespace

// A pseudo library file to wrap MemberFile, which in turn wraps ImageBaseAtom.
// The file the core linker handle is this.
//
// The reason why we don't pass MemberFile to the core linker is because, if we
// did so, ImageBaseAtom would always be emit to the resultant executable. By
// wrapping the file by a library file, we made it to emit ImageBaseAtom only
// when the atom is really referenced.
class LinkerGeneratedSymbolFile : public ArchiveLibraryFile {
public:
  LinkerGeneratedSymbolFile(const PECOFFLinkingContext &context)
      : ArchiveLibraryFile(context, "Linker Internal File"),
        _memberFile(context) {};

  virtual const File *find(StringRef name, bool dataSymbolOnly) const {
    if (_memberFile.contains(name))
      return &_memberFile;
    return nullptr;
  }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _noDefinedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _noUndefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _noSharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _noAbsoluteAtoms;
  }

private:
  MemberFile _memberFile;
};

} // end namespace coff
} // end namespace lld
