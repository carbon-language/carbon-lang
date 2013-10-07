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

// The symbol __ImageBase is a linker generated symbol. No standard library
// files define it, but the linker is expected to prepare it as if it was read
// from a file. The content of the atom is a 4-byte integer equal to the image
// base address. Note that because the name is prefixed by an underscore on x86
// Win32, the actual symbol name will be ___ImageBase (three underscores).
class ImageBaseAtom : public COFFLinkerInternalAtom {
public:
  ImageBaseAtom(const PECOFFLinkingContext &context, const File &file,
                uint32_t imageBase)
      : COFFLinkerInternalAtom(file, assembleRawContent(imageBase)),
        _name(context.decorateSymbol("__ImageBase")) {}

  virtual StringRef name() const { return _name; }
  virtual uint64_t ordinal() const { return 0; }
  virtual ContentType contentType() const { return typeData; }
  virtual ContentPermissions permissions() const { return permRW_; }
  virtual DeadStripKind deadStrip() const { return deadStripAlways; }

private:
  std::vector<uint8_t> assembleRawContent(uint32_t imageBase) {
    std::vector<uint8_t> data = std::vector<uint8_t>(4);
    *(reinterpret_cast<uint32_t *>(&data[0])) = imageBase;
    return data;
  }

  StringRef _name;
};

// The file to wrap ImageBaseAtom. This is the only member file of
// LinkerGeneratedSymbolFile.
class MemberFile : public SimpleFile {
public:
  MemberFile(const PECOFFLinkingContext &context)
      : SimpleFile(context, "Member of the Linker Internal File"),
        _atom(context, *this, context.getBaseAddress()) {
    addAtom(_atom);
  };

  bool contains(StringRef name) const {
    return _atom.name() == name;
  }

private:
  ImageBaseAtom _atom;
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
