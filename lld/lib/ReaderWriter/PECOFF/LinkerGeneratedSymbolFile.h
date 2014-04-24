//===- lib/ReaderWriter/PECOFF/LinkerGeneratedSymbolFile.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"

#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Simple.h"
#include "llvm/Support/Allocator.h"

namespace lld {
namespace pecoff {

/// The defined atom for dllexported symbols with __imp_ prefix.
class ImpPointerAtom : public COFFLinkerInternalAtom {
public:
  ImpPointerAtom(const File &file, StringRef symbolName)
      : COFFLinkerInternalAtom(file, /*oridnal*/ 0, std::vector<uint8_t>(4),
                               symbolName) {}

  uint64_t ordinal() const override { return 0; }
  Scope scope() const override { return scopeGlobal; }
  ContentType contentType() const override { return typeData; }
  Alignment alignment() const override { return Alignment(4); }
  ContentPermissions permissions() const override { return permR__; }
};

// A virtual file containing absolute symbol __ImageBase. __ImageBase (or
// ___ImageBase on x86) is a linker-generated symbol whose address is the same
// as the image base address.
class LinkerGeneratedSymbolFile : public SimpleFile {
public:
  LinkerGeneratedSymbolFile(const PECOFFLinkingContext &ctx)
      : SimpleFile("<linker-internal-file>"),
        _imageBaseAtom(*this, ctx.decorateSymbol("__ImageBase"),
                       Atom::scopeGlobal, ctx.getBaseAddress()) {
    addAtom(_imageBaseAtom);

    // Create implciit symbols for exported symbols.
    for (const PECOFFLinkingContext::ExportDesc &exp : ctx.getDllExports()) {
      UndefinedAtom *target = new (_alloc) SimpleUndefinedAtom(*this, exp.name);
      COFFLinkerInternalAtom *imp = createImpPointerAtom(ctx, exp.name);
      imp->addReference(std::unique_ptr<COFFReference>(
          new COFFReference(target, 0, llvm::COFF::IMAGE_REL_I386_DIR32)));
      addAtom(*target);
      addAtom(*imp);
    }
  };

private:
  COFFLinkerInternalAtom *createImpPointerAtom(const PECOFFLinkingContext &ctx,
                                               StringRef name) {
    std::string sym = "_imp_";
    sym.append(name);
    sym = ctx.decorateSymbol(sym);
    return new (_alloc) ImpPointerAtom(*this, ctx.allocate(sym));
  }

  COFFAbsoluteAtom _imageBaseAtom;
  llvm::BumpPtrAllocator _alloc;
};

} // end namespace pecoff
} // end namespace lld
