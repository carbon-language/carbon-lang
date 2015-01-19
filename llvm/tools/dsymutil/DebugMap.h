//===- tools/dsymutil/DebugMap.h - Generic debug map representation -------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This file contains the class declaration of the DebugMap
/// entity. A DebugMap lists all the object files linked together to
/// produce an executable along with the linked address of all the
/// atoms used in these object files.
/// The DebugMap is an input to the DwarfLinker class that will
/// extract the Dwarf debug information from the referenced object
/// files and link their usefull debug info together.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_DSYMUTIL_DEBUGMAP_H
#define LLVM_TOOLS_DSYMUTIL_DEBUGMAP_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Format.h"
#include <vector>

namespace llvm {
class raw_ostream;

namespace dsymutil {
class DebugMapObject;

/// \brief The DebugMap object stores the list of object files to
/// query for debug information along with the mapping between the
/// symbols' addresses in the object file to their linked address in
/// the linked binary.
///
/// A DebugMap producer could look like this:
/// DebugMap *DM = new DebugMap();
/// for (const auto &Obj: LinkedObjects) {
///     DebugMapObject &DMO = DM->addDebugMapObject(Obj.getPath());
///     for (const auto &Sym: Obj.getLinkedSymbols())
///         DMO.addSymbol(Sym.getName(), Sym.getObjectFileAddress(),
///                       Sym.getBinaryAddress());
/// }
///
/// A DebugMap consumer can then use the map to link the debug
/// information. For example something along the lines of:
/// for (const auto &DMO: DM->objects()) {
///     auto Obj = createBinary(DMO.getObjectFilename());
///     for (auto &DIE: Obj.getDwarfDIEs()) {
///         if (SymbolMapping *Sym = DMO.lookup(DIE.getName()))
///             DIE.relocate(Sym->ObjectAddress, Sym->BinaryAddress);
///         else
///             DIE.discardSubtree();
///     }
/// }
class DebugMap {
  Triple BinaryTriple;
  typedef std::vector<std::unique_ptr<DebugMapObject>> ObjectContainer;
  ObjectContainer Objects;

public:
  DebugMap(const Triple &BinaryTriple) : BinaryTriple(BinaryTriple) {}

  typedef ObjectContainer::const_iterator const_iterator;

  iterator_range<const_iterator> objects() const {
    return make_range(begin(), end());
  }

  const_iterator begin() const { return Objects.begin(); }

  const_iterator end() const { return Objects.end(); }

  /// This function adds an DebugMapObject to the list owned by this
  /// debug map.
  DebugMapObject &addDebugMapObject(StringRef ObjectFilePath);

  const Triple &getTriple() { return BinaryTriple; }

  void print(raw_ostream &OS) const;

#ifndef NDEBUG
  void dump() const;
#endif
};

/// \brief The DebugMapObject represents one object file described by
/// the DebugMap. It contains a list of mappings between addresses in
/// the object file and in the linked binary for all the linked atoms
/// in this object file.
class DebugMapObject {
public:
  struct SymbolMapping {
    uint64_t ObjectAddress;
    uint64_t BinaryAddress;
    SymbolMapping(uint64_t ObjectAddress, uint64_t BinaryAddress)
        : ObjectAddress(ObjectAddress), BinaryAddress(BinaryAddress) {}
  };

  /// \brief Adds a symbol mapping to this DebugMapObject.
  /// \returns false if the symbol was already registered. The request
  /// is discarded in this case.
  bool addSymbol(llvm::StringRef SymName, uint64_t ObjectAddress,
                 uint64_t LinkedAddress);

  /// \brief Lookup a symbol mapping.
  /// \returns null if the symbol isn't found.
  const SymbolMapping *lookupSymbol(StringRef SymbolName) const;

  llvm::StringRef getObjectFilename() const { return Filename; }

  void print(raw_ostream &OS) const;
#ifndef NDEBUG
  void dump() const;
#endif
private:
  friend class DebugMap;
  /// DebugMapObjects can only be constructed by the owning DebugMap.
  DebugMapObject(StringRef ObjectFilename);

  std::string Filename;
  StringMap<SymbolMapping> Symbols;
};
}
}

#endif // LLVM_TOOLS_DSYMUTIL_DEBUGMAP_H
