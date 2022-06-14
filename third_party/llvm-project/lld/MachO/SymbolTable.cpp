//===- SymbolTable.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolTable.h"
#include "ConcatOutputSection.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"

using namespace llvm;
using namespace lld;
using namespace lld::macho;

Symbol *SymbolTable::find(CachedHashStringRef cachedName) {
  auto it = symMap.find(cachedName);
  if (it == symMap.end())
    return nullptr;
  return symVector[it->second];
}

std::pair<Symbol *, bool> SymbolTable::insert(StringRef name,
                                              const InputFile *file) {
  auto p = symMap.insert({CachedHashStringRef(name), (int)symVector.size()});

  Symbol *sym;
  if (!p.second) {
    // Name already present in the symbol table.
    sym = symVector[p.first->second];
  } else {
    // Name is a new symbol.
    sym = reinterpret_cast<Symbol *>(make<SymbolUnion>());
    symVector.push_back(sym);
  }

  sym->isUsedInRegularObj |= !file || isa<ObjFile>(file);
  return {sym, p.second};
}

Defined *SymbolTable::addDefined(StringRef name, InputFile *file,
                                 InputSection *isec, uint64_t value,
                                 uint64_t size, bool isWeakDef,
                                 bool isPrivateExtern, bool isThumb,
                                 bool isReferencedDynamically, bool noDeadStrip,
                                 bool isWeakDefCanBeHidden) {
  Symbol *s;
  bool wasInserted;
  bool overridesWeakDef = false;
  std::tie(s, wasInserted) = insert(name, file);

  assert(!isWeakDef || (isa<BitcodeFile>(file) && !isec) ||
         (isa<ObjFile>(file) && file == isec->getFile()));

  if (!wasInserted) {
    if (auto *defined = dyn_cast<Defined>(s)) {
      if (isWeakDef) {
        // See further comment in createDefined() in InputFiles.cpp
        if (defined->isWeakDef()) {
          defined->privateExtern &= isPrivateExtern;
          defined->weakDefCanBeHidden &= isWeakDefCanBeHidden;
          defined->referencedDynamically |= isReferencedDynamically;
          defined->noDeadStrip |= noDeadStrip;
        }
        // FIXME: Handle this for bitcode files.
        if (auto concatIsec = dyn_cast_or_null<ConcatInputSection>(isec))
          concatIsec->wasCoalesced = true;
        return defined;
      }

      if (defined->isWeakDef()) {
        // FIXME: Handle this for bitcode files.
        if (auto concatIsec =
                dyn_cast_or_null<ConcatInputSection>(defined->isec)) {
          concatIsec->wasCoalesced = true;
          concatIsec->symbols.erase(llvm::find(concatIsec->symbols, defined));
        }
      } else {
        error("duplicate symbol: " + toString(*defined) + "\n>>> defined in " +
              toString(defined->getFile()) + "\n>>> defined in " +
              toString(file));
      }

    } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
      overridesWeakDef = !isWeakDef && dysym->isWeakDef();
      dysym->unreference();
    }
    // Defined symbols take priority over other types of symbols, so in case
    // of a name conflict, we fall through to the replaceSymbol() call below.
  }

  // With -flat_namespace, all extern symbols in dylibs are interposable.
  // FIXME: Add support for `-interposable` (PR53680).
  bool interposable = config->namespaceKind == NamespaceKind::flat &&
                      config->outputType != MachO::MH_EXECUTE &&
                      !isPrivateExtern;
  Defined *defined = replaceSymbol<Defined>(
      s, name, file, isec, value, size, isWeakDef, /*isExternal=*/true,
      isPrivateExtern, /*includeInSymtab=*/true, isThumb,
      isReferencedDynamically, noDeadStrip, overridesWeakDef,
      isWeakDefCanBeHidden, interposable);
  return defined;
}

Symbol *SymbolTable::addUndefined(StringRef name, InputFile *file,
                                  bool isWeakRef) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

  RefState refState = isWeakRef ? RefState::Weak : RefState::Strong;

  if (wasInserted)
    replaceSymbol<Undefined>(s, name, file, refState);
  else if (auto *lazy = dyn_cast<LazyArchive>(s))
    lazy->fetchArchiveMember();
  else if (isa<LazyObject>(s))
    extract(*s->getFile(), s->getName());
  else if (auto *dynsym = dyn_cast<DylibSymbol>(s))
    dynsym->reference(refState);
  else if (auto *undefined = dyn_cast<Undefined>(s))
    undefined->refState = std::max(undefined->refState, refState);
  return s;
}

Symbol *SymbolTable::addCommon(StringRef name, InputFile *file, uint64_t size,
                               uint32_t align, bool isPrivateExtern) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

  if (!wasInserted) {
    if (auto *common = dyn_cast<CommonSymbol>(s)) {
      if (size < common->size)
        return s;
    } else if (isa<Defined>(s)) {
      return s;
    }
    // Common symbols take priority over all non-Defined symbols, so in case of
    // a name conflict, we fall through to the replaceSymbol() call below.
  }

  replaceSymbol<CommonSymbol>(s, name, file, size, align, isPrivateExtern);
  return s;
}

Symbol *SymbolTable::addDylib(StringRef name, DylibFile *file, bool isWeakDef,
                              bool isTlv) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

  RefState refState = RefState::Unreferenced;
  if (!wasInserted) {
    if (auto *defined = dyn_cast<Defined>(s)) {
      if (isWeakDef && !defined->isWeakDef())
        defined->overridesWeakDef = true;
    } else if (auto *undefined = dyn_cast<Undefined>(s)) {
      refState = undefined->refState;
    } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
      refState = dysym->getRefState();
    }
  }

  bool isDynamicLookup = file == nullptr;
  if (wasInserted || isa<Undefined>(s) ||
      (isa<DylibSymbol>(s) &&
       ((!isWeakDef && s->isWeakDef()) ||
        (!isDynamicLookup && cast<DylibSymbol>(s)->isDynamicLookup())))) {
    if (auto *dynsym = dyn_cast<DylibSymbol>(s))
      dynsym->unreference();
    replaceSymbol<DylibSymbol>(s, file, name, isWeakDef, refState, isTlv);
  }

  return s;
}

Symbol *SymbolTable::addDynamicLookup(StringRef name) {
  return addDylib(name, /*file=*/nullptr, /*isWeakDef=*/false, /*isTlv=*/false);
}

Symbol *SymbolTable::addLazyArchive(StringRef name, ArchiveFile *file,
                                    const object::Archive::Symbol &sym) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, file);

  if (wasInserted) {
    replaceSymbol<LazyArchive>(s, file, sym);
  } else if (isa<Undefined>(s)) {
    file->fetch(sym);
  } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
    if (dysym->isWeakDef()) {
      if (dysym->getRefState() != RefState::Unreferenced)
        file->fetch(sym);
      else
        replaceSymbol<LazyArchive>(s, file, sym);
    }
  }
  return s;
}

Symbol *SymbolTable::addLazyObject(StringRef name, InputFile &file) {
  Symbol *s;
  bool wasInserted;
  std::tie(s, wasInserted) = insert(name, &file);

  if (wasInserted) {
    replaceSymbol<LazyObject>(s, file, name);
  } else if (isa<Undefined>(s)) {
    extract(file, name);
  } else if (auto *dysym = dyn_cast<DylibSymbol>(s)) {
    if (dysym->isWeakDef()) {
      if (dysym->getRefState() != RefState::Unreferenced)
        extract(file, name);
      else
        replaceSymbol<LazyObject>(s, file, name);
    }
  }
  return s;
}

Defined *SymbolTable::addSynthetic(StringRef name, InputSection *isec,
                                   uint64_t value, bool isPrivateExtern,
                                   bool includeInSymtab,
                                   bool referencedDynamically) {
  assert(!isec || !isec->getFile()); // See makeSyntheticInputSection().
  Defined *s =
      addDefined(name, /*file=*/nullptr, isec, value, /*size=*/0,
                 /*isWeakDef=*/false, isPrivateExtern, /*isThumb=*/false,
                 referencedDynamically, /*noDeadStrip=*/false,
                 /*isWeakDefCanBeHidden=*/false);
  s->includeInSymtab = includeInSymtab;
  return s;
}

enum class Boundary {
  Start,
  End,
};

static Defined *createBoundarySymbol(const Undefined &sym) {
  return symtab->addSynthetic(
      sym.getName(), /*isec=*/nullptr, /*value=*/-1, /*isPrivateExtern=*/true,
      /*includeInSymtab=*/false, /*referencedDynamically=*/false);
}

static void handleSectionBoundarySymbol(const Undefined &sym, StringRef segSect,
                                        Boundary which) {
  StringRef segName, sectName;
  std::tie(segName, sectName) = segSect.split('$');

  // Attach the symbol to any InputSection that will end up in the right
  // OutputSection -- it doesn't matter which one we pick.
  // Don't bother looking through inputSections for a matching
  // ConcatInputSection -- we need to create ConcatInputSection for
  // non-existing sections anyways, and that codepath works even if we should
  // already have a ConcatInputSection with the right name.

  OutputSection *osec = nullptr;
  // This looks for __TEXT,__cstring etc.
  for (SyntheticSection *ssec : syntheticSections)
    if (ssec->segname == segName && ssec->name == sectName) {
      osec = ssec->isec->parent;
      break;
    }

  if (!osec) {
    ConcatInputSection *isec = makeSyntheticInputSection(segName, sectName);

    // This runs after markLive() and is only called for Undefineds that are
    // live. Marking the isec live ensures an OutputSection is created that the
    // start/end symbol can refer to.
    assert(sym.isLive());
    isec->live = true;

    // This runs after gatherInputSections(), so need to explicitly set parent
    // and add to inputSections.
    osec = isec->parent = ConcatOutputSection::getOrCreateForInput(isec);
    inputSections.push_back(isec);
  }

  if (which == Boundary::Start)
    osec->sectionStartSymbols.push_back(createBoundarySymbol(sym));
  else
    osec->sectionEndSymbols.push_back(createBoundarySymbol(sym));
}

static void handleSegmentBoundarySymbol(const Undefined &sym, StringRef segName,
                                        Boundary which) {
  OutputSegment *seg = getOrCreateOutputSegment(segName);
  if (which == Boundary::Start)
    seg->segmentStartSymbols.push_back(createBoundarySymbol(sym));
  else
    seg->segmentEndSymbols.push_back(createBoundarySymbol(sym));
}

// Try to find a definition for an undefined symbol.
// Returns true if a definition was found and no diagnostics are needed.
static bool recoverFromUndefinedSymbol(const Undefined &sym) {
  // Handle start/end symbols.
  StringRef name = sym.getName();
  if (name.consume_front("section$start$")) {
    handleSectionBoundarySymbol(sym, name, Boundary::Start);
    return true;
  }
  if (name.consume_front("section$end$")) {
    handleSectionBoundarySymbol(sym, name, Boundary::End);
    return true;
  }
  if (name.consume_front("segment$start$")) {
    handleSegmentBoundarySymbol(sym, name, Boundary::Start);
    return true;
  }
  if (name.consume_front("segment$end$")) {
    handleSegmentBoundarySymbol(sym, name, Boundary::End);
    return true;
  }

  // Handle -U.
  if (config->explicitDynamicLookups.count(sym.getName())) {
    symtab->addDynamicLookup(sym.getName());
    return true;
  }

  // Handle -undefined.
  if (config->undefinedSymbolTreatment ==
          UndefinedSymbolTreatment::dynamic_lookup ||
      config->undefinedSymbolTreatment == UndefinedSymbolTreatment::suppress) {
    symtab->addDynamicLookup(sym.getName());
    return true;
  }

  // We do not return true here, as we still need to print diagnostics.
  if (config->undefinedSymbolTreatment == UndefinedSymbolTreatment::warning)
    symtab->addDynamicLookup(sym.getName());

  return false;
}

struct UndefinedDiag {
  struct SectionAndOffset {
    const InputSection *isec;
    uint64_t offset;
  };

  std::vector<SectionAndOffset> codeReferences;
  std::vector<std::string> otherReferences;
};

static MapVector<const Undefined *, UndefinedDiag> undefs;

void macho::reportPendingUndefinedSymbols() {
  for (const auto &undef : undefs) {
    const UndefinedDiag &locations = undef.second;

    std::string message = "undefined symbol";
    if (config->archMultiple)
      message += (" for arch " + getArchitectureName(config->arch())).str();
    message += ": " + toString(*undef.first);

    const size_t maxUndefinedReferences = 3;
    size_t i = 0;
    for (const std::string &loc : locations.otherReferences) {
      if (i >= maxUndefinedReferences)
        break;
      message += "\n>>> referenced by " + loc;
      ++i;
    }

    for (const UndefinedDiag::SectionAndOffset &loc :
         locations.codeReferences) {
      if (i >= maxUndefinedReferences)
        break;
      // TODO: Get source file/line from debug information.
      message += "\n>>> referenced by " + loc.isec->getLocation(loc.offset);
      ++i;
    }

    size_t totalReferences =
        locations.otherReferences.size() + locations.codeReferences.size();
    if (totalReferences > i)
      message +=
          ("\n>>> referenced " + Twine(totalReferences - i) + " more times")
              .str();

    if (config->undefinedSymbolTreatment == UndefinedSymbolTreatment::error)
      error(message);
    else if (config->undefinedSymbolTreatment ==
             UndefinedSymbolTreatment::warning)
      warn(message);
    else
      assert(false &&
             "diagnostics make sense for -undefined error|warning only");
  }

  // This function is called multiple times during execution. Clear the printed
  // diagnostics to avoid printing the same things again the next time.
  undefs.clear();
}

void macho::treatUndefinedSymbol(const Undefined &sym, StringRef source) {
  if (recoverFromUndefinedSymbol(sym))
    return;

  undefs[&sym].otherReferences.push_back(source.str());
}

void macho::treatUndefinedSymbol(const Undefined &sym, const InputSection *isec,
                                 uint64_t offset) {
  if (recoverFromUndefinedSymbol(sym))
    return;

  undefs[&sym].codeReferences.push_back({isec, offset});
}

std::unique_ptr<SymbolTable> macho::symtab;
