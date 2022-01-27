//===- LinkerScript.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the parser/evaluator of the linker script.
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TimeProfiler.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::elf;

std::unique_ptr<LinkerScript> elf::script;

static bool isSectionPrefix(StringRef prefix, StringRef name) {
  return name.consume_front(prefix) && (name.empty() || name[0] == '.');
}

static StringRef getOutputSectionName(const InputSectionBase *s) {
  if (config->relocatable)
    return s->name;

  // This is for --emit-relocs. If .text.foo is emitted as .text.bar, we want
  // to emit .rela.text.foo as .rela.text.bar for consistency (this is not
  // technically required, but not doing it is odd). This code guarantees that.
  if (auto *isec = dyn_cast<InputSection>(s)) {
    if (InputSectionBase *rel = isec->getRelocatedSection()) {
      OutputSection *out = rel->getOutputSection();
      if (s->type == SHT_RELA)
        return saver().save(".rela" + out->name);
      return saver().save(".rel" + out->name);
    }
  }

  // A BssSection created for a common symbol is identified as "COMMON" in
  // linker scripts. It should go to .bss section.
  if (s->name == "COMMON")
    return ".bss";

  if (script->hasSectionsCommand)
    return s->name;

  // When no SECTIONS is specified, emulate GNU ld's internal linker scripts
  // by grouping sections with certain prefixes.

  // GNU ld places text sections with prefix ".text.hot.", ".text.unknown.",
  // ".text.unlikely.", ".text.startup." or ".text.exit." before others.
  // We provide an option -z keep-text-section-prefix to group such sections
  // into separate output sections. This is more flexible. See also
  // sortISDBySectionOrder().
  // ".text.unknown" means the hotness of the section is unknown. When
  // SampleFDO is used, if a function doesn't have sample, it could be very
  // cold or it could be a new function never being sampled. Those functions
  // will be kept in the ".text.unknown" section.
  // ".text.split." holds symbols which are split out from functions in other
  // input sections. For example, with -fsplit-machine-functions, placing the
  // cold parts in .text.split instead of .text.unlikely mitigates against poor
  // profile inaccuracy. Techniques such as hugepage remapping can make
  // conservative decisions at the section granularity.
  if (isSectionPrefix(".text", s->name)) {
    if (config->zKeepTextSectionPrefix)
      for (StringRef v : {".text.hot", ".text.unknown", ".text.unlikely",
                          ".text.startup", ".text.exit", ".text.split"})
        if (isSectionPrefix(v.substr(5), s->name.substr(5)))
          return v;
    return ".text";
  }

  for (StringRef v :
       {".data.rel.ro", ".data", ".rodata", ".bss.rel.ro", ".bss",
        ".gcc_except_table", ".init_array", ".fini_array", ".tbss", ".tdata",
        ".ARM.exidx", ".ARM.extab", ".ctors", ".dtors"})
    if (isSectionPrefix(v, s->name))
      return v;

  return s->name;
}

uint64_t ExprValue::getValue() const {
  if (sec)
    return alignTo(sec->getOutputSection()->addr + sec->getOffset(val),
                   alignment);
  return alignTo(val, alignment);
}

uint64_t ExprValue::getSecAddr() const {
  return sec ? sec->getOutputSection()->addr + sec->getOffset(0) : 0;
}

uint64_t ExprValue::getSectionOffset() const {
  // If the alignment is trivial, we don't have to compute the full
  // value to know the offset. This allows this function to succeed in
  // cases where the output section is not yet known.
  if (alignment == 1 && !sec)
    return val;
  return getValue() - getSecAddr();
}

OutputSection *LinkerScript::createOutputSection(StringRef name,
                                                 StringRef location) {
  OutputSection *&secRef = nameToOutputSection[CachedHashStringRef(name)];
  OutputSection *sec;
  if (secRef && secRef->location.empty()) {
    // There was a forward reference.
    sec = secRef;
  } else {
    sec = make<OutputSection>(name, SHT_PROGBITS, 0);
    if (!secRef)
      secRef = sec;
  }
  sec->location = std::string(location);
  return sec;
}

OutputSection *LinkerScript::getOrCreateOutputSection(StringRef name) {
  OutputSection *&cmdRef = nameToOutputSection[CachedHashStringRef(name)];
  if (!cmdRef)
    cmdRef = make<OutputSection>(name, SHT_PROGBITS, 0);
  return cmdRef;
}

// Expands the memory region by the specified size.
static void expandMemoryRegion(MemoryRegion *memRegion, uint64_t size,
                               StringRef secName) {
  memRegion->curPos += size;
  uint64_t newSize = memRegion->curPos - (memRegion->origin)().getValue();
  uint64_t length = (memRegion->length)().getValue();
  if (newSize > length)
    error("section '" + secName + "' will not fit in region '" +
          memRegion->name + "': overflowed by " + Twine(newSize - length) +
          " bytes");
}

void LinkerScript::expandMemoryRegions(uint64_t size) {
  if (ctx->memRegion)
    expandMemoryRegion(ctx->memRegion, size, ctx->outSec->name);
  // Only expand the LMARegion if it is different from memRegion.
  if (ctx->lmaRegion && ctx->memRegion != ctx->lmaRegion)
    expandMemoryRegion(ctx->lmaRegion, size, ctx->outSec->name);
}

void LinkerScript::expandOutputSection(uint64_t size) {
  ctx->outSec->size += size;
  expandMemoryRegions(size);
}

void LinkerScript::setDot(Expr e, const Twine &loc, bool inSec) {
  uint64_t val = e().getValue();
  if (val < dot && inSec)
    error(loc + ": unable to move location counter backward for: " +
          ctx->outSec->name);

  // Update to location counter means update to section size.
  if (inSec)
    expandOutputSection(val - dot);

  dot = val;
}

// Used for handling linker symbol assignments, for both finalizing
// their values and doing early declarations. Returns true if symbol
// should be defined from linker script.
static bool shouldDefineSym(SymbolAssignment *cmd) {
  if (cmd->name == ".")
    return false;

  if (!cmd->provide)
    return true;

  // If a symbol was in PROVIDE(), we need to define it only
  // when it is a referenced undefined symbol.
  Symbol *b = symtab->find(cmd->name);
  if (b && !b->isDefined())
    return true;
  return false;
}

// Called by processSymbolAssignments() to assign definitions to
// linker-script-defined symbols.
void LinkerScript::addSymbol(SymbolAssignment *cmd) {
  if (!shouldDefineSym(cmd))
    return;

  // Define a symbol.
  ExprValue value = cmd->expression();
  SectionBase *sec = value.isAbsolute() ? nullptr : value.sec;
  uint8_t visibility = cmd->hidden ? STV_HIDDEN : STV_DEFAULT;

  // When this function is called, section addresses have not been
  // fixed yet. So, we may or may not know the value of the RHS
  // expression.
  //
  // For example, if an expression is `x = 42`, we know x is always 42.
  // However, if an expression is `x = .`, there's no way to know its
  // value at the moment.
  //
  // We want to set symbol values early if we can. This allows us to
  // use symbols as variables in linker scripts. Doing so allows us to
  // write expressions like this: `alignment = 16; . = ALIGN(., alignment)`.
  uint64_t symValue = value.sec ? 0 : value.getValue();

  Defined newSym(nullptr, cmd->name, STB_GLOBAL, visibility, value.type,
                 symValue, 0, sec);

  Symbol *sym = symtab->insert(cmd->name);
  sym->mergeProperties(newSym);
  sym->replace(newSym);
  cmd->sym = cast<Defined>(sym);
}

// This function is called from LinkerScript::declareSymbols.
// It creates a placeholder symbol if needed.
static void declareSymbol(SymbolAssignment *cmd) {
  if (!shouldDefineSym(cmd))
    return;

  uint8_t visibility = cmd->hidden ? STV_HIDDEN : STV_DEFAULT;
  Defined newSym(nullptr, cmd->name, STB_GLOBAL, visibility, STT_NOTYPE, 0, 0,
                 nullptr);

  // We can't calculate final value right now.
  Symbol *sym = symtab->insert(cmd->name);
  sym->mergeProperties(newSym);
  sym->replace(newSym);

  cmd->sym = cast<Defined>(sym);
  cmd->provide = false;
  sym->scriptDefined = true;
}

using SymbolAssignmentMap =
    DenseMap<const Defined *, std::pair<SectionBase *, uint64_t>>;

// Collect section/value pairs of linker-script-defined symbols. This is used to
// check whether symbol values converge.
static SymbolAssignmentMap
getSymbolAssignmentValues(ArrayRef<SectionCommand *> sectionCommands) {
  SymbolAssignmentMap ret;
  for (SectionCommand *cmd : sectionCommands) {
    if (auto *assign = dyn_cast<SymbolAssignment>(cmd)) {
      if (assign->sym) // sym is nullptr for dot.
        ret.try_emplace(assign->sym, std::make_pair(assign->sym->section,
                                                    assign->sym->value));
      continue;
    }
    for (SectionCommand *subCmd : cast<OutputSection>(cmd)->commands)
      if (auto *assign = dyn_cast<SymbolAssignment>(subCmd))
        if (assign->sym)
          ret.try_emplace(assign->sym, std::make_pair(assign->sym->section,
                                                      assign->sym->value));
  }
  return ret;
}

// Returns the lexicographical smallest (for determinism) Defined whose
// section/value has changed.
static const Defined *
getChangedSymbolAssignment(const SymbolAssignmentMap &oldValues) {
  const Defined *changed = nullptr;
  for (auto &it : oldValues) {
    const Defined *sym = it.first;
    if (std::make_pair(sym->section, sym->value) != it.second &&
        (!changed || sym->getName() < changed->getName()))
      changed = sym;
  }
  return changed;
}

// Process INSERT [AFTER|BEFORE] commands. For each command, we move the
// specified output section to the designated place.
void LinkerScript::processInsertCommands() {
  SmallVector<OutputSection *, 0> moves;
  for (const InsertCommand &cmd : insertCommands) {
    for (StringRef name : cmd.names) {
      // If base is empty, it may have been discarded by
      // adjustSectionsBeforeSorting(). We do not handle such output sections.
      auto from = llvm::find_if(sectionCommands, [&](SectionCommand *subCmd) {
        return isa<OutputSection>(subCmd) &&
               cast<OutputSection>(subCmd)->name == name;
      });
      if (from == sectionCommands.end())
        continue;
      moves.push_back(cast<OutputSection>(*from));
      sectionCommands.erase(from);
    }

    auto insertPos =
        llvm::find_if(sectionCommands, [&cmd](SectionCommand *subCmd) {
          auto *to = dyn_cast<OutputSection>(subCmd);
          return to != nullptr && to->name == cmd.where;
        });
    if (insertPos == sectionCommands.end()) {
      error("unable to insert " + cmd.names[0] +
            (cmd.isAfter ? " after " : " before ") + cmd.where);
    } else {
      if (cmd.isAfter)
        ++insertPos;
      sectionCommands.insert(insertPos, moves.begin(), moves.end());
    }
    moves.clear();
  }
}

// Symbols defined in script should not be inlined by LTO. At the same time
// we don't know their final values until late stages of link. Here we scan
// over symbol assignment commands and create placeholder symbols if needed.
void LinkerScript::declareSymbols() {
  assert(!ctx);
  for (SectionCommand *cmd : sectionCommands) {
    if (auto *assign = dyn_cast<SymbolAssignment>(cmd)) {
      declareSymbol(assign);
      continue;
    }

    // If the output section directive has constraints,
    // we can't say for sure if it is going to be included or not.
    // Skip such sections for now. Improve the checks if we ever
    // need symbols from that sections to be declared early.
    auto *sec = cast<OutputSection>(cmd);
    if (sec->constraint != ConstraintKind::NoConstraint)
      continue;
    for (SectionCommand *cmd : sec->commands)
      if (auto *assign = dyn_cast<SymbolAssignment>(cmd))
        declareSymbol(assign);
  }
}

// This function is called from assignAddresses, while we are
// fixing the output section addresses. This function is supposed
// to set the final value for a given symbol assignment.
void LinkerScript::assignSymbol(SymbolAssignment *cmd, bool inSec) {
  if (cmd->name == ".") {
    setDot(cmd->expression, cmd->location, inSec);
    return;
  }

  if (!cmd->sym)
    return;

  ExprValue v = cmd->expression();
  if (v.isAbsolute()) {
    cmd->sym->section = nullptr;
    cmd->sym->value = v.getValue();
  } else {
    cmd->sym->section = v.sec;
    cmd->sym->value = v.getSectionOffset();
  }
  cmd->sym->type = v.type;
}

static inline StringRef getFilename(const InputFile *file) {
  return file ? file->getNameForScript() : StringRef();
}

bool InputSectionDescription::matchesFile(const InputFile *file) const {
  if (filePat.isTrivialMatchAll())
    return true;

  if (!matchesFileCache || matchesFileCache->first != file)
    matchesFileCache.emplace(file, filePat.match(getFilename(file)));

  return matchesFileCache->second;
}

bool SectionPattern::excludesFile(const InputFile *file) const {
  if (excludedFilePat.empty())
    return false;

  if (!excludesFileCache || excludesFileCache->first != file)
    excludesFileCache.emplace(file, excludedFilePat.match(getFilename(file)));

  return excludesFileCache->second;
}

bool LinkerScript::shouldKeep(InputSectionBase *s) {
  for (InputSectionDescription *id : keptSections)
    if (id->matchesFile(s->file))
      for (SectionPattern &p : id->sectionPatterns)
        if (p.sectionPat.match(s->name) &&
            (s->flags & id->withFlags) == id->withFlags &&
            (s->flags & id->withoutFlags) == 0)
          return true;
  return false;
}

// A helper function for the SORT() command.
static bool matchConstraints(ArrayRef<InputSectionBase *> sections,
                             ConstraintKind kind) {
  if (kind == ConstraintKind::NoConstraint)
    return true;

  bool isRW = llvm::any_of(
      sections, [](InputSectionBase *sec) { return sec->flags & SHF_WRITE; });

  return (isRW && kind == ConstraintKind::ReadWrite) ||
         (!isRW && kind == ConstraintKind::ReadOnly);
}

static void sortSections(MutableArrayRef<InputSectionBase *> vec,
                         SortSectionPolicy k) {
  auto alignmentComparator = [](InputSectionBase *a, InputSectionBase *b) {
    // ">" is not a mistake. Sections with larger alignments are placed
    // before sections with smaller alignments in order to reduce the
    // amount of padding necessary. This is compatible with GNU.
    return a->alignment > b->alignment;
  };
  auto nameComparator = [](InputSectionBase *a, InputSectionBase *b) {
    return a->name < b->name;
  };
  auto priorityComparator = [](InputSectionBase *a, InputSectionBase *b) {
    return getPriority(a->name) < getPriority(b->name);
  };

  switch (k) {
  case SortSectionPolicy::Default:
  case SortSectionPolicy::None:
    return;
  case SortSectionPolicy::Alignment:
    return llvm::stable_sort(vec, alignmentComparator);
  case SortSectionPolicy::Name:
    return llvm::stable_sort(vec, nameComparator);
  case SortSectionPolicy::Priority:
    return llvm::stable_sort(vec, priorityComparator);
  }
}

// Sort sections as instructed by SORT-family commands and --sort-section
// option. Because SORT-family commands can be nested at most two depth
// (e.g. SORT_BY_NAME(SORT_BY_ALIGNMENT(.text.*))) and because the command
// line option is respected even if a SORT command is given, the exact
// behavior we have here is a bit complicated. Here are the rules.
//
// 1. If two SORT commands are given, --sort-section is ignored.
// 2. If one SORT command is given, and if it is not SORT_NONE,
//    --sort-section is handled as an inner SORT command.
// 3. If one SORT command is given, and if it is SORT_NONE, don't sort.
// 4. If no SORT command is given, sort according to --sort-section.
static void sortInputSections(MutableArrayRef<InputSectionBase *> vec,
                              SortSectionPolicy outer,
                              SortSectionPolicy inner) {
  if (outer == SortSectionPolicy::None)
    return;

  if (inner == SortSectionPolicy::Default)
    sortSections(vec, config->sortSection);
  else
    sortSections(vec, inner);
  sortSections(vec, outer);
}

// Compute and remember which sections the InputSectionDescription matches.
SmallVector<InputSectionBase *, 0>
LinkerScript::computeInputSections(const InputSectionDescription *cmd,
                                   ArrayRef<InputSectionBase *> sections) {
  SmallVector<InputSectionBase *, 0> ret;
  SmallVector<size_t, 0> indexes;
  DenseSet<size_t> seen;
  auto sortByPositionThenCommandLine = [&](size_t begin, size_t end) {
    llvm::sort(MutableArrayRef<size_t>(indexes).slice(begin, end - begin));
    for (size_t i = begin; i != end; ++i)
      ret[i] = sections[indexes[i]];
    sortInputSections(
        MutableArrayRef<InputSectionBase *>(ret).slice(begin, end - begin),
        config->sortSection, SortSectionPolicy::None);
  };

  // Collects all sections that satisfy constraints of Cmd.
  size_t sizeAfterPrevSort = 0;
  for (const SectionPattern &pat : cmd->sectionPatterns) {
    size_t sizeBeforeCurrPat = ret.size();

    for (size_t i = 0, e = sections.size(); i != e; ++i) {
      // Skip if the section is dead or has been matched by a previous input
      // section description or a previous pattern.
      InputSectionBase *sec = sections[i];
      if (!sec->isLive() || sec->parent || seen.contains(i))
        continue;

      // For --emit-relocs we have to ignore entries like
      //   .rela.dyn : { *(.rela.data) }
      // which are common because they are in the default bfd script.
      // We do not ignore SHT_REL[A] linker-synthesized sections here because
      // want to support scripts that do custom layout for them.
      if (isa<InputSection>(sec) &&
          cast<InputSection>(sec)->getRelocatedSection())
        continue;

      // Check the name early to improve performance in the common case.
      if (!pat.sectionPat.match(sec->name))
        continue;

      if (!cmd->matchesFile(sec->file) || pat.excludesFile(sec->file) ||
          (sec->flags & cmd->withFlags) != cmd->withFlags ||
          (sec->flags & cmd->withoutFlags) != 0)
        continue;

      ret.push_back(sec);
      indexes.push_back(i);
      seen.insert(i);
    }

    if (pat.sortOuter == SortSectionPolicy::Default)
      continue;

    // Matched sections are ordered by radix sort with the keys being (SORT*,
    // --sort-section, input order), where SORT* (if present) is most
    // significant.
    //
    // Matched sections between the previous SORT* and this SORT* are sorted by
    // (--sort-alignment, input order).
    sortByPositionThenCommandLine(sizeAfterPrevSort, sizeBeforeCurrPat);
    // Matched sections by this SORT* pattern are sorted using all 3 keys.
    // ret[sizeBeforeCurrPat,ret.size()) are already in the input order, so we
    // just sort by sortOuter and sortInner.
    sortInputSections(
        MutableArrayRef<InputSectionBase *>(ret).slice(sizeBeforeCurrPat),
        pat.sortOuter, pat.sortInner);
    sizeAfterPrevSort = ret.size();
  }
  // Matched sections after the last SORT* are sorted by (--sort-alignment,
  // input order).
  sortByPositionThenCommandLine(sizeAfterPrevSort, ret.size());
  return ret;
}

void LinkerScript::discard(InputSectionBase &s) {
  if (&s == in.shStrTab.get())
    error("discarding " + s.name + " section is not allowed");

  s.markDead();
  s.parent = nullptr;
  for (InputSection *sec : s.dependentSections)
    discard(*sec);
}

void LinkerScript::discardSynthetic(OutputSection &outCmd) {
  for (Partition &part : partitions) {
    if (!part.armExidx || !part.armExidx->isLive())
      continue;
    SmallVector<InputSectionBase *, 0> secs(
        part.armExidx->exidxSections.begin(),
        part.armExidx->exidxSections.end());
    for (SectionCommand *cmd : outCmd.commands)
      if (auto *isd = dyn_cast<InputSectionDescription>(cmd))
        for (InputSectionBase *s : computeInputSections(isd, secs))
          discard(*s);
  }
}

SmallVector<InputSectionBase *, 0>
LinkerScript::createInputSectionList(OutputSection &outCmd) {
  SmallVector<InputSectionBase *, 0> ret;

  for (SectionCommand *cmd : outCmd.commands) {
    if (auto *isd = dyn_cast<InputSectionDescription>(cmd)) {
      isd->sectionBases = computeInputSections(isd, inputSections);
      for (InputSectionBase *s : isd->sectionBases)
        s->parent = &outCmd;
      ret.insert(ret.end(), isd->sectionBases.begin(), isd->sectionBases.end());
    }
  }
  return ret;
}

// Create output sections described by SECTIONS commands.
void LinkerScript::processSectionCommands() {
  auto process = [this](OutputSection *osec) {
    SmallVector<InputSectionBase *, 0> v = createInputSectionList(*osec);

    // The output section name `/DISCARD/' is special.
    // Any input section assigned to it is discarded.
    if (osec->name == "/DISCARD/") {
      for (InputSectionBase *s : v)
        discard(*s);
      discardSynthetic(*osec);
      osec->commands.clear();
      return false;
    }

    // This is for ONLY_IF_RO and ONLY_IF_RW. An output section directive
    // ".foo : ONLY_IF_R[OW] { ... }" is handled only if all member input
    // sections satisfy a given constraint. If not, a directive is handled
    // as if it wasn't present from the beginning.
    //
    // Because we'll iterate over SectionCommands many more times, the easy
    // way to "make it as if it wasn't present" is to make it empty.
    if (!matchConstraints(v, osec->constraint)) {
      for (InputSectionBase *s : v)
        s->parent = nullptr;
      osec->commands.clear();
      return false;
    }

    // Handle subalign (e.g. ".foo : SUBALIGN(32) { ... }"). If subalign
    // is given, input sections are aligned to that value, whether the
    // given value is larger or smaller than the original section alignment.
    if (osec->subalignExpr) {
      uint32_t subalign = osec->subalignExpr().getValue();
      for (InputSectionBase *s : v)
        s->alignment = subalign;
    }

    // Set the partition field the same way OutputSection::recordSection()
    // does. Partitions cannot be used with the SECTIONS command, so this is
    // always 1.
    osec->partition = 1;
    return true;
  };

  // Process OVERWRITE_SECTIONS first so that it can overwrite the main script
  // or orphans.
  DenseMap<CachedHashStringRef, OutputSection *> map;
  size_t i = 0;
  for (OutputSection *osec : overwriteSections)
    if (process(osec) &&
        !map.try_emplace(CachedHashStringRef(osec->name), osec).second)
      warn("OVERWRITE_SECTIONS specifies duplicate " + osec->name);
  for (SectionCommand *&base : sectionCommands)
    if (auto *osec = dyn_cast<OutputSection>(base)) {
      if (OutputSection *overwrite =
              map.lookup(CachedHashStringRef(osec->name))) {
        log(overwrite->location + " overwrites " + osec->name);
        overwrite->sectionIndex = i++;
        base = overwrite;
      } else if (process(osec)) {
        osec->sectionIndex = i++;
      }
    }

  // If an OVERWRITE_SECTIONS specified output section is not in
  // sectionCommands, append it to the end. The section will be inserted by
  // orphan placement.
  for (OutputSection *osec : overwriteSections)
    if (osec->partition == 1 && osec->sectionIndex == UINT32_MAX)
      sectionCommands.push_back(osec);
}

void LinkerScript::processSymbolAssignments() {
  // Dot outside an output section still represents a relative address, whose
  // sh_shndx should not be SHN_UNDEF or SHN_ABS. Create a dummy aether section
  // that fills the void outside a section. It has an index of one, which is
  // indistinguishable from any other regular section index.
  aether = make<OutputSection>("", 0, SHF_ALLOC);
  aether->sectionIndex = 1;

  // ctx captures the local AddressState and makes it accessible deliberately.
  // This is needed as there are some cases where we cannot just thread the
  // current state through to a lambda function created by the script parser.
  AddressState state;
  ctx = &state;
  ctx->outSec = aether;

  for (SectionCommand *cmd : sectionCommands) {
    if (auto *assign = dyn_cast<SymbolAssignment>(cmd))
      addSymbol(assign);
    else
      for (SectionCommand *subCmd : cast<OutputSection>(cmd)->commands)
        if (auto *assign = dyn_cast<SymbolAssignment>(subCmd))
          addSymbol(assign);
  }

  ctx = nullptr;
}

static OutputSection *findByName(ArrayRef<SectionCommand *> vec,
                                 StringRef name) {
  for (SectionCommand *cmd : vec)
    if (auto *sec = dyn_cast<OutputSection>(cmd))
      if (sec->name == name)
        return sec;
  return nullptr;
}

static OutputSection *createSection(InputSectionBase *isec,
                                    StringRef outsecName) {
  OutputSection *sec = script->createOutputSection(outsecName, "<internal>");
  sec->recordSection(isec);
  return sec;
}

static OutputSection *
addInputSec(StringMap<TinyPtrVector<OutputSection *>> &map,
            InputSectionBase *isec, StringRef outsecName) {
  // Sections with SHT_GROUP or SHF_GROUP attributes reach here only when the -r
  // option is given. A section with SHT_GROUP defines a "section group", and
  // its members have SHF_GROUP attribute. Usually these flags have already been
  // stripped by InputFiles.cpp as section groups are processed and uniquified.
  // However, for the -r option, we want to pass through all section groups
  // as-is because adding/removing members or merging them with other groups
  // change their semantics.
  if (isec->type == SHT_GROUP || (isec->flags & SHF_GROUP))
    return createSection(isec, outsecName);

  // Imagine .zed : { *(.foo) *(.bar) } script. Both foo and bar may have
  // relocation sections .rela.foo and .rela.bar for example. Most tools do
  // not allow multiple REL[A] sections for output section. Hence we
  // should combine these relocation sections into single output.
  // We skip synthetic sections because it can be .rela.dyn/.rela.plt or any
  // other REL[A] sections created by linker itself.
  if (!isa<SyntheticSection>(isec) &&
      (isec->type == SHT_REL || isec->type == SHT_RELA)) {
    auto *sec = cast<InputSection>(isec);
    OutputSection *out = sec->getRelocatedSection()->getOutputSection();

    if (out->relocationSection) {
      out->relocationSection->recordSection(sec);
      return nullptr;
    }

    out->relocationSection = createSection(isec, outsecName);
    return out->relocationSection;
  }

  //  The ELF spec just says
  // ----------------------------------------------------------------
  // In the first phase, input sections that match in name, type and
  // attribute flags should be concatenated into single sections.
  // ----------------------------------------------------------------
  //
  // However, it is clear that at least some flags have to be ignored for
  // section merging. At the very least SHF_GROUP and SHF_COMPRESSED have to be
  // ignored. We should not have two output .text sections just because one was
  // in a group and another was not for example.
  //
  // It also seems that wording was a late addition and didn't get the
  // necessary scrutiny.
  //
  // Merging sections with different flags is expected by some users. One
  // reason is that if one file has
  //
  // int *const bar __attribute__((section(".foo"))) = (int *)0;
  //
  // gcc with -fPIC will produce a read only .foo section. But if another
  // file has
  //
  // int zed;
  // int *const bar __attribute__((section(".foo"))) = (int *)&zed;
  //
  // gcc with -fPIC will produce a read write section.
  //
  // Last but not least, when using linker script the merge rules are forced by
  // the script. Unfortunately, linker scripts are name based. This means that
  // expressions like *(.foo*) can refer to multiple input sections with
  // different flags. We cannot put them in different output sections or we
  // would produce wrong results for
  //
  // start = .; *(.foo.*) end = .; *(.bar)
  //
  // and a mapping of .foo1 and .bar1 to one section and .foo2 and .bar2 to
  // another. The problem is that there is no way to layout those output
  // sections such that the .foo sections are the only thing between the start
  // and end symbols.
  //
  // Given the above issues, we instead merge sections by name and error on
  // incompatible types and flags.
  TinyPtrVector<OutputSection *> &v = map[outsecName];
  for (OutputSection *sec : v) {
    if (sec->partition != isec->partition)
      continue;

    if (config->relocatable && (isec->flags & SHF_LINK_ORDER)) {
      // Merging two SHF_LINK_ORDER sections with different sh_link fields will
      // change their semantics, so we only merge them in -r links if they will
      // end up being linked to the same output section. The casts are fine
      // because everything in the map was created by the orphan placement code.
      auto *firstIsec = cast<InputSectionBase>(
          cast<InputSectionDescription>(sec->commands[0])->sectionBases[0]);
      OutputSection *firstIsecOut =
          firstIsec->flags & SHF_LINK_ORDER
              ? firstIsec->getLinkOrderDep()->getOutputSection()
              : nullptr;
      if (firstIsecOut != isec->getLinkOrderDep()->getOutputSection())
        continue;
    }

    sec->recordSection(isec);
    return nullptr;
  }

  OutputSection *sec = createSection(isec, outsecName);
  v.push_back(sec);
  return sec;
}

// Add sections that didn't match any sections command.
void LinkerScript::addOrphanSections() {
  StringMap<TinyPtrVector<OutputSection *>> map;
  SmallVector<OutputSection *, 0> v;

  auto add = [&](InputSectionBase *s) {
    if (s->isLive() && !s->parent) {
      orphanSections.push_back(s);

      StringRef name = getOutputSectionName(s);
      if (config->unique) {
        v.push_back(createSection(s, name));
      } else if (OutputSection *sec = findByName(sectionCommands, name)) {
        sec->recordSection(s);
      } else {
        if (OutputSection *os = addInputSec(map, s, name))
          v.push_back(os);
        assert(isa<MergeInputSection>(s) ||
               s->getOutputSection()->sectionIndex == UINT32_MAX);
      }
    }
  };

  // For further --emit-reloc handling code we need target output section
  // to be created before we create relocation output section, so we want
  // to create target sections first. We do not want priority handling
  // for synthetic sections because them are special.
  for (InputSectionBase *isec : inputSections) {
    // In -r links, SHF_LINK_ORDER sections are added while adding their parent
    // sections because we need to know the parent's output section before we
    // can select an output section for the SHF_LINK_ORDER section.
    if (config->relocatable && (isec->flags & SHF_LINK_ORDER))
      continue;

    if (auto *sec = dyn_cast<InputSection>(isec))
      if (InputSectionBase *rel = sec->getRelocatedSection())
        if (auto *relIS = dyn_cast_or_null<InputSectionBase>(rel->parent))
          add(relIS);
    add(isec);
    if (config->relocatable)
      for (InputSectionBase *depSec : isec->dependentSections)
        if (depSec->flags & SHF_LINK_ORDER)
          add(depSec);
  }

  // If no SECTIONS command was given, we should insert sections commands
  // before others, so that we can handle scripts which refers them,
  // for example: "foo = ABSOLUTE(ADDR(.text)));".
  // When SECTIONS command is present we just add all orphans to the end.
  if (hasSectionsCommand)
    sectionCommands.insert(sectionCommands.end(), v.begin(), v.end());
  else
    sectionCommands.insert(sectionCommands.begin(), v.begin(), v.end());
}

void LinkerScript::diagnoseOrphanHandling() const {
  llvm::TimeTraceScope timeScope("Diagnose orphan sections");
  if (config->orphanHandling == OrphanHandlingPolicy::Place)
    return;
  for (const InputSectionBase *sec : orphanSections) {
    // Input SHT_REL[A] retained by --emit-relocs are ignored by
    // computeInputSections(). Don't warn/error.
    if (isa<InputSection>(sec) &&
        cast<InputSection>(sec)->getRelocatedSection())
      continue;

    StringRef name = getOutputSectionName(sec);
    if (config->orphanHandling == OrphanHandlingPolicy::Error)
      error(toString(sec) + " is being placed in '" + name + "'");
    else
      warn(toString(sec) + " is being placed in '" + name + "'");
  }
}

// This function searches for a memory region to place the given output
// section in. If found, a pointer to the appropriate memory region is
// returned in the first member of the pair. Otherwise, a nullptr is returned.
// The second member of the pair is a hint that should be passed to the
// subsequent call of this method.
std::pair<MemoryRegion *, MemoryRegion *>
LinkerScript::findMemoryRegion(OutputSection *sec, MemoryRegion *hint) {
  // Non-allocatable sections are not part of the process image.
  if (!(sec->flags & SHF_ALLOC)) {
    if (!sec->memoryRegionName.empty())
      warn("ignoring memory region assignment for non-allocatable section '" +
           sec->name + "'");
    return {nullptr, nullptr};
  }

  // If a memory region name was specified in the output section command,
  // then try to find that region first.
  if (!sec->memoryRegionName.empty()) {
    if (MemoryRegion *m = memoryRegions.lookup(sec->memoryRegionName))
      return {m, m};
    error("memory region '" + sec->memoryRegionName + "' not declared");
    return {nullptr, nullptr};
  }

  // If at least one memory region is defined, all sections must
  // belong to some memory region. Otherwise, we don't need to do
  // anything for memory regions.
  if (memoryRegions.empty())
    return {nullptr, nullptr};

  // An orphan section should continue the previous memory region.
  if (sec->sectionIndex == UINT32_MAX && hint)
    return {hint, hint};

  // See if a region can be found by matching section flags.
  for (auto &pair : memoryRegions) {
    MemoryRegion *m = pair.second;
    if (m->compatibleWith(sec->flags))
      return {m, nullptr};
  }

  // Otherwise, no suitable region was found.
  error("no memory region specified for section '" + sec->name + "'");
  return {nullptr, nullptr};
}

static OutputSection *findFirstSection(PhdrEntry *load) {
  for (OutputSection *sec : outputSections)
    if (sec->ptLoad == load)
      return sec;
  return nullptr;
}

// This function assigns offsets to input sections and an output section
// for a single sections command (e.g. ".text { *(.text); }").
void LinkerScript::assignOffsets(OutputSection *sec) {
  const bool isTbss = (sec->flags & SHF_TLS) && sec->type == SHT_NOBITS;
  const bool sameMemRegion = ctx->memRegion == sec->memRegion;
  const bool prevLMARegionIsDefault = ctx->lmaRegion == nullptr;
  const uint64_t savedDot = dot;
  ctx->memRegion = sec->memRegion;
  ctx->lmaRegion = sec->lmaRegion;

  if (!(sec->flags & SHF_ALLOC)) {
    // Non-SHF_ALLOC sections have zero addresses.
    dot = 0;
  } else if (isTbss) {
    // Allow consecutive SHF_TLS SHT_NOBITS output sections. The address range
    // starts from the end address of the previous tbss section.
    if (ctx->tbssAddr == 0)
      ctx->tbssAddr = dot;
    else
      dot = ctx->tbssAddr;
  } else {
    if (ctx->memRegion)
      dot = ctx->memRegion->curPos;
    if (sec->addrExpr)
      setDot(sec->addrExpr, sec->location, false);

    // If the address of the section has been moved forward by an explicit
    // expression so that it now starts past the current curPos of the enclosing
    // region, we need to expand the current region to account for the space
    // between the previous section, if any, and the start of this section.
    if (ctx->memRegion && ctx->memRegion->curPos < dot)
      expandMemoryRegion(ctx->memRegion, dot - ctx->memRegion->curPos,
                         sec->name);
  }

  ctx->outSec = sec;
  if (sec->addrExpr && script->hasSectionsCommand) {
    // The alignment is ignored.
    sec->addr = dot;
  } else {
    // sec->alignment is the max of ALIGN and the maximum of input
    // section alignments.
    const uint64_t pos = dot;
    dot = alignTo(dot, sec->alignment);
    sec->addr = dot;
    expandMemoryRegions(dot - pos);
  }

  // ctx->lmaOffset is LMA minus VMA. If LMA is explicitly specified via AT() or
  // AT>, recompute ctx->lmaOffset; otherwise, if both previous/current LMA
  // region is the default, and the two sections are in the same memory region,
  // reuse previous lmaOffset; otherwise, reset lmaOffset to 0. This emulates
  // heuristics described in
  // https://sourceware.org/binutils/docs/ld/Output-Section-LMA.html
  if (sec->lmaExpr) {
    ctx->lmaOffset = sec->lmaExpr().getValue() - dot;
  } else if (MemoryRegion *mr = sec->lmaRegion) {
    uint64_t lmaStart = alignTo(mr->curPos, sec->alignment);
    if (mr->curPos < lmaStart)
      expandMemoryRegion(mr, lmaStart - mr->curPos, sec->name);
    ctx->lmaOffset = lmaStart - dot;
  } else if (!sameMemRegion || !prevLMARegionIsDefault) {
    ctx->lmaOffset = 0;
  }

  // Propagate ctx->lmaOffset to the first "non-header" section.
  if (PhdrEntry *l = sec->ptLoad)
    if (sec == findFirstSection(l))
      l->lmaOffset = ctx->lmaOffset;

  // We can call this method multiple times during the creation of
  // thunks and want to start over calculation each time.
  sec->size = 0;

  // We visited SectionsCommands from processSectionCommands to
  // layout sections. Now, we visit SectionsCommands again to fix
  // section offsets.
  for (SectionCommand *cmd : sec->commands) {
    // This handles the assignments to symbol or to the dot.
    if (auto *assign = dyn_cast<SymbolAssignment>(cmd)) {
      assign->addr = dot;
      assignSymbol(assign, true);
      assign->size = dot - assign->addr;
      continue;
    }

    // Handle BYTE(), SHORT(), LONG(), or QUAD().
    if (auto *data = dyn_cast<ByteCommand>(cmd)) {
      data->offset = dot - sec->addr;
      dot += data->size;
      expandOutputSection(data->size);
      continue;
    }

    // Handle a single input section description command.
    // It calculates and assigns the offsets for each section and also
    // updates the output section size.
    for (InputSection *isec : cast<InputSectionDescription>(cmd)->sections) {
      assert(isec->getParent() == sec);
      const uint64_t pos = dot;
      dot = alignTo(dot, isec->alignment);
      isec->outSecOff = dot - sec->addr;
      dot += isec->getSize();

      // Update output section size after adding each section. This is so that
      // SIZEOF works correctly in the case below:
      // .foo { *(.aaa) a = SIZEOF(.foo); *(.bbb) }
      expandOutputSection(dot - pos);
    }
  }

  // Non-SHF_ALLOC sections do not affect the addresses of other OutputSections
  // as they are not part of the process image.
  if (!(sec->flags & SHF_ALLOC)) {
    dot = savedDot;
  } else if (isTbss) {
    // NOBITS TLS sections are similar. Additionally save the end address.
    ctx->tbssAddr = dot;
    dot = savedDot;
  }
}

static bool isDiscardable(const OutputSection &sec) {
  if (sec.name == "/DISCARD/")
    return true;

  // We do not want to remove OutputSections with expressions that reference
  // symbols even if the OutputSection is empty. We want to ensure that the
  // expressions can be evaluated and report an error if they cannot.
  if (sec.expressionsUseSymbols)
    return false;

  // OutputSections may be referenced by name in ADDR and LOADADDR expressions,
  // as an empty Section can has a valid VMA and LMA we keep the OutputSection
  // to maintain the integrity of the other Expression.
  if (sec.usedInExpression)
    return false;

  for (SectionCommand *cmd : sec.commands) {
    if (auto assign = dyn_cast<SymbolAssignment>(cmd))
      // Don't create empty output sections just for unreferenced PROVIDE
      // symbols.
      if (assign->name != "." && !assign->sym)
        continue;

    if (!isa<InputSectionDescription>(*cmd))
      return false;
  }
  return true;
}

bool LinkerScript::isDiscarded(const OutputSection *sec) const {
  return hasSectionsCommand && (getFirstInputSection(sec) == nullptr) &&
         isDiscardable(*sec);
}

static void maybePropagatePhdrs(OutputSection &sec,
                                SmallVector<StringRef, 0> &phdrs) {
  if (sec.phdrs.empty()) {
    // To match the bfd linker script behaviour, only propagate program
    // headers to sections that are allocated.
    if (sec.flags & SHF_ALLOC)
      sec.phdrs = phdrs;
  } else {
    phdrs = sec.phdrs;
  }
}

void LinkerScript::adjustSectionsBeforeSorting() {
  // If the output section contains only symbol assignments, create a
  // corresponding output section. The issue is what to do with linker script
  // like ".foo : { symbol = 42; }". One option would be to convert it to
  // "symbol = 42;". That is, move the symbol out of the empty section
  // description. That seems to be what bfd does for this simple case. The
  // problem is that this is not completely general. bfd will give up and
  // create a dummy section too if there is a ". = . + 1" inside the section
  // for example.
  // Given that we want to create the section, we have to worry what impact
  // it will have on the link. For example, if we just create a section with
  // 0 for flags, it would change which PT_LOADs are created.
  // We could remember that particular section is dummy and ignore it in
  // other parts of the linker, but unfortunately there are quite a few places
  // that would need to change:
  //   * The program header creation.
  //   * The orphan section placement.
  //   * The address assignment.
  // The other option is to pick flags that minimize the impact the section
  // will have on the rest of the linker. That is why we copy the flags from
  // the previous sections. Only a few flags are needed to keep the impact low.
  uint64_t flags = SHF_ALLOC;

  SmallVector<StringRef, 0> defPhdrs;
  for (SectionCommand *&cmd : sectionCommands) {
    auto *sec = dyn_cast<OutputSection>(cmd);
    if (!sec)
      continue;

    // Handle align (e.g. ".foo : ALIGN(16) { ... }").
    if (sec->alignExpr)
      sec->alignment =
          std::max<uint32_t>(sec->alignment, sec->alignExpr().getValue());

    // The input section might have been removed (if it was an empty synthetic
    // section), but we at least know the flags.
    if (sec->hasInputSections)
      flags = sec->flags;

    // We do not want to keep any special flags for output section
    // in case it is empty.
    bool isEmpty = (getFirstInputSection(sec) == nullptr);
    if (isEmpty)
      sec->flags = flags & ((sec->nonAlloc ? 0 : (uint64_t)SHF_ALLOC) |
                            SHF_WRITE | SHF_EXECINSTR);

    // The code below may remove empty output sections. We should save the
    // specified program headers (if exist) and propagate them to subsequent
    // sections which do not specify program headers.
    // An example of such a linker script is:
    // SECTIONS { .empty : { *(.empty) } :rw
    //            .foo : { *(.foo) } }
    // Note: at this point the order of output sections has not been finalized,
    // because orphans have not been inserted into their expected positions. We
    // will handle them in adjustSectionsAfterSorting().
    if (sec->sectionIndex != UINT32_MAX)
      maybePropagatePhdrs(*sec, defPhdrs);

    if (isEmpty && isDiscardable(*sec)) {
      sec->markDead();
      cmd = nullptr;
    }
  }

  // It is common practice to use very generic linker scripts. So for any
  // given run some of the output sections in the script will be empty.
  // We could create corresponding empty output sections, but that would
  // clutter the output.
  // We instead remove trivially empty sections. The bfd linker seems even
  // more aggressive at removing them.
  llvm::erase_if(sectionCommands, [&](SectionCommand *cmd) { return !cmd; });
}

void LinkerScript::adjustSectionsAfterSorting() {
  // Try and find an appropriate memory region to assign offsets in.
  MemoryRegion *hint = nullptr;
  for (SectionCommand *cmd : sectionCommands) {
    if (auto *sec = dyn_cast<OutputSection>(cmd)) {
      if (!sec->lmaRegionName.empty()) {
        if (MemoryRegion *m = memoryRegions.lookup(sec->lmaRegionName))
          sec->lmaRegion = m;
        else
          error("memory region '" + sec->lmaRegionName + "' not declared");
      }
      std::tie(sec->memRegion, hint) = findMemoryRegion(sec, hint);
    }
  }

  // If output section command doesn't specify any segments,
  // and we haven't previously assigned any section to segment,
  // then we simply assign section to the very first load segment.
  // Below is an example of such linker script:
  // PHDRS { seg PT_LOAD; }
  // SECTIONS { .aaa : { *(.aaa) } }
  SmallVector<StringRef, 0> defPhdrs;
  auto firstPtLoad = llvm::find_if(phdrsCommands, [](const PhdrsCommand &cmd) {
    return cmd.type == PT_LOAD;
  });
  if (firstPtLoad != phdrsCommands.end())
    defPhdrs.push_back(firstPtLoad->name);

  // Walk the commands and propagate the program headers to commands that don't
  // explicitly specify them.
  for (SectionCommand *cmd : sectionCommands)
    if (auto *sec = dyn_cast<OutputSection>(cmd))
      maybePropagatePhdrs(*sec, defPhdrs);
}

static uint64_t computeBase(uint64_t min, bool allocateHeaders) {
  // If there is no SECTIONS or if the linkerscript is explicit about program
  // headers, do our best to allocate them.
  if (!script->hasSectionsCommand || allocateHeaders)
    return 0;
  // Otherwise only allocate program headers if that would not add a page.
  return alignDown(min, config->maxPageSize);
}

// When the SECTIONS command is used, try to find an address for the file and
// program headers output sections, which can be added to the first PT_LOAD
// segment when program headers are created.
//
// We check if the headers fit below the first allocated section. If there isn't
// enough space for these sections, we'll remove them from the PT_LOAD segment,
// and we'll also remove the PT_PHDR segment.
void LinkerScript::allocateHeaders(SmallVector<PhdrEntry *, 0> &phdrs) {
  uint64_t min = std::numeric_limits<uint64_t>::max();
  for (OutputSection *sec : outputSections)
    if (sec->flags & SHF_ALLOC)
      min = std::min<uint64_t>(min, sec->addr);

  auto it = llvm::find_if(
      phdrs, [](const PhdrEntry *e) { return e->p_type == PT_LOAD; });
  if (it == phdrs.end())
    return;
  PhdrEntry *firstPTLoad = *it;

  bool hasExplicitHeaders =
      llvm::any_of(phdrsCommands, [](const PhdrsCommand &cmd) {
        return cmd.hasPhdrs || cmd.hasFilehdr;
      });
  bool paged = !config->omagic && !config->nmagic;
  uint64_t headerSize = getHeaderSize();
  if ((paged || hasExplicitHeaders) &&
      headerSize <= min - computeBase(min, hasExplicitHeaders)) {
    min = alignDown(min - headerSize, config->maxPageSize);
    Out::elfHeader->addr = min;
    Out::programHeaders->addr = min + Out::elfHeader->size;
    return;
  }

  // Error if we were explicitly asked to allocate headers.
  if (hasExplicitHeaders)
    error("could not allocate headers");

  Out::elfHeader->ptLoad = nullptr;
  Out::programHeaders->ptLoad = nullptr;
  firstPTLoad->firstSec = findFirstSection(firstPTLoad);

  llvm::erase_if(phdrs,
                 [](const PhdrEntry *e) { return e->p_type == PT_PHDR; });
}

LinkerScript::AddressState::AddressState() {
  for (auto &mri : script->memoryRegions) {
    MemoryRegion *mr = mri.second;
    mr->curPos = (mr->origin)().getValue();
  }
}

// Here we assign addresses as instructed by linker script SECTIONS
// sub-commands. Doing that allows us to use final VA values, so here
// we also handle rest commands like symbol assignments and ASSERTs.
// Returns a symbol that has changed its section or value, or nullptr if no
// symbol has changed.
const Defined *LinkerScript::assignAddresses() {
  if (script->hasSectionsCommand) {
    // With a linker script, assignment of addresses to headers is covered by
    // allocateHeaders().
    dot = config->imageBase.getValueOr(0);
  } else {
    // Assign addresses to headers right now.
    dot = target->getImageBase();
    Out::elfHeader->addr = dot;
    Out::programHeaders->addr = dot + Out::elfHeader->size;
    dot += getHeaderSize();
  }

  AddressState state;
  ctx = &state;
  errorOnMissingSection = true;
  ctx->outSec = aether;

  SymbolAssignmentMap oldValues = getSymbolAssignmentValues(sectionCommands);
  for (SectionCommand *cmd : sectionCommands) {
    if (auto *assign = dyn_cast<SymbolAssignment>(cmd)) {
      assign->addr = dot;
      assignSymbol(assign, false);
      assign->size = dot - assign->addr;
      continue;
    }
    assignOffsets(cast<OutputSection>(cmd));
  }

  ctx = nullptr;
  return getChangedSymbolAssignment(oldValues);
}

// Creates program headers as instructed by PHDRS linker script command.
SmallVector<PhdrEntry *, 0> LinkerScript::createPhdrs() {
  SmallVector<PhdrEntry *, 0> ret;

  // Process PHDRS and FILEHDR keywords because they are not
  // real output sections and cannot be added in the following loop.
  for (const PhdrsCommand &cmd : phdrsCommands) {
    PhdrEntry *phdr = make<PhdrEntry>(cmd.type, cmd.flags.getValueOr(PF_R));

    if (cmd.hasFilehdr)
      phdr->add(Out::elfHeader);
    if (cmd.hasPhdrs)
      phdr->add(Out::programHeaders);

    if (cmd.lmaExpr) {
      phdr->p_paddr = cmd.lmaExpr().getValue();
      phdr->hasLMA = true;
    }
    ret.push_back(phdr);
  }

  // Add output sections to program headers.
  for (OutputSection *sec : outputSections) {
    // Assign headers specified by linker script
    for (size_t id : getPhdrIndices(sec)) {
      ret[id]->add(sec);
      if (!phdrsCommands[id].flags.hasValue())
        ret[id]->p_flags |= sec->getPhdrFlags();
    }
  }
  return ret;
}

// Returns true if we should emit an .interp section.
//
// We usually do. But if PHDRS commands are given, and
// no PT_INTERP is there, there's no place to emit an
// .interp, so we don't do that in that case.
bool LinkerScript::needsInterpSection() {
  if (phdrsCommands.empty())
    return true;
  for (PhdrsCommand &cmd : phdrsCommands)
    if (cmd.type == PT_INTERP)
      return true;
  return false;
}

ExprValue LinkerScript::getSymbolValue(StringRef name, const Twine &loc) {
  if (name == ".") {
    if (ctx)
      return {ctx->outSec, false, dot - ctx->outSec->addr, loc};
    error(loc + ": unable to get location counter value");
    return 0;
  }

  if (Symbol *sym = symtab->find(name)) {
    if (auto *ds = dyn_cast<Defined>(sym)) {
      ExprValue v{ds->section, false, ds->value, loc};
      // Retain the original st_type, so that the alias will get the same
      // behavior in relocation processing. Any operation will reset st_type to
      // STT_NOTYPE.
      v.type = ds->type;
      return v;
    }
    if (isa<SharedSymbol>(sym))
      if (!errorOnMissingSection)
        return {nullptr, false, 0, loc};
  }

  error(loc + ": symbol not found: " + name);
  return 0;
}

// Returns the index of the segment named Name.
static Optional<size_t> getPhdrIndex(ArrayRef<PhdrsCommand> vec,
                                     StringRef name) {
  for (size_t i = 0; i < vec.size(); ++i)
    if (vec[i].name == name)
      return i;
  return None;
}

// Returns indices of ELF headers containing specific section. Each index is a
// zero based number of ELF header listed within PHDRS {} script block.
SmallVector<size_t, 0> LinkerScript::getPhdrIndices(OutputSection *cmd) {
  SmallVector<size_t, 0> ret;

  for (StringRef s : cmd->phdrs) {
    if (Optional<size_t> idx = getPhdrIndex(phdrsCommands, s))
      ret.push_back(*idx);
    else if (s != "NONE")
      error(cmd->location + ": program header '" + s +
            "' is not listed in PHDRS");
  }
  return ret;
}
