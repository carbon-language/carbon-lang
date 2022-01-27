//===- LinkerScript.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_LINKER_SCRIPT_H
#define LLD_ELF_LINKER_SCRIPT_H

#include "Config.h"
#include "Writer.h"
#include "lld/Common/LLVM.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace lld {
namespace elf {

class Defined;
class InputFile;
class InputSection;
class InputSectionBase;
class OutputSection;
class SectionBase;
class Symbol;
class ThunkSection;

// This represents an r-value in the linker script.
struct ExprValue {
  ExprValue(SectionBase *sec, bool forceAbsolute, uint64_t val,
            const Twine &loc)
      : sec(sec), val(val), forceAbsolute(forceAbsolute), loc(loc.str()) {}

  ExprValue(uint64_t val) : ExprValue(nullptr, false, val, "") {}

  bool isAbsolute() const { return forceAbsolute || sec == nullptr; }
  uint64_t getValue() const;
  uint64_t getSecAddr() const;
  uint64_t getSectionOffset() const;

  // If a value is relative to a section, it has a non-null Sec.
  SectionBase *sec;

  uint64_t val;
  uint64_t alignment = 1;

  // The original st_type if the expression represents a symbol. Any operation
  // resets type to STT_NOTYPE.
  uint8_t type = llvm::ELF::STT_NOTYPE;

  // True if this expression is enclosed in ABSOLUTE().
  // This flag affects the return value of getValue().
  bool forceAbsolute;

  // Original source location. Used for error messages.
  std::string loc;
};

// This represents an expression in the linker script.
// ScriptParser::readExpr reads an expression and returns an Expr.
// Later, we evaluate the expression by calling the function.
using Expr = std::function<ExprValue()>;

// This enum is used to implement linker script SECTIONS command.
// https://sourceware.org/binutils/docs/ld/SECTIONS.html#SECTIONS
enum SectionsCommandKind {
  AssignmentKind, // . = expr or <sym> = expr
  OutputSectionKind,
  InputSectionKind,
  ByteKind    // BYTE(expr), SHORT(expr), LONG(expr) or QUAD(expr)
};

struct SectionCommand {
  SectionCommand(int k) : kind(k) {}
  int kind;
};

// This represents ". = <expr>" or "<symbol> = <expr>".
struct SymbolAssignment : SectionCommand {
  SymbolAssignment(StringRef name, Expr e, std::string loc)
      : SectionCommand(AssignmentKind), name(name), expression(e),
        location(loc) {}

  static bool classof(const SectionCommand *c) {
    return c->kind == AssignmentKind;
  }

  // The LHS of an expression. Name is either a symbol name or ".".
  StringRef name;
  Defined *sym = nullptr;

  // The RHS of an expression.
  Expr expression;

  // Command attributes for PROVIDE, HIDDEN and PROVIDE_HIDDEN.
  bool provide = false;
  bool hidden = false;

  // Holds file name and line number for error reporting.
  std::string location;

  // A string representation of this command. We use this for -Map.
  std::string commandString;

  // Address of this assignment command.
  uint64_t addr;

  // Size of this assignment command. This is usually 0, but if
  // you move '.' this may be greater than 0.
  uint64_t size;
};

// Linker scripts allow additional constraints to be put on output sections.
// If an output section is marked as ONLY_IF_RO, the section is created
// only if its input sections are read-only. Likewise, an output section
// with ONLY_IF_RW is created if all input sections are RW.
enum class ConstraintKind { NoConstraint, ReadOnly, ReadWrite };

// This struct is used to represent the location and size of regions of
// target memory. Instances of the struct are created by parsing the
// MEMORY command.
struct MemoryRegion {
  MemoryRegion(StringRef name, Expr origin, Expr length, uint32_t flags,
               uint32_t invFlags, uint32_t negFlags, uint32_t negInvFlags)
      : name(std::string(name)), origin(origin), length(length), flags(flags),
        invFlags(invFlags), negFlags(negFlags), negInvFlags(negInvFlags) {}

  std::string name;
  Expr origin;
  Expr length;
  // A section can be assigned to the region if any of these ELF section flags
  // are set...
  uint32_t flags;
  // ... or any of these flags are not set.
  // For example, the memory region attribute "r" maps to SHF_WRITE.
  uint32_t invFlags;
  // A section cannot be assigned to the region if any of these ELF section
  // flags are set...
  uint32_t negFlags;
  // ... or any of these flags are not set.
  // For example, the memory region attribute "!r" maps to SHF_WRITE.
  uint32_t negInvFlags;
  uint64_t curPos = 0;

  bool compatibleWith(uint32_t secFlags) const {
    if ((secFlags & negFlags) || (~secFlags & negInvFlags))
      return false;
    return (secFlags & flags) || (~secFlags & invFlags);
  }
};

// This struct represents one section match pattern in SECTIONS() command.
// It can optionally have negative match pattern for EXCLUDED_FILE command.
// Also it may be surrounded with SORT() command, so contains sorting rules.
class SectionPattern {
  StringMatcher excludedFilePat;

  // Cache of the most recent input argument and result of excludesFile().
  mutable llvm::Optional<std::pair<const InputFile *, bool>> excludesFileCache;

public:
  SectionPattern(StringMatcher &&pat1, StringMatcher &&pat2)
      : excludedFilePat(pat1), sectionPat(pat2),
        sortOuter(SortSectionPolicy::Default),
        sortInner(SortSectionPolicy::Default) {}

  bool excludesFile(const InputFile *file) const;

  StringMatcher sectionPat;
  SortSectionPolicy sortOuter;
  SortSectionPolicy sortInner;
};

class InputSectionDescription : public SectionCommand {
  SingleStringMatcher filePat;

  // Cache of the most recent input argument and result of matchesFile().
  mutable llvm::Optional<std::pair<const InputFile *, bool>> matchesFileCache;

public:
  InputSectionDescription(StringRef filePattern, uint64_t withFlags = 0,
                          uint64_t withoutFlags = 0)
      : SectionCommand(InputSectionKind), filePat(filePattern),
        withFlags(withFlags), withoutFlags(withoutFlags) {}

  static bool classof(const SectionCommand *c) {
    return c->kind == InputSectionKind;
  }

  bool matchesFile(const InputFile *file) const;

  // Input sections that matches at least one of SectionPatterns
  // will be associated with this InputSectionDescription.
  SmallVector<SectionPattern, 0> sectionPatterns;

  // Includes InputSections and MergeInputSections. Used temporarily during
  // assignment of input sections to output sections.
  SmallVector<InputSectionBase *, 0> sectionBases;

  // Used after the finalizeInputSections() pass. MergeInputSections have been
  // merged into MergeSyntheticSections.
  SmallVector<InputSection *, 0> sections;

  // Temporary record of synthetic ThunkSection instances and the pass that
  // they were created in. This is used to insert newly created ThunkSections
  // into Sections at the end of a createThunks() pass.
  SmallVector<std::pair<ThunkSection *, uint32_t>, 0> thunkSections;

  // SectionPatterns can be filtered with the INPUT_SECTION_FLAGS command.
  uint64_t withFlags;
  uint64_t withoutFlags;
};

// Represents BYTE(), SHORT(), LONG(), or QUAD().
struct ByteCommand : SectionCommand {
  ByteCommand(Expr e, unsigned size, std::string commandString)
      : SectionCommand(ByteKind), commandString(commandString), expression(e),
        size(size) {}

  static bool classof(const SectionCommand *c) { return c->kind == ByteKind; }

  // Keeps string representing the command. Used for -Map" is perhaps better.
  std::string commandString;

  Expr expression;

  // This is just an offset of this assignment command in the output section.
  unsigned offset;

  // Size of this data command.
  unsigned size;
};

struct InsertCommand {
  SmallVector<StringRef, 0> names;
  bool isAfter;
  StringRef where;
};

struct PhdrsCommand {
  StringRef name;
  unsigned type = llvm::ELF::PT_NULL;
  bool hasFilehdr = false;
  bool hasPhdrs = false;
  llvm::Optional<unsigned> flags;
  Expr lmaExpr = nullptr;
};

class LinkerScript final {
  // Temporary state used in processSectionCommands() and assignAddresses()
  // that must be reinitialized for each call to the above functions, and must
  // not be used outside of the scope of a call to the above functions.
  struct AddressState {
    AddressState();
    OutputSection *outSec = nullptr;
    MemoryRegion *memRegion = nullptr;
    MemoryRegion *lmaRegion = nullptr;
    uint64_t lmaOffset = 0;
    uint64_t tbssAddr = 0;
  };

  llvm::DenseMap<llvm::CachedHashStringRef, OutputSection *>
      nameToOutputSection;

  void addSymbol(SymbolAssignment *cmd);
  void assignSymbol(SymbolAssignment *cmd, bool inSec);
  void setDot(Expr e, const Twine &loc, bool inSec);
  void expandOutputSection(uint64_t size);
  void expandMemoryRegions(uint64_t size);

  SmallVector<InputSectionBase *, 0>
  computeInputSections(const InputSectionDescription *,
                       ArrayRef<InputSectionBase *>);

  SmallVector<InputSectionBase *, 0> createInputSectionList(OutputSection &cmd);

  void discardSynthetic(OutputSection &);

  SmallVector<size_t, 0> getPhdrIndices(OutputSection *sec);

  std::pair<MemoryRegion *, MemoryRegion *>
  findMemoryRegion(OutputSection *sec, MemoryRegion *hint);

  void assignOffsets(OutputSection *sec);

  // Ctx captures the local AddressState and makes it accessible
  // deliberately. This is needed as there are some cases where we cannot just
  // thread the current state through to a lambda function created by the
  // script parser.
  // This should remain a plain pointer as its lifetime is smaller than
  // LinkerScript.
  AddressState *ctx = nullptr;

  OutputSection *aether;

  uint64_t dot;

public:
  OutputSection *createOutputSection(StringRef name, StringRef location);
  OutputSection *getOrCreateOutputSection(StringRef name);

  bool hasPhdrsCommands() { return !phdrsCommands.empty(); }
  uint64_t getDot() { return dot; }
  void discard(InputSectionBase &s);

  ExprValue getSymbolValue(StringRef name, const Twine &loc);

  void addOrphanSections();
  void diagnoseOrphanHandling() const;
  void adjustSectionsBeforeSorting();
  void adjustSectionsAfterSorting();

  SmallVector<PhdrEntry *, 0> createPhdrs();
  bool needsInterpSection();

  bool shouldKeep(InputSectionBase *s);
  const Defined *assignAddresses();
  void allocateHeaders(SmallVector<PhdrEntry *, 0> &phdrs);
  void processSectionCommands();
  void processSymbolAssignments();
  void declareSymbols();

  bool isDiscarded(const OutputSection *sec) const;

  // Used to handle INSERT AFTER statements.
  void processInsertCommands();

  // SECTIONS command list.
  SmallVector<SectionCommand *, 0> sectionCommands;

  // PHDRS command list.
  SmallVector<PhdrsCommand, 0> phdrsCommands;

  bool hasSectionsCommand = false;
  bool errorOnMissingSection = false;

  // List of section patterns specified with KEEP commands. They will
  // be kept even if they are unused and --gc-sections is specified.
  SmallVector<InputSectionDescription *, 0> keptSections;

  // A map from memory region name to a memory region descriptor.
  llvm::MapVector<llvm::StringRef, MemoryRegion *> memoryRegions;

  // A list of symbols referenced by the script.
  SmallVector<llvm::StringRef, 0> referencedSymbols;

  // Used to implement INSERT [AFTER|BEFORE]. Contains output sections that need
  // to be reordered.
  SmallVector<InsertCommand, 0> insertCommands;

  // OutputSections specified by OVERWRITE_SECTIONS.
  SmallVector<OutputSection *, 0> overwriteSections;

  // Sections that will be warned/errored by --orphan-handling.
  SmallVector<const InputSectionBase *, 0> orphanSections;
};

extern std::unique_ptr<LinkerScript> script;

} // end namespace elf
} // end namespace lld

#endif // LLD_ELF_LINKER_SCRIPT_H
