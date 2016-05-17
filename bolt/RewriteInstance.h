//===--- RewriteInstance.h - Interface for machine-level function ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to control an instance of a binary rewriting process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_REWRITE_INSTANCE_H
#define LLVM_TOOLS_LLVM_BOLT_REWRITE_INSTANCE_H

#include "DebugData.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include <map>
#include <set>

namespace llvm {

class DWARFContext;
class DWARFFrame;
class tool_output_file;

namespace bolt {

class BinaryContext;
class BinaryFunction;
class CFIReaderWriter;
class DataReader;

/// Section information for mapping and re-writing.
struct SectionInfo {
  uint64_t AllocAddress;      /// Current location of the section in memory.
  uint64_t Size;              /// Section size.
  unsigned Alignment;         /// Alignment of the section.
  bool     IsCode{false};     /// Does this section contain code?
  bool     IsReadOnly{false}; /// Is the section read-only?
  uint64_t FileAddress{0};    /// Address for the output file (final address).
  uint64_t FileOffset{0};     /// Offset in the output file.
  uint64_t ShName{0};         /// Name offset in section header string table.
  unsigned SectionID{0};      /// Unique ID used for address mapping.

  struct Reloc {
    uint32_t Offset;
    uint8_t  Size;
    uint8_t  Type; // unused atm
    uint32_t Value;
  };

  /// Pending relocations for the section.
  std::vector<Reloc> PendingRelocs;

  SectionInfo(uint64_t Address = 0, uint64_t Size = 0, unsigned Alignment = 0,
              bool IsCode = false, bool IsReadOnly = false,
              uint64_t FileAddress = 0, uint64_t FileOffset = 0,
              unsigned SectionID = 0)
    : AllocAddress(Address), Size(Size), Alignment(Alignment), IsCode(IsCode),
      IsReadOnly(IsReadOnly), FileAddress(FileAddress), FileOffset(FileOffset),
      SectionID(SectionID) {}
};

/// Class responsible for allocating and managing code and data sections.
class ExecutableFileMemoryManager : public SectionMemoryManager {
private:
  uint8_t *allocateSection(intptr_t Size,
                           unsigned Alignment,
                           unsigned SectionID,
                           StringRef SectionName,
                           bool IsCode,
                           bool IsReadOnly);

public:

  /// Keep [section name] -> [section info] map for later remapping.
  std::map<std::string, SectionInfo> SectionMapInfo;

  /// Information about non-allocatable sections.
  std::map<std::string, SectionInfo> NoteSectionInfo;

  ExecutableFileMemoryManager() {}

  ~ExecutableFileMemoryManager();

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    return allocateSection(Size, Alignment, SectionID, SectionName,
                           /*IsCode=*/true, true);
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {
    return allocateSection(Size, Alignment, SectionID, SectionName,
                           /*IsCode=*/false, IsReadOnly);
  }

  uint8_t *recordNoteSection(const uint8_t *Data, uintptr_t Size,
                             unsigned Alignment, unsigned SectionID,
                             StringRef SectionName) override;

  // Tell EE that we guarantee we don't need stubs.
  bool allowStubAllocation() const override { return false; }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override;
};

/// This class encapsulates all data necessary to carry on binary reading,
/// disassembly, CFG building, BB reordering (among other binary-level
/// optimizations) and rewriting. It also has the logic to coordinate such
/// events.
class RewriteInstance {
public:
  RewriteInstance(llvm::object::ELFObjectFileBase *File, const DataReader &DR);
  ~RewriteInstance();

  /// Reset all state except for split hints. Used to run a second pass with
  /// function splitting information.
  void reset();

  /// Run all the necessary steps to read, optimize and rewrite the binary.
  void run();

  /// Populate array of binary functions and other objects of interest
  /// from meta data in the file.
  void discoverFileObjects();

  /// Read .eh_frame, .eh_frame_hdr and .gcc_except_table sections for exception
  /// and stack unwinding information.
  void readSpecialSections();

  /// Read information from debug sections.
  void readDebugInfo();

  /// Read information from debug sections that depends on disassembled
  /// functions.
  void readFunctionDebugInfo();

  /// Disassemble each function in the binary and associate it with a
  /// BinaryFunction object, preparing all information necessary for binary
  /// optimization.
  void disassembleFunctions();

  /// Run optimizations that operate at the binary, or post-linker, level.
  void runOptimizationPasses();

  /// Write all functions to an intermediary object file, map virtual to real
  /// addresses and link this object file, resolving all relocations and
  /// performing final relaxation.
  void emitFunctions();

  /// Update debug information in the file for re-written code.
  void updateDebugInfo();

  /// Check which functions became larger than their original version and
  /// annotate function splitting information.
  ///
  /// Returns true if any function was annotated, requiring us to perform a
  /// second pass to emit those functions in two parts.
  bool checkLargeFunctions();

  /// Updates debug line information for non-simple functions, which are not
  /// rewritten.
  void updateDebugLineInfoForNonSimpleFunctions();

  /// Rewrite back all functions (hopefully optimized) that fit in the original
  /// memory footprint for that function. If the function is now larger and does
  /// not fit in the binary, reject it and preserve the original version of the
  /// function. If we couldn't understand the function for some reason in
  /// disassembleFunctions(), also preserve the original version.
  void rewriteFile();

private:

  /// Detect addresses and offsets available in the binary for allocating
  /// new sections.
  void discoverStorage();

  /// Rewrite non-allocatable sections with modifications.
  void rewriteNoteSections();

  /// Patch ELF book-keeping info.
  void patchELF();
  void patchELFPHDRTable();
  void patchELFSectionHeaderTable();

  /// Computes output .debug_line line table offsets for each compile unit, and
  /// stores them into BinaryContext::CompileUnitLineTableOffset.
  void computeLineTableOffsets();

  /// Adds an entry to be saved in the .debug_aranges/.debug_ranges section.
  /// \p OriginalFunctionAddress function's address in the original binary,
  /// used for compile unit lookup.
  /// \p RangeBegin first address of the address range being added.
  /// \p RangeSie size in bytes of the address range.
  void addDebugRangesEntry(uint64_t OriginalFunctionAddress,
                           uint64_t RangeBegin,
                           uint64_t RangeSize);

  /// Update internal function ranges after functions have been written.
  void updateFunctionRanges();

  /// Update objects with address ranges after optimization.
  void updateAddressRangesObjects();

  /// Generate new contents for .debug_loc.
  void updateLocationLists();

  /// Generate new contents for .debug_ranges and .debug_aranges section.
  void generateDebugRanges();

  /// Patches the binary for DWARF address ranges (e.g. in functions and lexical
  /// blocks) to be updated.
  void updateDWARFAddressRanges();

  /// Patches the binary for an object's address ranges to be updated.
  /// The object can be a anything that has associated address ranges via either
  /// DW_AT_low/high_pc or DW_AT_ranges (i.e. functions, lexical blocks, etc).
  /// \p DebugRangesOffset is the offset in .debug_ranges of the object's
  /// new address ranges in the output binary.
  /// \p Unit Compile uniit the object belongs to.
  /// \p DIE is the object's DIE in the input binary.
  void updateDWARFObjectAddressRanges(uint32_t DebugRangesOffset,
                                      const DWARFUnit *Unit,
                                      const DWARFDebugInfoEntryMinimal *DIE);

  /// Updates pointers in .debug_info to location lists in .debug_loc.
  void updateLocationListPointers(
      const DWARFUnit *Unit,
      const DWARFDebugInfoEntryMinimal *DIE,
      const std::map<uint32_t, uint32_t> &UpdatedOffsets);

  /// Return file offset corresponding to a given virtual address.
  uint64_t getFileOffsetFor(uint64_t Address) {
    assert(Address >= NewTextSegmentAddress &&
           "address in not in the new text segment");
    return Address - NewTextSegmentAddress + NewTextSegmentOffset;
  }

  /// Return true if we should overwrite contents of the section instead
  /// of appending contents to it.
  bool shouldOverwriteSection(StringRef SectionName);

private:

  /// If we are updating debug info, these are the section we need to overwrite.
  static constexpr const char *DebugSectionsToOverwrite[] = {
    ".debug_aranges",
    ".debug_line"};

  /// Huge page size used for alignment.
  static constexpr unsigned PageAlign = 0x200000;

  /// An instance of the input binary we are processing, externally owned.
  llvm::object::ELFObjectFileBase *InputFile;

  std::unique_ptr<BinaryContext> BC;
  std::unique_ptr<CFIReaderWriter> CFIRdWrt;
  /// Our in-memory intermediary object file where we hold final code for
  /// rewritten functions.
  std::unique_ptr<ExecutableFileMemoryManager> SectionMM;
  /// Our output file where we mix original code from the input binary and
  /// optimized code for selected functions.
  std::unique_ptr<tool_output_file> Out;

  /// Offset in the input file where non-allocatable sections start.
  uint64_t FirstNonAllocatableOffset{0};

  /// Information about program header table.
  uint64_t PHDRTableAddress{0};
  uint64_t PHDRTableOffset{0};
  unsigned Phnum{0};

  /// New code segment info.
  uint64_t NewTextSegmentAddress{0};
  uint64_t NewTextSegmentOffset{0};
  uint64_t NewTextSegmentSize{0};

  /// Track next available address in the new text segment.
  uint64_t NextAvailableAddress{0};

  /// Information on sections to re-write in the binary.
  std::map<std::string, SectionInfo> SectionsToRewrite;

  /// Store all non-zero symbols in this map for a quick address lookup.
  std::map<uint64_t, llvm::object::SymbolRef> FileSymRefs;

  /// Store all functions seen in the binary, sorted by address.
  std::map<uint64_t, BinaryFunction> BinaryFunctions;

  /// Stores and serializes information that will be put into the .debug_ranges
  /// and .debug_aranges DWARF sections.
  DebugRangesSectionsWriter RangesSectionsWriter;

  /// Patchers used to apply simple changes to sections of the input binary.
  /// Maps section name -> patcher.
  std::map<std::string, std::unique_ptr<BinaryPatcher>> SectionPatchers;

  /// Exception handling and stack unwinding information in this binary.
  ArrayRef<uint8_t> LSDAData;
  uint64_t LSDAAddress{0};
  std::vector<char> FrameHdrCopy;
  uint64_t FrameHdrAddress{0};
  uint64_t FrameHdrAlign{1};
  const llvm::DWARFFrame *EHFrame{nullptr};
  StringRef NewEhFrameContents;

  /// Keep track of functions we fail to write in the binary. We need to avoid
  /// rewriting CFI info for these functions.
  std::vector<uint64_t> FailedAddresses;

  /// Size of the .debug_loc section in input.
  uint32_t DebugLocSize{0};

  /// Size of the .debug_ranges section on input.
  uint32_t DebugRangesSize{0};

  /// Keep track of which functions didn't fit in their original space in the
  /// last emission, so that we may either decide to split or not optimize them.
  std::set<uint64_t> LargeFunctions;

  /// Total hotness score according to profiling data for this binary.
  uint64_t TotalScore{0};

};

} // namespace bolt
} // namespace llvm

#endif
