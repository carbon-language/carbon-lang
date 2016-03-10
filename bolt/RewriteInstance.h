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
              uint64_t FileAddress = 0, uint64_t FileOffset = 0)
    : AllocAddress(Address), Size(Size), Alignment(Alignment),
      IsCode(IsCode), IsReadOnly(IsReadOnly), FileAddress(FileAddress),
      FileOffset(FileOffset) {}
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

  /// Populate array of binary functions and file symbols from file symbol
  /// table.
  void readSymbolTable();

  /// Read .eh_frame, .eh_frame_hdr and .gcc_except_table sections for exception
  /// and stack unwinding information.
  void readSpecialSections();

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

  /// Rewrite back all functions (hopefully optimized) that fit in the original
  /// memory footprint for that function. If the function is now larger and does
  /// not fit in the binary, reject it and preserve the original version of the
  /// function. If we couldn't understand the function for some reason in
  /// disassembleFunctions(), also preserve the original version.
  void rewriteFile();

private:

  /// Huge page size used for alignment.
  static constexpr unsigned PageAlign = 0x200000;

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

  /// Return file offset corresponding to a given virtual address.
  uint64_t getFileOffsetFor(uint64_t Address) {
    assert(Address >= NewTextSegmentAddress &&
           "address in not in the new text segment");
    return Address - NewTextSegmentAddress + NewTextSegmentOffset;
  }


private:
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

  /// Size of the .debug_line section on input.
  uint32_t DebugLineSize{0};

  /// Total hotness score according to profiling data for this binary.
  uint64_t TotalScore{0};

};

} // namespace bolt
} // namespace llvm

#endif
