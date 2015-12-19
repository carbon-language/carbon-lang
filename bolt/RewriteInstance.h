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

#ifndef LLVM_TOOLS_LLVM_FLO_REWRITE_INSTANCE_H
#define LLVM_TOOLS_LLVM_FLO_REWRITE_INSTANCE_H

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

namespace flo {

class BinaryContext;
class BinaryFunction;
class CFIReaderWriter;
class DataReader;

/// Section information for mapping and re-writing.
struct SectionInfo {
  uint64_t AllocAddress;      /// Current location of the section in memory.
  uint64_t Size;              /// Section size.
  unsigned Alignment;         /// Alignment of the section.
  uint64_t FileAddress{0};    /// Address in the output file.
  uint64_t FileOffset{0};     /// Offset in the output file.
  bool     IsCode{false};     /// Does this section contain code.

  SectionInfo(uint64_t Address = 0, uint64_t Size = 0, unsigned Alignment = 0,
              bool IsCode = false)
    : AllocAddress(Address), Size(Size), Alignment(Alignment),
      IsCode(IsCode) {}
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

  // Keep [section name] -> [section info] map for later remapping.
  std::map<std::string, SectionInfo> SectionMapInfo;

  ExecutableFileMemoryManager() {}

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

  /// Check which functions became larger than their original version and
  /// annotate function splitting information.
  ///
  /// Returns true if any function was annotated, requiring us to perform a
  /// second pass to emit those functions in two parts.
  bool splitLargeFunctions();

  /// Rewrite back all functions (hopefully optimized) that fit in the original
  /// memory footprint for that function. If the function is now larger and does
  /// not fit in the binary, reject it and preserve the original version of the
  /// function. If we couldn't understand the function for some reason in
  /// disassembleFunctions(), also preserve the original version.
  void rewriteFile();

private:
  /// An instance of the input binary we are processing, externally owned.
  llvm::object::ELFObjectFileBase *File;

  std::unique_ptr<BinaryContext> BC;
  std::unique_ptr<DWARFContext> DwCtx;
  std::unique_ptr<CFIReaderWriter> CFIRdWrt;
  /// Our in-memory intermediary object file where we hold final code for
  /// rewritten functions.
  std::unique_ptr<ExecutableFileMemoryManager> SectionMM;
  /// Our output file where we mix original code from the input binary and
  /// optimized code for selected functions.
  std::unique_ptr<tool_output_file> Out;

  /// Represent free space we have in the binary to write extra bytes. This free
  /// space is pre-delimited in the binary via a linker script that allocates
  /// space and inserts a new symbol __flo_storage in the binary. We also use
  /// the symbol __flo_storage_end to delimit the end of the contiguous space in
  /// the binary where it is safe for us to write new content. We use this extra
  /// space for the following activities:
  ///
  ///   * Writing new .eh_frame entries for functions we changed the layout
  ///   * Writing a new .eh_frame_hdr to allow us to expand the number of
  ///     .eh_frame entries (FDEs). Note we also keep the old .eh_frame in the
  ///     binary instact for functions we don't touch.
  ///   * Writing cold basic blocks
  ///
  struct BlobTy {
    uint64_t Addr;
    uint64_t FileOffset;
    uint64_t Size;
    uint64_t AddrEnd;
    /// BumpPtr is a trivial way to keep track of space utilization in this blob
    uint64_t BumpPtr;
  };
  BlobTy ExtraStorage{0, 0, 0, 0, 0};

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
  uint64_t NewEhFrameAddress{0};
  uint64_t NewEhFrameOffset{0};

  /// Keep track of functions we fail to write in the binary. We need to avoid
  /// rewriting CFI info for these functions.
  std::vector<uint64_t> FailedAddresses;

  /// Keep track of which functions to split in a second pass.
  std::set<uint64_t> ToSplit;

  /// Total hotness score according to profiling data for this binary.
  uint64_t TotalScore{0};

};

} // namespace flo
} // namespace llvm

#endif
