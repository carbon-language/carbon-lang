//===- ModInfo.h - PDB module information -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_MODINFO_H
#define LLVM_DEBUGINFO_PDB_RAW_MODINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamRef.h"
#include "llvm/Support/Endian.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace pdb {

class ModInfo {
  friend class DbiStreamBuilder;

private:
  typedef support::ulittle16_t ulittle16_t;
  typedef support::ulittle32_t ulittle32_t;
  typedef support::little32_t little32_t;

  struct SCBytes {
    ulittle16_t Section;
    char Padding1[2];
    little32_t Offset;
    little32_t Size;
    ulittle32_t Characteristics;
    ulittle16_t ModuleIndex;
    char Padding2[2];
    ulittle32_t DataCrc;
    ulittle32_t RelocCrc;
  };

  // struct Flags {
  //  uint16_t fWritten : 1;   // True if ModInfo is dirty
  //  uint16_t fECEnabled : 1; // Is EC symbolic info present?  (What is EC?)
  //  uint16_t unused : 6;     // Reserved
  //  uint16_t iTSM : 8;       // Type Server Index for this module
  //};
  const uint16_t HasECFlagMask = 0x2;

  const uint16_t TypeServerIndexMask = 0xFF00;
  const uint16_t TypeServerIndexShift = 8;

  struct FileLayout {
    ulittle32_t Mod;          // Currently opened module.  This field is a
                              // pointer in the reference implementation, but
                              // that won't work on 64-bit systems, and anyway
                              // it doesn't make sense to read a pointer from a
                              // file.  For now it is unused, so just ignore it.
    SCBytes SC;               // First section contribution of this module.
    ulittle16_t Flags;        // See Flags definition.
    ulittle16_t ModDiStream;  // Stream Number of module debug info
    ulittle32_t SymBytes;     // Size of local symbol debug info in above stream
    ulittle32_t LineBytes;    // Size of line number debug info in above stream
    ulittle32_t C13Bytes;     // Size of C13 line number info in above stream
    ulittle16_t NumFiles;     // Number of files contributing to this module
    char Padding1[2];         // Padding so the next field is 4-byte aligned.
    ulittle32_t FileNameOffs; // array of [0..NumFiles) DBI name buffer offsets.
                              // This field is a pointer in the reference
                              // implementation, but as with `Mod`, we ignore it
                              // for now since it is unused.
    ulittle32_t SrcFileNameNI; // Name Index for src file name
    ulittle32_t PdbFilePathNI; // Name Index for path to compiler PDB
                               // Null terminated Module name
                               // Null terminated Obj File Name
  };

public:
  ModInfo();
  ModInfo(const ModInfo &Info);
  ~ModInfo();

  static Error initialize(codeview::StreamRef Stream, ModInfo &Info);

  bool hasECInfo() const;
  uint16_t getTypeServerIndex() const;
  uint16_t getModuleStreamIndex() const;
  uint32_t getSymbolDebugInfoByteSize() const;
  uint32_t getLineInfoByteSize() const;
  uint32_t getC13LineInfoByteSize() const;
  uint32_t getNumberOfFiles() const;
  uint32_t getSourceFileNameIndex() const;
  uint32_t getPdbFilePathNameIndex() const;

  StringRef getModuleName() const;
  StringRef getObjFileName() const;

  uint32_t getRecordLength() const;

private:
  StringRef ModuleName;
  StringRef ObjFileName;
  const FileLayout *Layout;
};

struct ModuleInfoEx {
  ModuleInfoEx(const ModInfo &Info) : Info(Info) {}
  ModuleInfoEx(const ModuleInfoEx &Ex)
      : Info(Ex.Info), SourceFiles(Ex.SourceFiles) {}

  ModInfo Info;
  std::vector<StringRef> SourceFiles;
};

} // end namespace pdb

namespace codeview {
template <> struct VarStreamArrayExtractor<pdb::ModInfo> {
  Error operator()(StreamRef Stream, uint32_t &Length,
                   pdb::ModInfo &Info) const {
    if (auto EC = pdb::ModInfo::initialize(Stream, Info))
      return EC;
    Length = Info.getRecordLength();
    return Error::success();
  }
};
}

} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_RAW_MODINFO_H
