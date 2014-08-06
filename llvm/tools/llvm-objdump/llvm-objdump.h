//===-- llvm-objdump.h ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJDUMP_H
#define LLVM_OBJDUMP_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/StringRefMemoryObject.h"

namespace llvm {
namespace object {
  class COFFObjectFile;
  class MachOObjectFile;
  class ObjectFile;
  class RelocationRef;
}

extern cl::opt<std::string> TripleName;
extern cl::opt<std::string> ArchName;
extern cl::opt<std::string> MCPU;
extern cl::list<std::string> MAttrs;

// Various helper functions.
bool error(std::error_code ec);
bool RelocAddressLess(object::RelocationRef a, object::RelocationRef b);
void DumpBytes(StringRef bytes);
void DisassembleInputMachO(StringRef Filename);
void printCOFFUnwindInfo(const object::COFFObjectFile* o);
void printMachOUnwindInfo(const object::MachOObjectFile* o);

void printELFFileHeader(const object::ObjectFile *o);
void printCOFFFileHeader(const object::ObjectFile *o);

} // end namespace llvm

#endif
