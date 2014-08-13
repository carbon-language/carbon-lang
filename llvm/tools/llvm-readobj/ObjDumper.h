//===-- ObjDumper.h -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_OBJDUMPER_H
#define LLVM_TOOLS_LLVM_READOBJ_OBJDUMPER_H

#include <memory>
#include <system_error>

namespace llvm {
namespace object {
  class ObjectFile;
}

class StreamWriter;

class ObjDumper {
public:
  ObjDumper(StreamWriter& Writer);
  virtual ~ObjDumper();

  virtual void printFileHeaders() = 0;
  virtual void printSections() = 0;
  virtual void printRelocations() = 0;
  virtual void printSymbols() = 0;
  virtual void printDynamicSymbols() = 0;
  virtual void printUnwindInfo() = 0;

  // Only implemented for ELF at this time.
  virtual void printDynamicTable() { }
  virtual void printNeededLibraries() { }
  virtual void printProgramHeaders() { }

  // Only implemented for ARM ELF at this time.
  virtual void printAttributes() { }

  // Only implemented for MIPS ELF at this time.
  virtual void printMipsPLTGOT() { }

protected:
  StreamWriter& W;
};

std::error_code createCOFFDumper(const object::ObjectFile *Obj,
                                 StreamWriter &Writer,
                                 std::unique_ptr<ObjDumper> &Result);

std::error_code createELFDumper(const object::ObjectFile *Obj,
                                StreamWriter &Writer,
                                std::unique_ptr<ObjDumper> &Result);

std::error_code createMachODumper(const object::ObjectFile *Obj,
                                  StreamWriter &Writer,
                                  std::unique_ptr<ObjDumper> &Result);

} // namespace llvm

#endif
