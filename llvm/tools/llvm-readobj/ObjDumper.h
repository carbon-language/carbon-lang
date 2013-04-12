//===-- ObjDumper.h -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_READOBJ_OBJDUMPER_H
#define LLVM_READOBJ_OBJDUMPER_H

namespace llvm {

namespace object {
  class ObjectFile;
}

class error_code;

template<typename T>
class OwningPtr;

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

protected:
  StreamWriter& W;
};

error_code createCOFFDumper(const object::ObjectFile *Obj,
                            StreamWriter& Writer,
                            OwningPtr<ObjDumper> &Result);

error_code createELFDumper(const object::ObjectFile *Obj,
                           StreamWriter& Writer,
                           OwningPtr<ObjDumper> &Result);

error_code createMachODumper(const object::ObjectFile *Obj,
                             StreamWriter& Writer,
                             OwningPtr<ObjDumper> &Result);

} // namespace llvm

#endif
