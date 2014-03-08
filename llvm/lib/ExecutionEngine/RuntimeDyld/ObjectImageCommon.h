//===-- ObjectImageCommon.h - Format independent executuable object image -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a file format independent ObjectImage class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLD_OBJECTIMAGECOMMON_H
#define LLVM_RUNTIMEDYLD_OBJECTIMAGECOMMON_H

#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/Object/ObjectFile.h"

namespace llvm {

namespace object {
  class ObjectFile;
}

class ObjectImageCommon : public ObjectImage {
  ObjectImageCommon(); // = delete
  ObjectImageCommon(const ObjectImageCommon &other); // = delete
  void anchor() override;

protected:
  object::ObjectFile *ObjFile;

  // This form of the constructor allows subclasses to use
  // format-specific subclasses of ObjectFile directly
  ObjectImageCommon(ObjectBuffer *Input, object::ObjectFile *Obj)
  : ObjectImage(Input), // saves Input as Buffer and takes ownership
    ObjFile(Obj)
  {
  }

public:
  ObjectImageCommon(ObjectBuffer* Input)
  : ObjectImage(Input) // saves Input as Buffer and takes ownership
  {
    ObjFile =
        object::ObjectFile::createObjectFile(Buffer->getMemBuffer()).get();
  }
  ObjectImageCommon(object::ObjectFile* Input)
  : ObjectImage(NULL), ObjFile(Input)  {}
  virtual ~ObjectImageCommon() { delete ObjFile; }

  object::symbol_iterator begin_symbols() const override
      { return ObjFile->symbol_begin(); }
  object::symbol_iterator end_symbols() const override
      { return ObjFile->symbol_end(); }

  object::section_iterator begin_sections() const override
      { return ObjFile->section_begin(); }
  object::section_iterator end_sections() const override
      { return ObjFile->section_end(); }

  /* Triple::ArchType */ unsigned getArch() const override
      { return ObjFile->getArch(); }

  StringRef getData() const override { return ObjFile->getData(); }

  object::ObjectFile* getObjectFile() const override { return ObjFile; }

  // Subclasses can override these methods to update the image with loaded
  // addresses for sections and common symbols
  void updateSectionAddress(const object::SectionRef &Sec,
                            uint64_t Addr) override {}
  void updateSymbolAddress(const object::SymbolRef &Sym,
                           uint64_t Addr) override {}

  // Subclasses can override these methods to provide JIT debugging support
  void registerWithDebugger() override {}
  void deregisterWithDebugger() override {}
};

} // end namespace llvm

#endif // LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H
