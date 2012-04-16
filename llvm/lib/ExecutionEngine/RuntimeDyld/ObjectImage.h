//===---- ObjectImage.h - Format independent executuable object image -----===//
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

#ifndef LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H
#define LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H

#include "llvm/Object/ObjectFile.h"

namespace llvm {

class ObjectImage {
  ObjectImage(); // = delete
  ObjectImage(const ObjectImage &other); // = delete
protected:
  object::ObjectFile *ObjFile;

public:
  ObjectImage(object::ObjectFile *Obj) { ObjFile = Obj; }
  virtual ~ObjectImage() {}

  virtual object::symbol_iterator begin_symbols() const
              { return ObjFile->begin_symbols(); }
  virtual object::symbol_iterator end_symbols() const
              { return ObjFile->end_symbols(); }

  virtual object::section_iterator begin_sections() const
              { return ObjFile->begin_sections(); }
  virtual object::section_iterator end_sections() const
              { return ObjFile->end_sections(); }

  virtual /* Triple::ArchType */ unsigned getArch() const
              { return ObjFile->getArch(); }

  // Subclasses can override these methods to update the image with loaded
  // addresses for sections and common symbols
  virtual void updateSectionAddress(const object::SectionRef &Sec,
                                    uint64_t Addr) {}
  virtual void updateSymbolAddress(const object::SymbolRef &Sym, uint64_t Addr)
              {}

  // Subclasses can override this method to provide JIT debugging support
  virtual void registerWithDebugger() {}
  virtual void deregisterWithDebugger() {}
};

} // end namespace llvm

#endif // LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H

