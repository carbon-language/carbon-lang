//===- llvm/Linker.h - Module Linker Interface ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKER_H
#define LLVM_LINKER_H

#include "llvm/ADT/SmallPtrSet.h"
#include <string>

namespace llvm {

class Module;
class StringRef;
class StructType;

/// This class provides the core functionality of linking in LLVM. It keeps a
/// pointer to the merged module so far. It doesn't take ownership of the
/// module since it is assumed that the user of this class will want to do
/// something with it after the linking.
class Linker {
  public:
    enum LinkerMode {
      DestroySource = 0, // Allow source module to be destroyed.
      PreserveSource = 1 // Preserve the source module.
    };

    Linker(Module *M);
    ~Linker();

    Module *getModule() const { return Composite; }
    void deleteModule();

    /// \brief Link \p Src into the composite. The source is destroyed if
    /// \p Mode is DestroySource and preserved if it is PreserveSource.
    /// If \p ErrorMsg is not null, information about any error is written
    /// to it.
    /// Returns true on error.
    bool linkInModule(Module *Src, unsigned Mode, std::string *ErrorMsg);
    bool linkInModule(Module *Src, std::string *ErrorMsg) {
      return linkInModule(Src, Linker::DestroySource, ErrorMsg);
    }

    static bool LinkModules(Module *Dest, Module *Src, unsigned Mode,
                            std::string *ErrorMsg);

  private:
    Module *Composite;
    SmallPtrSet<StructType*, 32> IdentifiedStructTypes;
};

} // End llvm namespace

#endif
