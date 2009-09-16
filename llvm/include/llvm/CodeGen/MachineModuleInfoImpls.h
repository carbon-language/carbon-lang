//===-- llvm/CodeGen/MachineModuleInfoImpls.h -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H
#define LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H

#include "llvm/CodeGen/MachineModuleInfo.h"

namespace llvm {
  class MCSymbol;
  
  /// MachineModuleInfoMachO - This is a MachineModuleInfoImpl implementation
  /// for MachO targets.
  class MachineModuleInfoMachO : public MachineModuleInfoImpl {
    /// FnStubs - Darwin '$stub' stubs.  The key is something like "Lfoo$stub",
    /// the value is something like "_foo".
    DenseMap<const MCSymbol*, const MCSymbol*> FnStubs;
    
    /// GVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something like
    /// "Lfoo$non_lazy_ptr", the value is something like "_foo".
    DenseMap<const MCSymbol*, const MCSymbol*> GVStubs;
    
    /// HiddenGVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something like
    /// "Lfoo$non_lazy_ptr", the value is something like "_foo".  Unlike GVStubs
    /// these are for things with hidden visibility.
    DenseMap<const MCSymbol*, const MCSymbol*> HiddenGVStubs;
    
    virtual void Anchor();  // Out of line virtual method.
  public:
    MachineModuleInfoMachO(const MachineModuleInfo &) {}
    
    const MCSymbol *&getFnStubEntry(const MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return FnStubs[Sym];
    }

    const MCSymbol *&getGVStubEntry(const MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return GVStubs[Sym];
    }

    const MCSymbol *&getHiddenGVStubEntry(const MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return HiddenGVStubs[Sym];
    }
    
    /// Accessor methods to return the set of stubs in sorted order.
    typedef std::vector<std::pair<const MCSymbol*, const MCSymbol*> >
      SymbolListTy;
    
    SymbolListTy GetFnStubList() const {
      return GetSortedStubs(FnStubs);
    }
    SymbolListTy GetGVStubList() const {
      return GetSortedStubs(GVStubs);
    }
    SymbolListTy GetHiddenGVStubList() const {
      return GetSortedStubs(HiddenGVStubs);
    }
    
  private:
    static SymbolListTy
    GetSortedStubs(const DenseMap<const MCSymbol*, const MCSymbol*> &Map);
  };
  
} // end namespace llvm

#endif
