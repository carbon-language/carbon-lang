//===-- llvm/MC/MCModule.h - MCModule class ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCModule class, which is used to
// represent a complete, disassembled object file or executable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMODULE_H
#define LLVM_MC_MCMODULE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class MCAtom;
class MCDataAtom;
class MCFunction;
class MCObjectDisassembler;
class MCTextAtom;

/// \brief A completely disassembled object file or executable.
/// It comprises a list of MCAtom's, each representing a contiguous range of
/// either instructions or data.
/// An MCModule is created using MCObjectDisassembler::buildModule.
class MCModule {
  /// \name Atom tracking
  /// @{

  /// \brief Atoms in this module, sorted by begin address.
  /// FIXME: This doesn't handle overlapping atoms (which happen when a basic
  /// block starts in the middle of an instruction of another basic block.)
  typedef std::vector<MCAtom*> AtomListTy;
  AtomListTy Atoms;

  friend class MCAtom;
  /// \brief Remap \p Atom to the given range, and update its Begin/End fields.
  /// \param Atom An atom belonging to this module.
  /// An atom should always use this method to update its bounds, because this
  /// enables the owning MCModule to keep track of its atoms.
  void remap(MCAtom *Atom, uint64_t NewBegin, uint64_t NewEnd);

  /// \brief Insert an atom in the module, using its Begin and End addresses.
  void map(MCAtom *NewAtom);
  /// @}

  /// \name Function tracking
  /// @{
  typedef std::vector<MCFunction*> FunctionListTy;
  FunctionListTy Functions;
  /// @}

  MCModule           (const MCModule &) LLVM_DELETED_FUNCTION;
  MCModule& operator=(const MCModule &) LLVM_DELETED_FUNCTION;

  // MCObjectDisassembler creates MCModules.
  friend class MCObjectDisassembler;
  MCModule() : Atoms() { }

public:
  ~MCModule();

  /// \name Create a new MCAtom covering the specified offset range.
  /// @{
  MCTextAtom *createTextAtom(uint64_t Begin, uint64_t End);
  MCDataAtom *createDataAtom(uint64_t Begin, uint64_t End);
  /// @}

  /// \name Access to the owned atom list, ordered by begin address.
  /// @{
  const MCAtom *findAtomContaining(uint64_t Addr) const;
        MCAtom *findAtomContaining(uint64_t Addr);

  typedef AtomListTy::const_iterator const_atom_iterator;
  typedef AtomListTy::      iterator       atom_iterator;
  const_atom_iterator atom_begin() const { return Atoms.begin(); }
        atom_iterator atom_begin()       { return Atoms.begin(); }
  const_atom_iterator atom_end()   const { return Atoms.end(); }
        atom_iterator atom_end()         { return Atoms.end(); }
  /// @}

  /// \name Create a new MCFunction.
  MCFunction *createFunction(const StringRef &Name);

  /// \name Access to the owned function list.
  /// @{
  typedef FunctionListTy::const_iterator const_func_iterator;
  typedef FunctionListTy::      iterator       func_iterator;
  const_func_iterator func_begin() const { return Functions.begin(); }
        func_iterator func_begin()       { return Functions.begin(); }
  const_func_iterator func_end()   const { return Functions.end(); }
        func_iterator func_end()         { return Functions.end(); }
  /// @}
};

}

#endif
