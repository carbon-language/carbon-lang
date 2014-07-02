//===-- MCAtom.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCAtom class, which is used to
// represent a contiguous region in a decoded object that is uniformly data or
// instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCANALYSIS_MCATOM_H
#define LLVM_MC_MCANALYSIS_MCATOM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class MCModule;

class MCAtom;
class MCTextAtom;
class MCDataAtom;

/// \brief Represents a contiguous range of either instructions (a TextAtom)
/// or data (a DataAtom).  Address ranges are expressed as _closed_ intervals.
class MCAtom {
  virtual void anchor();
public:
  virtual ~MCAtom() {}

  enum AtomKind { TextAtom, DataAtom };
  AtomKind getKind() const { return Kind; }

  /// \brief Get the start address of the atom.
  uint64_t getBeginAddr() const { return Begin; }
  /// \brief Get the end address, i.e. the last one inside the atom.
  uint64_t getEndAddr() const { return End; }

  /// \name Atom modification methods:
  /// When modifying a TextAtom, keep instruction boundaries in mind.
  /// For instance, split must me given the start address of an instruction.
  /// @{

  /// \brief Splits the atom in two at a given address.
  /// \param SplitPt Address at which to start a new atom, splitting this one.
  /// \returns The newly created atom starting at \p SplitPt.
  virtual MCAtom *split(uint64_t SplitPt) = 0;

  /// \brief Truncates an atom, discarding everything after \p TruncPt.
  /// \param TruncPt Last byte address to be contained in this atom.
  virtual void truncate(uint64_t TruncPt) = 0;
  /// @}

  /// \name Naming:
  ///
  /// This is mostly for display purposes, and may contain anything that hints
  /// at what the atom contains: section or symbol name, BB start address, ..
  /// @{
  StringRef getName() const { return Name; }
  void setName(StringRef NewName) { Name = NewName.str(); }
  /// @}

protected:
  const AtomKind Kind;
  std::string Name;
  MCModule *Parent;
  uint64_t Begin, End;

  friend class MCModule;
  MCAtom(AtomKind K, MCModule *P, uint64_t B, uint64_t E)
    : Kind(K), Name("(unknown)"), Parent(P), Begin(B), End(E) { }

  /// \name Atom remapping helpers
  /// @{

  /// \brief Remap the atom, using the given range, updating Begin/End.
  /// One or both of the bounds can remain the same, but overlapping with other
  /// atoms in the module is still forbidden.
  void remap(uint64_t NewBegin, uint64_t NewEnd);

  /// \brief Remap the atom to prepare for a truncation at TruncPt.
  /// Equivalent to:
  /// \code
  ///   // Bound checks
  ///   remap(Begin, TruncPt);
  /// \endcode
  void remapForTruncate(uint64_t TruncPt);

  /// \brief Remap the atom to prepare for a split at SplitPt.
  /// The bounds for the resulting atoms are returned in {L,R}{Begin,End}.
  /// The current atom is truncated to \p LEnd.
  void remapForSplit(uint64_t SplitPt,
                     uint64_t &LBegin, uint64_t &LEnd,
                     uint64_t &RBegin, uint64_t &REnd);
  /// @}
};

/// \name Text atom
/// @{

/// \brief An entry in an MCTextAtom: a disassembled instruction.
/// NOTE: Both the Address and Size field are actually redundant when taken in
/// the context of the text atom, and may better be exposed in an iterator
/// instead of stored in the atom, which would replace this class.
class MCDecodedInst {
public:
  MCInst Inst;
  uint64_t Address;
  uint64_t Size;
  MCDecodedInst(const MCInst &Inst, uint64_t Address, uint64_t Size)
    : Inst(Inst), Address(Address), Size(Size) {}
};

/// \brief An atom consisting of disassembled instructions.
class MCTextAtom : public MCAtom {
private:
  typedef std::vector<MCDecodedInst> InstListTy;
  InstListTy Insts;

  /// \brief The address of the next appended instruction, i.e., the
  /// address immediately after the last instruction in the atom.
  uint64_t NextInstAddress;
public:
  /// Append an instruction, expanding the atom if necessary.
  void addInst(const MCInst &Inst, uint64_t Size);

  /// \name Instruction list access
  /// @{
  typedef InstListTy::const_iterator const_iterator;
  const_iterator begin() const { return Insts.begin(); }
  const_iterator end()   const { return Insts.end(); }

  const MCDecodedInst &back() const { return Insts.back(); }
  const MCDecodedInst &at(size_t n) const { return Insts.at(n); }
  size_t size() const { return Insts.size(); }
  /// @}

  /// \name Atom type specific split/truncate logic.
  /// @{
  MCTextAtom *split(uint64_t SplitPt) override;
  void     truncate(uint64_t TruncPt) override;
  /// @}

  // Class hierarchy.
  static bool classof(const MCAtom *A) { return A->getKind() == TextAtom; }
private:
  friend class MCModule;
  // Private constructor - only callable by MCModule
  MCTextAtom(MCModule *P, uint64_t Begin, uint64_t End)
    : MCAtom(TextAtom, P, Begin, End), NextInstAddress(Begin) {}
};
/// @}

/// \name Data atom
/// @{

/// \brief An entry in an MCDataAtom.
// NOTE: This may change to a more complex type in the future.
typedef uint8_t MCData;

/// \brief An atom consising of a sequence of bytes.
class MCDataAtom : public MCAtom {
  std::vector<MCData> Data;

public:
  /// Append a data entry, expanding the atom if necessary.
  void addData(const MCData &D);

  /// Get a reference to the data in this atom.
  ArrayRef<MCData> getData() const { return Data; }

  /// \name Atom type specific split/truncate logic.
  /// @{
  MCDataAtom *split(uint64_t SplitPt) override;
  void     truncate(uint64_t TruncPt) override;
  /// @}

  // Class hierarchy.
  static bool classof(const MCAtom *A) { return A->getKind() == DataAtom; }
private:
  friend class MCModule;
  // Private constructor - only callable by MCModule
  MCDataAtom(MCModule *P, uint64_t Begin, uint64_t End)
    : MCAtom(DataAtom, P, Begin, End) {
    Data.reserve(End + 1 - Begin);
  }
};

}

#endif
