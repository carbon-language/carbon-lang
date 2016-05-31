//===-- llvm/GlobalObject.h - Class to represent global objects -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This represents an independent object. That is, a function or a global
// variable, but not an alias.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GLOBALOBJECT_H
#define LLVM_IR_GLOBALOBJECT_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"

namespace llvm {
class Comdat;
class MDNode;
class Module;

class GlobalObject : public GlobalValue {
  GlobalObject(const GlobalObject &) = delete;

protected:
  GlobalObject(Type *Ty, ValueTy VTy, Use *Ops, unsigned NumOps,
               LinkageTypes Linkage, const Twine &Name,
               unsigned AddressSpace = 0)
      : GlobalValue(Ty, VTy, Ops, NumOps, Linkage, Name, AddressSpace),
        ObjComdat(nullptr) {
    setGlobalValueSubClassData(0);
  }

  std::string Section;     // Section to emit this into, empty means default
  Comdat *ObjComdat;
  enum {
    LastAlignmentBit = 4,
    HasMetadataHashEntryBit,

    GlobalObjectBits,
  };
  static const unsigned GlobalObjectSubClassDataBits =
      GlobalValueSubClassDataBits - GlobalObjectBits;

private:
  static const unsigned AlignmentBits = LastAlignmentBit + 1;
  static const unsigned AlignmentMask = (1 << AlignmentBits) - 1;
  static const unsigned GlobalObjectMask = (1 << GlobalObjectBits) - 1;

public:
  unsigned getAlignment() const {
    unsigned Data = getGlobalValueSubClassData();
    unsigned AlignmentData = Data & AlignmentMask;
    return (1u << AlignmentData) >> 1;
  }
  void setAlignment(unsigned Align);

  unsigned getGlobalObjectSubClassData() const;
  void setGlobalObjectSubClassData(unsigned Val);

  bool hasSection() const { return !getSection().empty(); }
  StringRef getSection() const { return Section; }
  void setSection(StringRef S);

  bool hasComdat() const { return getComdat() != nullptr; }
  const Comdat *getComdat() const { return ObjComdat; }
  Comdat *getComdat() { return ObjComdat; }
  void setComdat(Comdat *C) { ObjComdat = C; }

  /// Check if this has any metadata.
  bool hasMetadata() const { return hasMetadataHashEntry(); }

  /// Get the current metadata attachment, if any.
  ///
  /// Returns \c nullptr if such an attachment is missing.
  /// @{
  MDNode *getMetadata(unsigned KindID) const;
  MDNode *getMetadata(StringRef Kind) const;
  /// @}

  /// Set a particular kind of metadata attachment.
  ///
  /// Sets the given attachment to \c MD, erasing it if \c MD is \c nullptr or
  /// replacing it if it already exists.
  /// @{
  void setMetadata(unsigned KindID, MDNode *MD);
  void setMetadata(StringRef Kind, MDNode *MD);
  /// @}

  /// Get all current metadata attachments.
  void
  getAllMetadata(SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const;

  /// Drop metadata not in the given list.
  ///
  /// Drop all metadata from \c this not included in \c KnownIDs.
  void dropUnknownMetadata(ArrayRef<unsigned> KnownIDs);

  void copyAttributesFrom(const GlobalValue *Src) override;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::FunctionVal ||
           V->getValueID() == Value::GlobalVariableVal;
  }

protected:
  void clearMetadata();

private:
  bool hasMetadataHashEntry() const {
    return getGlobalValueSubClassData() & (1 << HasMetadataHashEntryBit);
  }
  void setHasMetadataHashEntry(bool HasEntry) {
    unsigned Mask = 1 << HasMetadataHashEntryBit;
    setGlobalValueSubClassData((~Mask & getGlobalValueSubClassData()) |
                               (HasEntry ? Mask : 0u));
  }
};

} // End llvm namespace

#endif
