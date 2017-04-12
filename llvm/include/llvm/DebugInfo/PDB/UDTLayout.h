//===- UDTLayout.h - UDT layout info ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_UDTLAYOUT_H
#define LLVM_DEBUGINFO_PDB_UDTLAYOUT_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"

#include <list>
#include <memory>

namespace llvm {

class raw_ostream;

namespace pdb {

class PDBSymTypeBaseClass;
class PDBSymbolData;
class PDBSymbolTypeUDT;
class PDBSymbolTypeVTable;

class ClassLayout;
class BaseClassLayout;
class StorageItemBase;
class UDTLayoutBase;

class StorageItemBase {
public:
  StorageItemBase(const UDTLayoutBase &Parent, const PDBSymbol &Symbol,
                  const std::string &Name, uint32_t OffsetInParent,
                  uint32_t Size);
  virtual ~StorageItemBase() {}

  virtual uint32_t deepPaddingSize() const;

  const UDTLayoutBase &getParent() const { return Parent; }
  StringRef getName() const { return Name; }
  uint32_t getOffsetInParent() const { return OffsetInParent; }
  uint32_t getSize() const { return SizeOf; }
  const PDBSymbol &getSymbol() const { return Symbol; }

protected:
  const UDTLayoutBase &Parent;
  const PDBSymbol &Symbol;
  BitVector UsedBytes;
  std::string Name;
  uint32_t OffsetInParent = 0;
  uint32_t SizeOf = 0;
};

class DataMemberLayoutItem : public StorageItemBase {
public:
  DataMemberLayoutItem(const UDTLayoutBase &Parent,
                       std::unique_ptr<PDBSymbolData> DataMember);

  virtual uint32_t deepPaddingSize() const;

  const PDBSymbolData &getDataMember();

private:
  std::unique_ptr<PDBSymbolData> DataMember;
  std::unique_ptr<ClassLayout> UdtLayout;
};

class VTableLayoutItem : public StorageItemBase {
public:
  VTableLayoutItem(const UDTLayoutBase &Parent,
                   std::unique_ptr<PDBSymbolTypeVTable> VTable);

private:
  std::unique_ptr<PDBSymbolTypeVTable> VTable;
  std::vector<std::unique_ptr<PDBSymbolFunc>> VTableFuncs;
};

class UDTLayoutBase {
public:
  UDTLayoutBase(const PDBSymbol &Symbol, const std::string &Name,
                uint32_t Size);

  uint32_t shallowPaddingSize() const;
  uint32_t deepPaddingSize() const;

  const BitVector &usedBytes() const { return UsedBytes; }

  uint32_t getClassSize() const { return SizeOf; }

  const PDBSymbol &getSymbol() const { return Symbol; }

  ArrayRef<std::unique_ptr<StorageItemBase>> layout_items() const {
    return ChildStorage;
  }

  ArrayRef<std::unique_ptr<PDBSymbol>> other_items() const {
    return NonStorageItems;
  }

protected:
  void initializeChildren(const PDBSymbol &Sym);

  void addChildToLayout(std::unique_ptr<StorageItemBase> Child);

  uint32_t SizeOf = 0;
  std::string Name;

  const PDBSymbol &Symbol;
  BitVector UsedBytes;
  std::vector<std::unique_ptr<PDBSymbol>> NonStorageItems;
  std::vector<std::unique_ptr<StorageItemBase>> ChildStorage;
  std::vector<std::list<StorageItemBase *>> ChildrenPerByte;
  std::vector<BaseClassLayout *> BaseClasses;
  VTableLayoutItem *VTable = nullptr;
};

class ClassLayout : public UDTLayoutBase {
public:
  explicit ClassLayout(std::unique_ptr<PDBSymbolTypeUDT> UDT);

private:
  std::unique_ptr<PDBSymbolTypeUDT> Type;
};

class BaseClassLayout : public UDTLayoutBase, public StorageItemBase {
public:
  BaseClassLayout(const UDTLayoutBase &Parent,
                  std::unique_ptr<PDBSymbolTypeBaseClass> Base);

private:
  std::unique_ptr<PDBSymbolTypeBaseClass> Base;
  bool IsVirtualBase;
};
}
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_UDTLAYOUT_H
