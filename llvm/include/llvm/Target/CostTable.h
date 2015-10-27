//===-- CostTable.h - Instruction Cost Table handling -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Cost tables and simple lookup functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_COSTTABLE_H_
#define LLVM_TARGET_COSTTABLE_H_

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

/// Cost Table Entry
template <class TypeTy>
struct CostTblEntry {
  int ISD;
  TypeTy Type;
  unsigned Cost;
};

/// Find in cost table, TypeTy must be comparable to CompareTy by ==
template <class TypeTy, class CompareTy>
const CostTblEntry<TypeTy> *CostTableLookup(ArrayRef<CostTblEntry<TypeTy>> Tbl,
                                            int ISD, CompareTy Ty) {
  auto I = std::find_if(Tbl.begin(), Tbl.end(),
                        [=](const CostTblEntry<TypeTy> &Entry) {
                          return ISD == Entry.ISD && Ty == Entry.Type; });
  if (I != Tbl.end())
    return I;

  // Could not find an entry.
  return nullptr;
}

/// Find in cost table, TypeTy must be comparable to CompareTy by ==
template <class TypeTy, class CompareTy, unsigned N>
const CostTblEntry<TypeTy> *CostTableLookup(const CostTblEntry<TypeTy>(&Tbl)[N],
                                            int ISD, CompareTy Ty) {
  return CostTableLookup(makeArrayRef(Tbl), ISD, Ty);
}

/// Type Conversion Cost Table
template <class TypeTy>
struct TypeConversionCostTblEntry {
  int ISD;
  TypeTy Dst;
  TypeTy Src;
  unsigned Cost;
};

/// Find in type conversion cost table, TypeTy must be comparable to CompareTy
/// by ==
template <class TypeTy, class CompareTy>
const TypeConversionCostTblEntry<TypeTy> *
ConvertCostTableLookup(ArrayRef<TypeConversionCostTblEntry<TypeTy>> Tbl,
                       int ISD, CompareTy Dst, CompareTy Src) {
  auto I = std::find_if(Tbl.begin(), Tbl.end(),
                        [=](const TypeConversionCostTblEntry<TypeTy> &Entry) {
                          return ISD == Entry.ISD && Src == Entry.Src &&
                                 Dst == Entry.Dst;
                        });
  if (I != Tbl.end())
    return I;

  // Could not find an entry.
  return nullptr;
}

/// Find in type conversion cost table, TypeTy must be comparable to CompareTy
/// by ==
template <class TypeTy, class CompareTy, unsigned N>
const TypeConversionCostTblEntry<TypeTy> *
ConvertCostTableLookup(const TypeConversionCostTblEntry<TypeTy>(&Tbl)[N],
                       int ISD, CompareTy Dst, CompareTy Src) {
  return ConvertCostTableLookup(makeArrayRef(Tbl), ISD, Dst, Src);
}

} // namespace llvm


#endif /* LLVM_TARGET_COSTTABLE_H_ */
