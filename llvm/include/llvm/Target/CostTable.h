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

namespace llvm {

/// Cost Table Entry
template <class TypeTy>
struct CostTblEntry {
  int ISD;
  TypeTy Type;
  unsigned Cost;
};

/// Find in cost table, TypeTy must be comparable by ==
template <class TypeTy>
int CostTableLookup(const CostTblEntry<TypeTy> *Tbl,
                    unsigned len, int ISD, TypeTy Ty) {
  for (unsigned int i = 0; i < len; ++i)
    if (Tbl[i].ISD == ISD && Tbl[i].Type == Ty)
      return i;

  // Could not find an entry.
  return -1;
}

/// Type Conversion Cost Table
template <class TypeTy>
struct TypeConversionCostTblEntry {
  int ISD;
  TypeTy Dst;
  TypeTy Src;
  unsigned Cost;
};

/// Find in type conversion cost table, TypeTy must be comparable by ==
template <class TypeTy>
int ConvertCostTableLookup(const TypeConversionCostTblEntry<TypeTy> *Tbl,
                           unsigned len, int ISD, TypeTy Dst, TypeTy Src) {
  for (unsigned int i = 0; i < len; ++i)
    if (Tbl[i].ISD == ISD && Tbl[i].Src == Src && Tbl[i].Dst == Dst)
      return i;

  // Could not find an entry.
  return -1;
}

} // namespace llvm


#endif /* LLVM_TARGET_COSTTABLE_H_ */
