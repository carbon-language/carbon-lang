//===---------- CostAllocator.h - PBQP Cost Allocator -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines classes conforming to the PBQP cost value manager concept.
//
// Cost value managers are memory managers for PBQP cost values (vectors and
// matrices). Since PBQP graphs can grow very large (E.g. hundreds of thousands
// of edges on the largest function in SPEC2006).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_COSTALLOCATOR_H
#define LLVM_CODEGEN_PBQP_COSTALLOCATOR_H

#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <type_traits>

namespace llvm {
namespace PBQP {

template <typename CostT>
class CostPool {
public:
  typedef std::shared_ptr<CostT> PoolRef;

private:

  class PoolEntry : public std::enable_shared_from_this<PoolEntry> {
  public:
    template <typename CostKeyT>
    PoolEntry(CostPool &pool, CostKeyT cost)
        : pool(pool), cost(std::move(cost)) {}
    ~PoolEntry() { pool.removeEntry(this); }
    CostT& getCost() { return cost; }
    const CostT& getCost() const { return cost; }
  private:
    CostPool &pool;
    CostT cost;
  };

  class PoolEntryDSInfo {
  public:
    static inline PoolEntry* getEmptyKey() { return nullptr; }

    static inline PoolEntry* getTombstoneKey() {
      return reinterpret_cast<PoolEntry*>(static_cast<uintptr_t>(1));
    }

    template <typename CostKeyT>
    static unsigned getHashValue(const CostKeyT &C) {
      return hash_value(C);
    }

    static unsigned getHashValue(PoolEntry *P) {
      return getHashValue(P->getCost());
    }

    static unsigned getHashValue(const PoolEntry *P) {
      return getHashValue(P->getCost());
    }

    template <typename CostKeyT1, typename CostKeyT2>
    static
    bool isEqual(const CostKeyT1 &C1, const CostKeyT2 &C2) {
      return C1 == C2;
    }

    template <typename CostKeyT>
    static bool isEqual(const CostKeyT &C, PoolEntry *P) {
      if (P == getEmptyKey() || P == getTombstoneKey())
        return false;
      return isEqual(C, P->getCost());
    }

    static bool isEqual(PoolEntry *P1, PoolEntry *P2) {
      if (P1 == getEmptyKey() || P1 == getTombstoneKey())
        return P1 == P2;
      return isEqual(P1->getCost(), P2);
    }

  };

  typedef DenseSet<PoolEntry*, PoolEntryDSInfo> EntrySet;

  EntrySet entrySet;

  void removeEntry(PoolEntry *p) { entrySet.erase(p); }

public:
  template <typename CostKeyT> PoolRef getCost(CostKeyT costKey) {
    typename EntrySet::iterator itr = entrySet.find_as(costKey);

    if (itr != entrySet.end())
      return PoolRef((*itr)->shared_from_this(), &(*itr)->getCost());

    auto p = std::make_shared<PoolEntry>(*this, std::move(costKey));
    entrySet.insert(p.get());
    return PoolRef(std::move(p), &p->getCost());
  }
};

template <typename VectorT, typename MatrixT>
class PoolCostAllocator {
private:
  typedef CostPool<VectorT> VectorCostPool;
  typedef CostPool<MatrixT> MatrixCostPool;
public:
  typedef VectorT Vector;
  typedef MatrixT Matrix;
  typedef typename VectorCostPool::PoolRef VectorPtr;
  typedef typename MatrixCostPool::PoolRef MatrixPtr;

  template <typename VectorKeyT>
  VectorPtr getVector(VectorKeyT v) { return vectorPool.getCost(std::move(v)); }

  template <typename MatrixKeyT>
  MatrixPtr getMatrix(MatrixKeyT m) { return matrixPool.getCost(std::move(m)); }
private:
  VectorCostPool vectorPool;
  MatrixCostPool matrixPool;
};

} // namespace PBQP
} // namespace llvm

#endif
