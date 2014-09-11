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

#include <memory>
#include <set>
#include <type_traits>

namespace PBQP {

template <typename CostT,
          typename CostKeyTComparator>
class CostPool {
public:
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

  typedef std::shared_ptr<CostT> PoolRef;

private:
  class EntryComparator {
  public:
    template <typename CostKeyT>
    typename std::enable_if<
               !std::is_same<PoolEntry*,
                             typename std::remove_const<CostKeyT>::type>::value,
               bool>::type
    operator()(const PoolEntry* a, const CostKeyT &b) {
      return compare(a->getCost(), b);
    }
    bool operator()(const PoolEntry* a, const PoolEntry* b) {
      return compare(a->getCost(), b->getCost());
    }
  private:
    CostKeyTComparator compare;
  };

  typedef std::set<PoolEntry*, EntryComparator> EntrySet;

  EntrySet entrySet;

  void removeEntry(PoolEntry *p) { entrySet.erase(p); }

public:
  template <typename CostKeyT> PoolRef getCost(CostKeyT costKey) {
    typename EntrySet::iterator itr =
      std::lower_bound(entrySet.begin(), entrySet.end(), costKey,
                       EntryComparator());

    if (itr != entrySet.end() && costKey == (*itr)->getCost())
      return PoolRef((*itr)->shared_from_this(), &(*itr)->getCost());

    auto p = std::make_shared<PoolEntry>(*this, std::move(costKey));
    entrySet.insert(itr, p.get());
    return PoolRef(std::move(p), &p->getCost());
  }
};

template <typename VectorT, typename VectorTComparator,
          typename MatrixT, typename MatrixTComparator>
class PoolCostAllocator {
private:
  typedef CostPool<VectorT, VectorTComparator> VectorCostPool;
  typedef CostPool<MatrixT, MatrixTComparator> MatrixCostPool;
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

}

#endif
