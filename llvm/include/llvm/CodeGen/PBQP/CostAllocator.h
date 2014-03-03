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

#ifndef LLVM_COSTALLOCATOR_H
#define LLVM_COSTALLOCATOR_H

#include <set>
#include <type_traits>

namespace PBQP {

template <typename CostT,
          typename CostKeyTComparator>
class CostPool {
public:

  class PoolEntry {
  public:
    template <typename CostKeyT>
    PoolEntry(CostPool &pool, CostKeyT cost)
      : pool(pool), cost(std::move(cost)), refCount(0) {}
    ~PoolEntry() { pool.removeEntry(this); }
    void incRef() { ++refCount; }
    bool decRef() { --refCount; return (refCount == 0); }
    CostT& getCost() { return cost; }
    const CostT& getCost() const { return cost; }
  private:
    CostPool &pool;
    CostT cost;
    std::size_t refCount;
  };

  class PoolRef {
  public:
    PoolRef(PoolEntry *entry) : entry(entry) {
      this->entry->incRef();
    }
    PoolRef(const PoolRef &r) {
      entry = r.entry;
      entry->incRef();
    }
    PoolRef& operator=(const PoolRef &r) {
      assert(entry != 0 && "entry should not be null.");
      PoolEntry *temp = r.entry;
      temp->incRef();
      entry->decRef();
      entry = temp;
      return *this;
    }

    ~PoolRef() {
      if (entry->decRef())
        delete entry;
    }
    void reset(PoolEntry *entry) {
      entry->incRef();
      this->entry->decRef();
      this->entry = entry;
    }
    CostT& operator*() { return entry->getCost(); }
    const CostT& operator*() const { return entry->getCost(); }
    CostT* operator->() { return &entry->getCost(); }
    const CostT* operator->() const { return &entry->getCost(); }
  private:
    PoolEntry *entry;
  };

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

  template <typename CostKeyT>
  PoolRef getCost(CostKeyT costKey) {
    typename EntrySet::iterator itr =
      std::lower_bound(entrySet.begin(), entrySet.end(), costKey,
                       EntryComparator());

    if (itr != entrySet.end() && costKey == (*itr)->getCost())
      return PoolRef(*itr);

    PoolEntry *p = new PoolEntry(*this, std::move(costKey));
    entrySet.insert(itr, p);
    return PoolRef(p);
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

#endif // LLVM_COSTALLOCATOR_H
