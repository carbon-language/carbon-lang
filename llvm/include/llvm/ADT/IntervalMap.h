//===- llvm/ADT/IntervalMap.h - A sorted interval map -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a coalescing interval map for small objects.
//
// KeyT objects are mapped to ValT objects. Intervals of keys that map to the
// same value are represented in a compressed form.
//
// Iterators provide ordered access to the compressed intervals rather than the
// individual keys, and insert and erase operations use key intervals as well.
//
// Like SmallVector, IntervalMap will store the first N intervals in the map
// object itself without any allocations. When space is exhausted it switches to
// a B+-tree representation with very small overhead for small key and value
// objects.
//
// A Traits class specifies how keys are compared. It also allows IntervalMap to
// work with both closed and half-open intervals.
//
// Keys and values are not stored next to each other in a std::pair, so we don't
// provide such a value_type. Dereferencing iterators only returns the mapped
// value. The interval bounds are accessible through the start() and stop()
// iterator methods.
//
// IntervalMap is optimized for small key and value objects, 4 or 8 bytes each
// is the optimal size. For large objects use std::map instead.
//
//===----------------------------------------------------------------------===//
//
// Synopsis:
//
// template <typename KeyT, typename ValT, unsigned N, typename Traits>
// class IntervalMap {
// public:
//   typedef KeyT key_type;
//   typedef ValT mapped_type;
//   typedef RecyclingAllocator<...> Allocator;
//   class iterator;
//   class const_iterator;
//
//   explicit IntervalMap(Allocator&);
//   ~IntervalMap():
//
//   bool empty() const;
//   KeyT start() const;
//   KeyT stop() const;
//   ValT lookup(KeyT x, Value NotFound = Value()) const;
//
//   const_iterator begin() const;
//   const_iterator end() const;
//   iterator begin();
//   iterator end();
//   const_iterator find(KeyT x) const;
//   iterator find(KeyT x);
//
//   void insert(KeyT a, KeyT b, ValT y);
//   void clear();
// };
//
// template <typename KeyT, typename ValT, unsigned N, typename Traits>
// class IntervalMap::const_iterator :
//   public std::iterator<std::bidirectional_iterator_tag, ValT> {
// public:
//   bool operator==(const const_iterator &) const;
//   bool operator!=(const const_iterator &) const;
//   bool valid() const;
//
//   const KeyT &start() const;
//   const KeyT &stop() const;
//   const ValT &value() const;
//   const ValT &operator*() const;
//   const ValT *operator->() const;
//
//   const_iterator &operator++();
//   const_iterator &operator++(int);
//   const_iterator &operator--();
//   const_iterator &operator--(int);
//   void goToBegin();
//   void goToEnd();
//   void find(KeyT x);
//   void advanceTo(KeyT x);
// };
//
// template <typename KeyT, typename ValT, unsigned N, typename Traits>
// class IntervalMap::iterator : public const_iterator {
// public:
//   void insert(KeyT a, KeyT b, Value y);
//   void erase();
// };
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INTERVALMAP_H
#define LLVM_ADT_INTERVALMAP_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <limits>
#include <iterator>

// FIXME: Remove debugging code
#ifndef NDEBUG
#include "llvm/Support/raw_ostream.h"
#endif

namespace llvm {


//===----------------------------------------------------------------------===//
//---                              Key traits                              ---//
//===----------------------------------------------------------------------===//
//
// The IntervalMap works with closed or half-open intervals.
// Adjacent intervals that map to the same value are coalesced.
//
// The IntervalMapInfo traits class is used to determine if a key is contained
// in an interval, and if two intervals are adjacent so they can be coalesced.
// The provided implementation works for closed integer intervals, other keys
// probably need a specialized version.
//
// The point x is contained in [a;b] when !startLess(x, a) && !stopLess(b, x).
//
// It is assumed that (a;b] half-open intervals are not used, only [a;b) is
// allowed. This is so that stopLess(a, b) can be used to determine if two
// intervals overlap.
//
//===----------------------------------------------------------------------===//

template <typename T>
struct IntervalMapInfo {

  /// startLess - Return true if x is not in [a;b].
  /// This is x < a both for closed intervals and for [a;b) half-open intervals.
  static inline bool startLess(const T &x, const T &a) {
    return x < a;
  }

  /// stopLess - Return true if x is not in [a;b].
  /// This is b < x for a closed interval, b <= x for [a;b) half-open intervals.
  static inline bool stopLess(const T &b, const T &x) {
    return b < x;
  }

  /// adjacent - Return true when the intervals [x;a] and [b;y] can coalesce.
  /// This is a+1 == b for closed intervals, a == b for half-open intervals.
  static inline bool adjacent(const T &a, const T &b) {
    return a+1 == b;
  }

};

/// IntervalMapImpl - Namespace used for IntervalMap implementation details.
/// It should be considered private to the implementation.
namespace IntervalMapImpl {

// Forward declarations.
template <typename, typename, unsigned, typename> class LeafNode;
template <typename, typename, unsigned, typename> class BranchNode;

typedef std::pair<unsigned,unsigned> IdxPair;


//===----------------------------------------------------------------------===//
//---                            Node Storage                              ---//
//===----------------------------------------------------------------------===//
//
// Both leaf and branch nodes store vectors of (key,value) pairs.
// Leaves store ((KeyT, KeyT), ValT) pairs, branches use (KeyT, NodeRef).
//
// Keys and values are stored in separate arrays to avoid padding caused by
// different object alignments. This also helps improve locality of reference
// when searching the keys.
//
// The nodes don't know how many elements they contain - that information is
// stored elsewhere. Omitting the size field prevents padding and allows a node
// to fill the allocated cache lines completely.
//
// These are typical key and value sizes, the node branching factor (N), and
// wasted space when nodes are sized to fit in three cache lines (192 bytes):
//
//   KT  VT   N Waste  Used by
//    4   4  24   0    Branch<4> (32-bit pointers)
//    4   8  16   0    Branch<4>
//    8   4  16   0    Leaf<4,4>
//    8   8  12   0    Leaf<4,8>, Branch<8>
//   16   4   9  12    Leaf<8,4>
//   16   8   8   0    Leaf<8,8>
//
//===----------------------------------------------------------------------===//

template <typename KT, typename VT, unsigned N>
class NodeBase {
public:
  enum { Capacity = N };

  KT key[N];
  VT val[N];

  /// copy - Copy elements from another node.
  /// @param other Node elements are copied from.
  /// @param i     Beginning of the source range in other.
  /// @param j     Beginning of the destination range in this.
  /// @param count Number of elements to copy.
  template <unsigned M>
  void copy(const NodeBase<KT, VT, M> &Other, unsigned i,
            unsigned j, unsigned Count) {
    assert(i + Count <= M && "Invalid source range");
    assert(j + Count <= N && "Invalid dest range");
    std::copy(Other.key + i, Other.key + i + Count, key + j);
    std::copy(Other.val + i, Other.val + i + Count, val + j);
  }

  /// lmove - Move elements to the left.
  /// @param i     Beginning of the source range.
  /// @param j     Beginning of the destination range.
  /// @param count Number of elements to copy.
  void lmove(unsigned i, unsigned j, unsigned Count) {
    assert(j <= i && "Use rmove shift elements right");
    copy(*this, i, j, Count);
  }

  /// rmove - Move elements to the right.
  /// @param i     Beginning of the source range.
  /// @param j     Beginning of the destination range.
  /// @param count Number of elements to copy.
  void rmove(unsigned i, unsigned j, unsigned Count) {
    assert(i <= j && "Use lmove shift elements left");
    assert(j + Count <= N && "Invalid range");
    std::copy_backward(key + i, key + i + Count, key + j + Count);
    std::copy_backward(val + i, val + i + Count, val + j + Count);
  }

  /// erase - Erase elements [i;j).
  /// @param i    Beginning of the range to erase.
  /// @param j    End of the range. (Exclusive).
  /// @param size Number of elements in node.
  void erase(unsigned i, unsigned j, unsigned Size) {
    lmove(j, i, Size - j);
  }

  /// shift - Shift elements [i;size) 1 position to the right.
  /// @param i    Beginning of the range to move.
  /// @param size Number of elements in node.
  void shift(unsigned i, unsigned Size) {
    rmove(i, i + 1, Size - i);
  }

  /// xferLeft - Transfer elements to a left sibling node.
  /// @param size  Number of elements in this.
  /// @param sib   Left sibling node.
  /// @param ssize Number of elements in sib.
  /// @param count Number of elements to transfer.
  void xferLeft(unsigned Size, NodeBase &Sib, unsigned SSize, unsigned Count) {
    Sib.copy(*this, 0, SSize, Count);
    erase(0, Count, Size);
  }

  /// xferRight - Transfer elements to a right sibling node.
  /// @param size  Number of elements in this.
  /// @param sib   Right sibling node.
  /// @param ssize Number of elements in sib.
  /// @param count Number of elements to transfer.
  void xferRight(unsigned Size, NodeBase &Sib, unsigned SSize, unsigned Count) {
    Sib.rmove(0, Count, SSize);
    Sib.copy(*this, Size-Count, 0, Count);
  }

  /// adjLeftSib - Adjust the number if elements in this node by moving
  /// elements to or from a left sibling node.
  /// @param size  Number of elements in this.
  /// @param sib   Right sibling node.
  /// @param ssize Number of elements in sib.
  /// @param add   The number of elements to add to this node, possibly < 0.
  /// @return      Number of elements added to this node, possibly negative.
  int adjLeftSib(unsigned Size, NodeBase &Sib, unsigned SSize, int Add) {
    if (Add > 0) {
      // We want to grow, copy from sib.
      unsigned Count = std::min(std::min(unsigned(Add), SSize), N - Size);
      Sib.xferRight(SSize, *this, Size, Count);
      return Count;
    } else {
      // We want to shrink, copy to sib.
      unsigned Count = std::min(std::min(unsigned(-Add), Size), N - SSize);
      xferLeft(Size, Sib, SSize, Count);
      return -Count;
    }
  }
};


//===----------------------------------------------------------------------===//
//---                             NodeSizer                                ---//
//===----------------------------------------------------------------------===//
//
// Compute node sizes from key and value types.
//
// The branching factors are chosen to make nodes fit in three cache lines.
// This may not be possible if keys or values are very large. Such large objects
// are handled correctly, but a std::map would probably give better performance.
//
//===----------------------------------------------------------------------===//

enum {
  // Cache line size. Most architectures have 32 or 64 byte cache lines.
  // We use 64 bytes here because it provides good branching factors.
  Log2CacheLine = 6,
  CacheLineBytes = 1 << Log2CacheLine,
  DesiredNodeBytes = 3 * CacheLineBytes
};

template <typename KeyT, typename ValT>
struct NodeSizer {
  enum {
    // Compute the leaf node branching factor that makes a node fit in three
    // cache lines. The branching factor must be at least 3, or some B+-tree
    // balancing algorithms won't work.
    // LeafSize can't be larger than CacheLineBytes. This is required by the
    // PointerIntPair used by NodeRef.
    DesiredLeafSize = DesiredNodeBytes /
      static_cast<unsigned>(2*sizeof(KeyT)+sizeof(ValT)),
    MinLeafSize = 3,
    LeafSize = DesiredLeafSize > MinLeafSize ? DesiredLeafSize : MinLeafSize
  };

  typedef NodeBase<std::pair<KeyT, KeyT>, ValT, LeafSize> LeafBase;

  enum {
    // Now that we have the leaf branching factor, compute the actual allocation
    // unit size by rounding up to a whole number of cache lines.
    AllocBytes = (sizeof(LeafBase) + CacheLineBytes-1) & ~(CacheLineBytes-1),

    // Determine the branching factor for branch nodes.
    BranchSize = AllocBytes /
      static_cast<unsigned>(sizeof(KeyT) + sizeof(void*))
  };

  /// Allocator - The recycling allocator used for both branch and leaf nodes.
  /// This typedef is very likely to be identical for all IntervalMaps with
  /// reasonably sized entries, so the same allocator can be shared among
  /// different kinds of maps.
  typedef RecyclingAllocator<BumpPtrAllocator, char,
                             AllocBytes, CacheLineBytes> Allocator;

};


//===----------------------------------------------------------------------===//
//---                              NodeRef                                 ---//
//===----------------------------------------------------------------------===//
//
// B+-tree nodes can be leaves or branches, so we need a polymorphic node
// pointer that can point to both kinds.
//
// All nodes are cache line aligned and the low 6 bits of a node pointer are
// always 0. These bits are used to store the number of elements in the
// referenced node. Besides saving space, placing node sizes in the parents
// allow tree balancing algorithms to run without faulting cache lines for nodes
// that may not need to be modified.
//
// A NodeRef doesn't know whether it references a leaf node or a branch node.
// It is the responsibility of the caller to use the correct types.
//
// Nodes are never supposed to be empty, and it is invalid to store a node size
// of 0 in a NodeRef. The valid range of sizes is 1-64.
//
//===----------------------------------------------------------------------===//

struct CacheAlignedPointerTraits {
  static inline void *getAsVoidPointer(void *P) { return P; }
  static inline void *getFromVoidPointer(void *P) { return P; }
  enum { NumLowBitsAvailable = Log2CacheLine };
};

template <typename KeyT, typename ValT, typename Traits>
class NodeRef {
public:
  typedef LeafNode<KeyT, ValT, NodeSizer<KeyT, ValT>::LeafSize, Traits> Leaf;
  typedef BranchNode<KeyT, ValT, NodeSizer<KeyT, ValT>::BranchSize,
                     Traits> Branch;

private:
  PointerIntPair<void*, Log2CacheLine, unsigned, CacheAlignedPointerTraits> pip;

public:
  /// NodeRef - Create a null ref.
  NodeRef() {}

  /// operator bool - Detect a null ref.
  operator bool() const { return pip.getOpaqueValue(); }

  /// NodeRef - Create a reference to the leaf node p with n elements.
  NodeRef(Leaf *p, unsigned n) : pip(p, n - 1) {}

  /// NodeRef - Create a reference to the branch node p with n elements.
  NodeRef(Branch *p, unsigned n) : pip(p, n - 1) {}

  /// size - Return the number of elements in the referenced node.
  unsigned size() const { return pip.getInt() + 1; }

  /// setSize - Update the node size.
  void setSize(unsigned n) { pip.setInt(n - 1); }

  /// leaf - Return the referenced leaf node.
  /// Note there are no dynamic type checks.
  Leaf &leaf() const {
    return *reinterpret_cast<Leaf*>(pip.getPointer());
  }

  /// branch - Return the referenced branch node.
  /// Note there are no dynamic type checks.
  Branch &branch() const {
    return *reinterpret_cast<Branch*>(pip.getPointer());
  }

  bool operator==(const NodeRef &RHS) const {
    if (pip == RHS.pip)
      return true;
    assert(pip.getPointer() != RHS.pip.getPointer() && "Inconsistent NodeRefs");
    return false;
  }

  bool operator!=(const NodeRef &RHS) const {
    return !operator==(RHS);
  }
};

//===----------------------------------------------------------------------===//
//---                            Leaf nodes                                ---//
//===----------------------------------------------------------------------===//
//
// Leaf nodes store up to N disjoint intervals with corresponding values.
//
// The intervals are kept sorted and fully coalesced so there are no adjacent
// intervals mapping to the same value.
//
// These constraints are always satisfied:
//
// - Traits::stopLess(key[i].start, key[i].stop) - Non-empty, sane intervals.
//
// - Traits::stopLess(key[i].stop, key[i + 1].start) - Sorted.
//
// - val[i] != val[i + 1] ||
//     !Traits::adjacent(key[i].stop, key[i + 1].start) - Fully coalesced.
//
//===----------------------------------------------------------------------===//

template <typename KeyT, typename ValT, unsigned N, typename Traits>
class LeafNode : public NodeBase<std::pair<KeyT, KeyT>, ValT, N> {
public:
  const KeyT &start(unsigned i) const { return this->key[i].first; }
  const KeyT &stop(unsigned i) const { return this->key[i].second; }
  const ValT &value(unsigned i) const { return this->val[i]; }

  KeyT &start(unsigned i) { return this->key[i].first; }
  KeyT &stop(unsigned i) { return this->key[i].second; }
  ValT &value(unsigned i) { return this->val[i]; }

  /// findFrom - Find the first interval after i that may contain x.
  /// @param i    Starting index for the search.
  /// @param size Number of elements in node.
  /// @param x    Key to search for.
  /// @return     First index with !stopLess(key[i].stop, x), or size.
  ///             This is the first interval that can possibly contain x.
  unsigned findFrom(unsigned i, unsigned Size, KeyT x) const {
    assert(i <= Size && Size <= N && "Bad indices");
    assert((i == 0 || Traits::stopLess(stop(i - 1), x)) &&
           "Index is past the needed point");
    while (i != Size && Traits::stopLess(stop(i), x)) ++i;
    return i;
  }

  /// safeFind - Find an interval that is known to exist. This is the same as
  /// findFrom except is it assumed that x is at least within range of the last
  /// interval.
  /// @param i Starting index for the search.
  /// @param x Key to search for.
  /// @return  First index with !stopLess(key[i].stop, x), never size.
  ///          This is the first interval that can possibly contain x.
  unsigned safeFind(unsigned i, KeyT x) const {
    assert(i < N && "Bad index");
    assert((i == 0 || Traits::stopLess(stop(i - 1), x)) &&
           "Index is past the needed point");
    while (Traits::stopLess(stop(i), x)) ++i;
    assert(i < N && "Unsafe intervals");
    return i;
  }

  /// safeLookup - Lookup mapped value for a safe key.
  /// It is assumed that x is within range of the last entry.
  /// @param x        Key to search for.
  /// @param NotFound Value to return if x is not in any interval.
  /// @return         The mapped value at x or NotFound.
  ValT safeLookup(KeyT x, ValT NotFound) const {
    unsigned i = safeFind(0, x);
    return Traits::startLess(x, start(i)) ? NotFound : value(i);
  }

  IdxPair insertFrom(unsigned i, unsigned Size, KeyT a, KeyT b, ValT y);
  unsigned extendStop(unsigned i, unsigned Size, KeyT b);

#ifndef NDEBUG
  void dump(unsigned Size) {
    errs() << "  N" << this << " [shape=record label=\"{ " << Size << '/' << N;
    for (unsigned i = 0; i != Size; ++i)
      errs() << " | {" << start(i) << '-' << stop(i) << "|" << value(i) << '}';
    errs() << "}\"];\n";
  }
#endif

};

/// insertFrom - Add mapping of [a;b] to y if possible, coalescing as much as
/// possible. This may cause the node to grow by 1, or it may cause the node
/// to shrink because of coalescing.
/// @param i    Starting index = insertFrom(0, size, a)
/// @param size Number of elements in node.
/// @param a    Interval start.
/// @param b    Interval stop.
/// @param y    Value be mapped.
/// @return     (insert position, new size), or (i, Capacity+1) on overflow.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
IdxPair LeafNode<KeyT, ValT, N, Traits>::
insertFrom(unsigned i, unsigned Size, KeyT a, KeyT b, ValT y) {
  assert(i <= Size && Size <= N && "Invalid index");
  assert(!Traits::stopLess(b, a) && "Invalid interval");

  // Verify the findFrom invariant.
  assert((i == 0 || Traits::stopLess(stop(i - 1), a)));
  assert((i == Size || !Traits::stopLess(stop(i), a)));

  // Coalesce with previous interval.
  if (i && value(i - 1) == y && Traits::adjacent(stop(i - 1), a))
    return IdxPair(i - 1, extendStop(i - 1, Size, b));

  // Detect overflow.
  if (i == N)
    return IdxPair(i, N + 1);

  // Add new interval at end.
  if (i == Size) {
    start(i) = a;
    stop(i) = b;
    value(i) = y;
    return IdxPair(i, Size + 1);
  }

  // Overlapping intervals?
  if (!Traits::stopLess(b, start(i))) {
    assert(value(i) == y && "Inconsistent values in overlapping intervals");
    if (Traits::startLess(a, start(i)))
      start(i) = a;
    return IdxPair(i, extendStop(i, Size, b));
  }

  // Try to coalesce with following interval.
  if (value(i) == y && Traits::adjacent(b, start(i))) {
    start(i) = a;
    return IdxPair(i, Size);
  }

  // We must insert before i. Detect overflow.
  if (Size == N)
    return IdxPair(i, N + 1);

  // Insert before i.
  this->shift(i, Size);
  start(i) = a;
  stop(i) = b;
  value(i) = y;
  return IdxPair(i, Size + 1);
}

/// extendStop - Extend stop(i) to b, coalescing with following intervals.
/// @param i    Interval to extend.
/// @param size Number of elements in node.
/// @param b    New interval end point.
/// @return     New node size after coalescing.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
unsigned LeafNode<KeyT, ValT, N, Traits>::
extendStop(unsigned i, unsigned Size, KeyT b) {
  assert(i < Size && Size <= N && "Bad indices");

  // Are we even extending the interval?
  if (Traits::startLess(b, stop(i)))
    return Size;

  // Find the first interval that may be preserved.
  unsigned j = findFrom(i + 1, Size, b);
  if (j < Size) {
    // Would key[i] overlap key[j] after the extension?
    if (Traits::stopLess(b, start(j))) {
      // Not overlapping. Perhaps adjacent and coalescable?
      if (value(i) == value(j) && Traits::adjacent(b, start(j)))
        b = stop(j++);
    } else {
      // Overlap. Include key[j] in the new interval.
      assert(value(i) == value(j) && "Overlapping values");
      b = stop(j++);
    }
  }
  stop(i) =  b;

  // Entries [i+1;j) were coalesced.
  if (i + 1 < j && j < Size)
    this->erase(i + 1, j, Size);
  return Size - (j - (i + 1));
}


//===----------------------------------------------------------------------===//
//---                             Branch nodes                             ---//
//===----------------------------------------------------------------------===//
//
// A branch node stores references to 1--N subtrees all of the same height.
//
// The key array in a branch node holds the rightmost stop key of each subtree.
// It is redundant to store the last stop key since it can be found in the
// parent node, but doing so makes tree balancing a lot simpler.
//
// It is unusual for a branch node to only have one subtree, but it can happen
// in the root node if it is smaller than the normal nodes.
//
// When all of the leaf nodes from all the subtrees are concatenated, they must
// satisfy the same constraints as a single leaf node. They must be sorted,
// sane, and fully coalesced.
//
//===----------------------------------------------------------------------===//

template <typename KeyT, typename ValT, unsigned N, typename Traits>
class BranchNode : public NodeBase<KeyT, NodeRef<KeyT, ValT, Traits>, N> {
  typedef  NodeRef<KeyT, ValT, Traits> NodeRefT;
public:
  const KeyT &stop(unsigned i) const { return this->key[i]; }
  const NodeRefT &subtree(unsigned i) const { return this->val[i]; }

  KeyT &stop(unsigned i) { return this->key[i]; }
  NodeRefT &subtree(unsigned i) { return this->val[i]; }

  /// findFrom - Find the first subtree after i that may contain x.
  /// @param i    Starting index for the search.
  /// @param size Number of elements in node.
  /// @param x    Key to search for.
  /// @return     First index with !stopLess(key[i], x), or size.
  ///             This is the first subtree that can possibly contain x.
  unsigned findFrom(unsigned i, unsigned Size, KeyT x) const {
    assert(i <= Size && Size <= N && "Bad indices");
    assert((i == 0 || Traits::stopLess(stop(i - 1), x)) &&
           "Index to findFrom is past the needed point");
    while (i != Size && Traits::stopLess(stop(i), x)) ++i;
    return i;
  }

  /// safeFind - Find a subtree that is known to exist. This is the same as
  /// findFrom except is it assumed that x is in range.
  /// @param i Starting index for the search.
  /// @param x Key to search for.
  /// @return  First index with !stopLess(key[i], x), never size.
  ///          This is the first subtree that can possibly contain x.
  unsigned safeFind(unsigned i, KeyT x) const {
    assert(i < N && "Bad index");
    assert((i == 0 || Traits::stopLess(stop(i - 1), x)) &&
           "Index is past the needed point");
    while (Traits::stopLess(stop(i), x)) ++i;
    assert(i < N && "Unsafe intervals");
    return i;
  }

  /// safeLookup - Get the subtree containing x, Assuming that x is in range.
  /// @param x Key to search for.
  /// @return  Subtree containing x
  NodeRefT safeLookup(KeyT x) const {
    return subtree(safeFind(0, x));
  }

  /// insert - Insert a new (subtree, stop) pair.
  /// @param i    Insert position, following entries will be shifted.
  /// @param size Number of elements in node.
  /// @param node Subtree to insert.
  /// @param stp  Last key in subtree.
  void insert(unsigned i, unsigned Size, NodeRefT Node, KeyT Stop) {
    assert(Size < N && "branch node overflow");
    assert(i <= Size && "Bad insert position");
    this->shift(i, Size);
    subtree(i) = Node;
    stop(i) = Stop;
  }

#ifndef NDEBUG
  void dump(unsigned Size) {
    errs() << "  N" << this << " [shape=record label=\"" << Size << '/' << N;
    for (unsigned i = 0; i != Size; ++i)
      errs() << " | <s" << i << "> " << stop(i);
    errs() << "\"];\n";
    for (unsigned i = 0; i != Size; ++i)
      errs() << "  N" << this << ":s" << i << " -> N"
             << &subtree(i).branch() << ";\n";
  }
#endif

};

} // namespace IntervalMapImpl


//===----------------------------------------------------------------------===//
//---                          IntervalMap                                ----//
//===----------------------------------------------------------------------===//

template <typename KeyT, typename ValT,
          unsigned N = IntervalMapImpl::NodeSizer<KeyT, ValT>::LeafSize,
          typename Traits = IntervalMapInfo<KeyT> >
class IntervalMap {
  typedef IntervalMapImpl::NodeRef<KeyT, ValT, Traits> NodeRef;
  typedef IntervalMapImpl::NodeSizer<KeyT, ValT> NodeSizer;
  typedef typename NodeRef::Leaf Leaf;
  typedef typename NodeRef::Branch Branch;
  typedef IntervalMapImpl::LeafNode<KeyT, ValT, N, Traits> RootLeaf;
  typedef IntervalMapImpl::IdxPair IdxPair;

  // The RootLeaf capacity is given as a template parameter. We must compute the
  // corresponding RootBranch capacity.
  enum {
    DesiredRootBranchCap = (sizeof(RootLeaf) - sizeof(KeyT)) /
      (sizeof(KeyT) + sizeof(NodeRef)),
    RootBranchCap = DesiredRootBranchCap ? DesiredRootBranchCap : 1
  };

  typedef IntervalMapImpl::BranchNode<KeyT, ValT, RootBranchCap, Traits> RootBranch;

  // When branched, we store a global start key as well as the branch node.
  struct RootBranchData {
    KeyT start;
    RootBranch node;
  };

  enum {
    RootDataSize = sizeof(RootBranchData) > sizeof(RootLeaf) ?
                   sizeof(RootBranchData) : sizeof(RootLeaf)
  };

public:
  typedef typename NodeSizer::Allocator Allocator;

private:
  // The root data is either a RootLeaf or a RootBranchData instance.
  // We can't put them in a union since C++03 doesn't allow non-trivial
  // constructors in unions.
  // Instead, we use a char array with pointer alignment. The alignment is
  // ensured by the allocator member in the class, but still verified in the
  // constructor. We don't support keys or values that are more aligned than a
  // pointer.
  char data[RootDataSize];

  // Tree height.
  // 0: Leaves in root.
  // 1: Root points to leaf.
  // 2: root->branch->leaf ...
  unsigned height;

  // Number of entries in the root node.
  unsigned rootSize;

  // Allocator used for creating external nodes.
  Allocator &allocator;

  /// dataAs - Represent data as a node type without breaking aliasing rules.
  template <typename T>
  T &dataAs() const {
    union {
      const char *d;
      T *t;
    } u;
    u.d = data;
    return *u.t;
  }

  const RootLeaf &rootLeaf() const {
    assert(!branched() && "Cannot acces leaf data in branched root");
    return dataAs<RootLeaf>();
  }
  RootLeaf &rootLeaf() {
    assert(!branched() && "Cannot acces leaf data in branched root");
    return dataAs<RootLeaf>();
  }
  RootBranchData &rootBranchData() const {
    assert(branched() && "Cannot access branch data in non-branched root");
    return dataAs<RootBranchData>();
  }
  RootBranchData &rootBranchData() {
    assert(branched() && "Cannot access branch data in non-branched root");
    return dataAs<RootBranchData>();
  }
  const RootBranch &rootBranch() const { return rootBranchData().node; }
  RootBranch &rootBranch()             { return rootBranchData().node; }
  KeyT rootBranchStart() const { return rootBranchData().start; }
  KeyT &rootBranchStart()      { return rootBranchData().start; }

  Leaf *allocLeaf()  {
    return new(allocator.template Allocate<Leaf>()) Leaf();
  }
  void freeLeaf(Leaf *P) {
    P->~Leaf();
    allocator.Deallocate(P);
  }

  Branch *allocBranch() {
    return new(allocator.template Allocate<Branch>()) Branch();
  }
  void freeBranch(Branch *P) {
    P->~Branch();
    allocator.Deallocate(P);
  }


  IdxPair branchRoot(unsigned Position);
  IdxPair splitRoot(unsigned Position);

  void switchRootToBranch() {
    rootLeaf().~RootLeaf();
    height = 1;
    new (&rootBranchData()) RootBranchData();
  }

  void switchRootToLeaf() {
    rootBranchData().~RootBranchData();
    height = 0;
    new(&rootLeaf()) RootLeaf();
  }

  bool branched() const { return height > 0; }

  ValT treeSafeLookup(KeyT x, ValT NotFound) const;

  void visitNodes(void (IntervalMap::*f)(NodeRef, unsigned Level));

public:
  explicit IntervalMap(Allocator &a) : height(0), rootSize(0), allocator(a) {
    assert((uintptr_t(data) & (alignOf<RootLeaf>() - 1)) == 0 &&
           "Insufficient alignment");
    new(&rootLeaf()) RootLeaf();
  }

  /// empty -  Return true when no intervals are mapped.
  bool empty() const {
    return rootSize == 0;
  }

  /// start - Return the smallest mapped key in a non-empty map.
  KeyT start() const {
    assert(!empty() && "Empty IntervalMap has no start");
    return !branched() ? rootLeaf().start(0) : rootBranchStart();
  }

  /// stop - Return the largest mapped key in a non-empty map.
  KeyT stop() const {
    assert(!empty() && "Empty IntervalMap has no stop");
    return !branched() ? rootLeaf().stop(rootSize - 1) :
                         rootBranch().stop(rootSize - 1);
  }

  /// lookup - Return the mapped value at x or NotFound.
  ValT lookup(KeyT x, ValT NotFound = ValT()) const {
    if (empty() || Traits::startLess(x, start()) || Traits::stopLess(stop(), x))
      return NotFound;
    return branched() ? treeSafeLookup(x, NotFound) :
                        rootLeaf().safeLookup(x, NotFound);
  }

  /// insert - Add a mapping of [a;b] to y, coalesce with adjacent intervals.
  /// It is assumed that no key in the interval is mapped to another value, but
  /// overlapping intervals already mapped to y will be coalesced.
  void insert(KeyT a, KeyT b, ValT y) {
    find(a).insert(a, b, y);
  }

  class const_iterator;
  class iterator;
  friend class const_iterator;
  friend class iterator;

  const_iterator begin() const {
    iterator I(*this);
    I.goToBegin();
    return I;
  }

  iterator begin() {
    iterator I(*this);
    I.goToBegin();
    return I;
  }

  const_iterator end() const {
    iterator I(*this);
    I.goToEnd();
    return I;
  }

  iterator end() {
    iterator I(*this);
    I.goToEnd();
    return I;
  }

  /// find - Return an iterator pointing to the first interval ending at or
  /// after x, or end().
  const_iterator find(KeyT x) const {
    iterator I(*this);
    I.find(x);
    return I;
  }

  iterator find(KeyT x) {
    iterator I(*this);
    I.find(x);
    return I;
  }

#ifndef NDEBUG
  void dump();
  void dumpNode(NodeRef Node, unsigned Height);
#endif
};

/// treeSafeLookup - Return the mapped value at x or NotFound, assuming a
/// branched root.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
ValT IntervalMap<KeyT, ValT, N, Traits>::
treeSafeLookup(KeyT x, ValT NotFound) const {
  assert(branched() && "treeLookup assumes a branched root");

  NodeRef NR = rootBranch().safeLookup(x);
  for (unsigned h = height-1; h; --h)
    NR = NR.branch().safeLookup(x);
  return NR.leaf().safeLookup(x, NotFound);
}


// branchRoot - Switch from a leaf root to a branched root.
// Return the new (root offset, node offset) corresponding to Position.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
IntervalMapImpl::IdxPair IntervalMap<KeyT, ValT, N, Traits>::
branchRoot(unsigned Position) {
  // How many external leaf nodes to hold RootLeaf+1?
  const unsigned Nodes = RootLeaf::Capacity / Leaf::Capacity + 1;

  // Compute element distribution among new nodes.
  unsigned size[Nodes];
  IdxPair NewOffset(0, Position);

  // Is is very common for the root node to be smaller than external nodes.
  if (Nodes == 1)
    size[0] = rootSize;
  else
    NewOffset = distribute(Nodes, rootSize, Leaf::Capacity,  NULL, size,
                           Position, true);

  // Allocate new nodes.
  unsigned pos = 0;
  NodeRef node[Nodes];
  for (unsigned n = 0; n != Nodes; ++n) {
    node[n] = NodeRef(allocLeaf(), size[n]);
    node[n].leaf().copy(rootLeaf(), pos, 0, size[n]);
    pos += size[n];
  }

  // Destroy the old leaf node, construct branch node instead.
  switchRootToBranch();
  for (unsigned n = 0; n != Nodes; ++n) {
    rootBranch().stop(n) = node[n].leaf().stop(size[n]-1);
    rootBranch().subtree(n) = node[n];
  }
  rootBranchStart() = node[0].leaf().start(0);
  rootSize = Nodes;
  return NewOffset;
}

// splitRoot - Split the current BranchRoot into multiple Branch nodes.
// Return the new (root offset, node offset) corresponding to Position.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
IntervalMapImpl::IdxPair IntervalMap<KeyT, ValT, N, Traits>::
splitRoot(unsigned Position) {
  // How many external leaf nodes to hold RootBranch+1?
  const unsigned Nodes = RootBranch::Capacity / Branch::Capacity + 1;

  // Compute element distribution among new nodes.
  unsigned Size[Nodes];
  IdxPair NewOffset(0, Position);

  // Is is very common for the root node to be smaller than external nodes.
  if (Nodes == 1)
    Size[0] = rootSize;
  else
    NewOffset = distribute(Nodes, rootSize, Leaf::Capacity,  NULL, Size,
                           Position, true);

  // Allocate new nodes.
  unsigned Pos = 0;
  NodeRef Node[Nodes];
  for (unsigned n = 0; n != Nodes; ++n) {
    Node[n] = NodeRef(allocBranch(), Size[n]);
    Node[n].branch().copy(rootBranch(), Pos, 0, Size[n]);
    Pos += Size[n];
  }

  for (unsigned n = 0; n != Nodes; ++n) {
    rootBranch().stop(n) = Node[n].branch().stop(Size[n]-1);
    rootBranch().subtree(n) = Node[n];
  }
  rootSize = Nodes;
  return NewOffset;
}

/// visitNodes - Visit each external node.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
visitNodes(void (IntervalMap::*f)(NodeRef, unsigned Height)) {
  if (!branched())
    return;
  SmallVector<NodeRef, 4> Refs, NextRefs;

  // Collect level 0 nodes from the root.
  for (unsigned i = 0; i != rootSize; ++i)
    Refs.push_back(rootBranch().subtree(i));

  // Visit all branch nodes.
  for (unsigned h = height - 1; h; --h) {
    for (unsigned i = 0, e = Refs.size(); i != e; ++i) {
      Branch &B = Refs[i].branch();
      for (unsigned j = 0, s = Refs[i].size(); j != s; ++j)
        NextRefs.push_back(B.subtree(j));
      (this->*f)(Refs[i], h);
    }
    Refs.clear();
    Refs.swap(NextRefs);
  }

  // Visit all leaf nodes.
  for (unsigned i = 0, e = Refs.size(); i != e; ++i)
    (this->*f)(Refs[i], 0);
}

#ifndef NDEBUG
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
dumpNode(NodeRef Node, unsigned Height) {
  if (Height)
    Node.branch().dump(Node.size());
  else
    Node.leaf().dump(Node.size());
}

template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
dump() {
  errs() << "digraph {\n";
  if (branched())
    rootBranch().dump(rootSize);
  else
    rootLeaf().dump(rootSize);
  visitNodes(&IntervalMap::dumpNode);
  errs() << "}\n";
}
#endif

//===----------------------------------------------------------------------===//
//---                             const_iterator                          ----//
//===----------------------------------------------------------------------===//

template <typename KeyT, typename ValT, unsigned N, typename Traits>
class IntervalMap<KeyT, ValT, N, Traits>::const_iterator :
  public std::iterator<std::bidirectional_iterator_tag, ValT> {
protected:
  friend class IntervalMap;
  typedef std::pair<NodeRef, unsigned> PathEntry;
  typedef SmallVector<PathEntry, 4> Path;

  // The map referred to.
  IntervalMap *map;

  // The offset into map's root node.
  unsigned rootOffset;

  // We store a full path from the root to the current position.
  //
  // When rootOffset == map->rootSize, we are at end() and path() is empty.
  // Otherwise, when branched these conditions hold:
  //
  // 1. path.front().first == rootBranch().subtree(rootOffset)
  // 2. path[i].first == path[i-1].first.branch().subtree(path[i-1].second)
  // 3. path.size() == map->height.
  //
  // Thus, path.back() always refers to the current leaf node unless the root is
  // unbranched.
  //
  // The path may be partially filled, but never between iterator calls.
  Path path;

  explicit const_iterator(IntervalMap &map)
    : map(&map), rootOffset(map.rootSize) {}

  bool branched() const {
    assert(map && "Invalid iterator");
    return map->branched();
  }

  NodeRef   pathNode(unsigned h)   const { return path[h].first; }
  NodeRef  &pathNode(unsigned h)         { return path[h].first; }
  unsigned  pathOffset(unsigned h) const { return path[h].second; }
  unsigned &pathOffset(unsigned h)       { return path[h].second; }

  Leaf &treeLeaf() const {
    assert(branched() && path.size() == map->height);
    return path.back().first.leaf();
  }
  unsigned treeLeafSize() const {
    assert(branched() && path.size() == map->height);
    return path.back().first.size();
  }
  unsigned &treeLeafOffset() {
    assert(branched() && path.size() == map->height);
    return path.back().second;
  }
  unsigned treeLeafOffset() const {
    assert(branched() && path.size() == map->height);
    return path.back().second;
  }

  // Get the next node ptr for an incomplete path.
  NodeRef pathNextDown() {
    assert(path.size() < map->height && "Path is already complete");

    if (path.empty())
      return map->rootBranch().subtree(rootOffset);
    else
      return path.back().first.branch().subtree(path.back().second);
  }

  void pathFillLeft();
  void pathFillFind(KeyT x);
  void pathFillRight();

  NodeRef leftSibling(unsigned level) const;
  NodeRef rightSibling(unsigned level) const;

  void treeIncrement();
  void treeDecrement();
  void treeFind(KeyT x);

public:
  /// valid - Return true if the current position is valid, false for end().
  bool valid() const {
    assert(map && "Invalid iterator");
    return rootOffset < map->rootSize;
  }

  /// start - Return the beginning of the current interval.
  const KeyT &start() const {
    assert(valid() && "Cannot access invalid iterator");
    return branched() ? treeLeaf().start(treeLeafOffset()) :
                        map->rootLeaf().start(rootOffset);
  }

  /// stop - Return the end of the current interval.
  const KeyT &stop() const {
    assert(valid() && "Cannot access invalid iterator");
    return branched() ? treeLeaf().stop(treeLeafOffset()) :
                        map->rootLeaf().stop(rootOffset);
  }

  /// value - Return the mapped value at the current interval.
  const ValT &value() const {
    assert(valid() && "Cannot access invalid iterator");
    return branched() ? treeLeaf().value(treeLeafOffset()) :
                        map->rootLeaf().value(rootOffset);
  }

  const ValT &operator*() const {
    return value();
  }

  bool operator==(const const_iterator &RHS) const {
    assert(map == RHS.map && "Cannot compare iterators from different maps");
    return rootOffset == RHS.rootOffset &&
             (!valid() || !branched() || path.back() == RHS.path.back());
  }

  bool operator!=(const const_iterator &RHS) const {
    return !operator==(RHS);
  }

  /// goToBegin - Move to the first interval in map.
  void goToBegin() {
    rootOffset = 0;
    path.clear();
    if (branched())
      pathFillLeft();
  }

  /// goToEnd - Move beyond the last interval in map.
  void goToEnd() {
    rootOffset = map->rootSize;
    path.clear();
  }

  /// preincrement - move to the next interval.
  const_iterator &operator++() {
    assert(valid() && "Cannot increment end()");
    if (!branched())
      ++rootOffset;
    else if (treeLeafOffset() != treeLeafSize() - 1)
      ++treeLeafOffset();
    else
      treeIncrement();
    return *this;
  }

  /// postincrement - Dont do that!
  const_iterator operator++(int) {
    const_iterator tmp = *this;
    operator++();
    return tmp;
  }

  /// predecrement - move to the previous interval.
  const_iterator &operator--() {
    if (!branched()) {
      assert(rootOffset && "Cannot decrement begin()");
      --rootOffset;
    } else if (treeLeafOffset())
      --treeLeafOffset();
    else
      treeDecrement();
    return *this;
  }

  /// postdecrement - Dont do that!
  const_iterator operator--(int) {
    const_iterator tmp = *this;
    operator--();
    return tmp;
  }

  /// find - Move to the first interval with stop >= x, or end().
  /// This is a full search from the root, the current position is ignored.
  void find(KeyT x) {
    if (branched())
      treeFind(x);
    else
      rootOffset = map->rootLeaf().findFrom(0, map->rootSize, x);
  }

  /// advanceTo - Move to the first interval with stop >= x, or end().
  /// The search is started from the current position, and no earlier positions
  /// can be found. This is much faster than find() for small moves.
  void advanceTo(KeyT x) {
    if (branched())
      treeAdvanceTo(x);
    else
      rootOffset = map->rootLeaf().findFrom(rootOffset, map->rootSize, x);
  }

};

// pathFillLeft - Complete path by following left-most branches.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::pathFillLeft() {
  NodeRef NR = pathNextDown();
  for (unsigned i = map->height - path.size() - 1; i; --i) {
    path.push_back(PathEntry(NR, 0));
    NR = NR.branch().subtree(0);
  }
  path.push_back(PathEntry(NR, 0));
}

// pathFillFind - Complete path by searching for x.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::pathFillFind(KeyT x) {
  NodeRef NR = pathNextDown();
  for (unsigned i = map->height - path.size() - 1; i; --i) {
    unsigned p = NR.branch().safeFind(0, x);
    path.push_back(PathEntry(NR, p));
    NR = NR.branch().subtree(p);
  }
  path.push_back(PathEntry(NR, NR.leaf().safeFind(0, x)));
}

// pathFillRight - Complete path by adding rightmost entries.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::pathFillRight() {
  NodeRef NR = pathNextDown();
  for (unsigned i = map->height - path.size() - 1; i; --i) {
    unsigned p = NR.size() - 1;
    path.push_back(PathEntry(NR, p));
    NR = NR.branch().subtree(p);
  }
  path.push_back(PathEntry(NR, NR.size() - 1));
}

/// leftSibling - find the left sibling node to path[level].
/// @param level 0 is just below the root, map->height - 1 for the leaves.
/// @return The left sibling NodeRef, or NULL.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
typename IntervalMap<KeyT, ValT, N, Traits>::NodeRef
IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::leftSibling(unsigned level) const {
  assert(branched() && "Not at a branched node");
  assert(level <= path.size() && "Bad level");

  // Go up the tree until we can go left.
  unsigned h = level;
  while (h && pathOffset(h - 1) == 0)
    --h;

  // We are at the first leaf node, no left sibling.
  if (!h && rootOffset == 0)
    return NodeRef();

  // NR is the subtree containing our left sibling.
  NodeRef NR = h ?
    pathNode(h - 1).branch().subtree(pathOffset(h - 1) - 1) :
    map->rootBranch().subtree(rootOffset - 1);

  // Keep right all the way down.
  for (; h != level; ++h)
    NR = NR.branch().subtree(NR.size() - 1);
  return NR;
}

/// rightSibling - find the right sibling node to path[level].
/// @param level 0 is just below the root, map->height - 1 for the leaves.
/// @return The right sibling NodeRef, or NULL.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
typename IntervalMap<KeyT, ValT, N, Traits>::NodeRef
IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::rightSibling(unsigned level) const {
  assert(branched() && "Not at a branched node");
  assert(level <= this->path.size() && "Bad level");

  // Go up the tree until we can go right.
  unsigned h = level;
  while (h && pathOffset(h - 1) == pathNode(h - 1).size() - 1)
    --h;

  // We are at the last leaf node, no right sibling.
  if (!h && rootOffset == map->rootSize - 1)
    return NodeRef();

  // NR is the subtree containing our right sibling.
  NodeRef NR = h ?
    pathNode(h - 1).branch().subtree(pathOffset(h - 1) + 1) :
    map->rootBranch().subtree(rootOffset + 1);

  // Keep left all the way down.
  for (; h != level; ++h)
    NR = NR.branch().subtree(0);
  return NR;
}

// treeIncrement - Move to the beginning of the next leaf node.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::treeIncrement() {
  assert(branched() && "treeIncrement is not for small maps");
  assert(path.size() == map->height && "inconsistent iterator");
  do path.pop_back();
  while (!path.empty() && path.back().second == path.back().first.size() - 1);
  if (path.empty()) {
    ++rootOffset;
    if (!valid())
      return;
  } else
    ++path.back().second;
  pathFillLeft();
}

// treeDecrement - Move to the end of the previous leaf node.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::treeDecrement() {
  assert(branched() && "treeDecrement is not for small maps");
  if (valid()) {
    assert(path.size() == map->height && "inconsistent iterator");
    do path.pop_back();
    while (!path.empty() && path.back().second == 0);
  }
  if (path.empty()) {
    assert(rootOffset && "cannot treeDecrement() on begin()");
    --rootOffset;
  } else
    --path.back().second;
  pathFillRight();
}

// treeFind - Find in a branched tree.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
const_iterator::treeFind(KeyT x) {
  path.clear();
  rootOffset = map->rootBranch().findFrom(0, map->rootSize, x);
  if (valid())
    pathFillFind(x);
}


//===----------------------------------------------------------------------===//
//---                                iterator                             ----//
//===----------------------------------------------------------------------===//

namespace IntervalMapImpl {

  /// distribute - Compute a new distribution of node elements after an overflow
  /// or underflow. Reserve space for a new element at Position, and compute the
  /// node that will hold Position after redistributing node elements.
  ///
  /// It is required that
  ///
  ///   Elements == sum(CurSize), and
  ///   Elements + Grow <= Nodes * Capacity.
  ///
  /// NewSize[] will be filled in such that:
  ///
  ///   sum(NewSize) == Elements, and
  ///   NewSize[i] <= Capacity.
  ///
  /// The returned index is the node where Position will go, so:
  ///
  ///   sum(NewSize[0..idx-1]) <= Position
  ///   sum(NewSize[0..idx])   >= Position
  ///
  /// The last equality, sum(NewSize[0..idx]) == Position, can only happen when
  /// Grow is set and NewSize[idx] == Capacity-1. The index points to the node
  /// before the one holding the Position'th element where there is room for an
  /// insertion.
  ///
  /// @param Nodes    The number of nodes.
  /// @param Elements Total elements in all nodes.
  /// @param Capacity The capacity of each node.
  /// @param CurSize  Array[Nodes] of current node sizes, or NULL.
  /// @param NewSize  Array[Nodes] to receive the new node sizes.
  /// @param Position Insert position.
  /// @param Grow     Reserve space for a new element at Position.
  /// @return         (node, offset) for Position.
  IdxPair distribute(unsigned Nodes, unsigned Elements, unsigned Capacity,
                     const unsigned *CurSize, unsigned NewSize[],
                     unsigned Position, bool Grow);

}

template <typename KeyT, typename ValT, unsigned N, typename Traits>
class IntervalMap<KeyT, ValT, N, Traits>::iterator : public const_iterator {
  friend class IntervalMap;
  typedef IntervalMapImpl::IdxPair IdxPair;

  explicit iterator(IntervalMap &map) : const_iterator(map) {}

  void setNodeSize(unsigned Level, unsigned Size);
  void setNodeStop(unsigned Level, KeyT Stop);
  void insertNode(unsigned Level, NodeRef Node, KeyT Stop);
  void overflowLeaf();
  void treeInsert(KeyT a, KeyT b, ValT y);

public:
  /// insert - Insert mapping [a;b] -> y before the current position.
  void insert(KeyT a, KeyT b, ValT y);

};

/// setNodeSize - Set the size of the node at path[level], updating both path
/// and the real tree.
/// @param level 0 is just below the root, map->height - 1 for the leaves.
/// @param size  New node size.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
iterator::setNodeSize(unsigned Level, unsigned Size) {
  this->pathNode(Level).setSize(Size);
  if (Level)
    this->pathNode(Level-1).branch()
      .subtree(this->pathOffset(Level-1)).setSize(Size);
  else
    this->map->rootBranch().subtree(this->rootOffset).setSize(Size);
}

/// setNodeStop - Update the stop key of the current node at level and above.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
iterator::setNodeStop(unsigned Level, KeyT Stop) {
  while (Level--) {
    this->pathNode(Level).branch().stop(this->pathOffset(Level)) = Stop;
    if (this->pathOffset(Level) != this->pathNode(Level).size() - 1)
      return;
  }
  this->map->rootBranch().stop(this->rootOffset) = Stop;
}

/// insertNode - insert a node before the current path at level.
/// Leave the current path pointing at the new node.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
iterator::insertNode(unsigned Level, NodeRef Node, KeyT Stop) {
  if (!Level) {
    // Insert into the root branch node.
    IntervalMap &IM = *this->map;
    if (IM.rootSize < RootBranch::Capacity) {
      IM.rootBranch().insert(this->rootOffset, IM.rootSize, Node, Stop);
      ++IM.rootSize;
      return;
    }

    // We need to split the root while keeping our position.
    IdxPair Offset = IM.splitRoot(this->rootOffset);
    this->rootOffset = Offset.first;
    this->path.insert(this->path.begin(),std::make_pair(
      this->map->rootBranch().subtree(Offset.first), Offset.second));
    Level = 1;
  }

  // When inserting before end(), make sure we have a valid path.
  if (!this->valid()) {
    this->treeDecrement();
    ++this->pathOffset(Level-1);
  }

  // Insert into the branch node at level-1.
  NodeRef NR = this->pathNode(Level-1);
  unsigned Offset = this->pathOffset(Level-1);
  assert(NR.size() < Branch::Capacity && "Branch overflow");
  NR.branch().insert(Offset, NR.size(), Node, Stop);
  setNodeSize(Level - 1, NR.size() + 1);
}

// insert
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
iterator::insert(KeyT a, KeyT b, ValT y) {
  if (this->branched())
    return treeInsert(a, b, y);
  IdxPair IP = this->map->rootLeaf().insertFrom(this->rootOffset,
                                                this->map->rootSize,
                                                a, b, y);
  if (IP.second <= RootLeaf::Capacity) {
    this->rootOffset = IP.first;
    this->map->rootSize = IP.second;
    return;
  }
  IdxPair Offset = this->map->branchRoot(this->rootOffset);
  this->rootOffset = Offset.first;
  this->path.push_back(std::make_pair(
    this->map->rootBranch().subtree(Offset.first), Offset.second));
  treeInsert(a, b, y);
}


template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
iterator::treeInsert(KeyT a, KeyT b, ValT y) {
  if (!this->valid()) {
    // end() has an empty path. Go back to the last leaf node and use an
    // invalid offset instead.
    this->treeDecrement();
    ++this->treeLeafOffset();
  }
  IdxPair IP = this->treeLeaf().insertFrom(this->treeLeafOffset(),
                                           this->treeLeafSize(), a, b, y);
  this->treeLeafOffset() = IP.first;
  if (IP.second <= Leaf::Capacity) {
    setNodeSize(this->map->height - 1, IP.second);
    if (IP.first == IP.second - 1)
      setNodeStop(this->map->height - 1, this->treeLeaf().stop(IP.first));
    return;
  }
  // Leaf node has no space.
  overflowLeaf();
  IP = this->treeLeaf().insertFrom(this->treeLeafOffset(),
                                   this->treeLeafSize(), a, b, y);
  this->treeLeafOffset() = IP.first;
  setNodeSize(this->map->height-1, IP.second);
  if (IP.first == IP.second - 1)
    setNodeStop(this->map->height - 1, this->treeLeaf().stop(IP.first));

  // FIXME: Handle cross-node coalescing.
}

// overflowLeaf - Distribute entries of the current leaf node evenly among
// its siblings and ensure that the current node is not full.
// This may require allocating a new node.
template <typename KeyT, typename ValT, unsigned N, typename Traits>
void IntervalMap<KeyT, ValT, N, Traits>::
iterator::overflowLeaf() {
  unsigned CurSize[4];
  Leaf *Node[4];
  unsigned Nodes = 0;
  unsigned Elements = 0;
  unsigned Offset = this->treeLeafOffset();

  // Do we have a left sibling?
  NodeRef LeftSib = this->leftSibling(this->map->height-1);
  if (LeftSib) {
    Offset += Elements = CurSize[Nodes] = LeftSib.size();
    Node[Nodes++] = &LeftSib.leaf();
  }

  // Current leaf node.
  Elements += CurSize[Nodes] = this->treeLeafSize();
  Node[Nodes++] = &this->treeLeaf();

  // Do we have a right sibling?
  NodeRef RightSib = this->rightSibling(this->map->height-1);
  if (RightSib) {
    Offset += Elements = CurSize[Nodes] = RightSib.size();
    Node[Nodes++] = &RightSib.leaf();
  }

  // Do we need to allocate a new node?
  unsigned NewNode = 0;
  if (Elements + 1 > Nodes * Leaf::Capacity) {
    // Insert NewNode at the penultimate position, or after a single node.
    NewNode = Nodes == 1 ? 1 : Nodes - 1;
    CurSize[Nodes] = CurSize[NewNode];
    Node[Nodes] = Node[NewNode];
    CurSize[NewNode] = 0;
    Node[NewNode] = this->map->allocLeaf();
    ++Nodes;
  }

  // Compute the new element distribution.
  unsigned NewSize[4];
  IdxPair NewOffset =
    IntervalMapImpl::distribute(Nodes, Elements, Leaf::Capacity,
                                CurSize, NewSize, Offset, true);

  // Move current location to the leftmost node.
  if (LeftSib)
    this->treeDecrement();

  // Move elements right.
  for (int n = Nodes - 1; n; --n) {
    if (CurSize[n] == NewSize[n])
      continue;
    for (int m = n - 1; m != -1; --m) {
      int d = Node[n]->adjLeftSib(CurSize[n], *Node[m], CurSize[m],
                                        NewSize[n] - CurSize[n]);
      CurSize[m] -= d;
      CurSize[n] += d;
      // Keep going if the current node was exhausted.
      if (CurSize[n] >= NewSize[n])
          break;
    }
  }

  // Move elements left.
  for (unsigned n = 0; n != Nodes - 1; ++n) {
    if (CurSize[n] == NewSize[n])
      continue;
    for (unsigned m = n + 1; m != Nodes; ++m) {
      int d = Node[m]->adjLeftSib(CurSize[m], *Node[n], CurSize[n],
                                        CurSize[n] -  NewSize[n]);
      CurSize[m] += d;
      CurSize[n] -= d;
      // Keep going if the current node was exhausted.
      if (CurSize[n] >= NewSize[n])
          break;
    }
  }

#ifndef NDEBUG
  for (unsigned n = 0; n != Nodes; n++)
    assert(CurSize[n] == NewSize[n] && "Insufficient element shuffle");
#endif

  // Elements have been rearranged, now update node sizes and stops.
  unsigned Pos = 0;
  for (;;) {
    KeyT Stop = Node[Pos]->stop(NewSize[Pos]-1);
    if (NewNode && Pos == NewNode)
      insertNode(this->map->height - 1, NodeRef(Node[Pos], NewSize[Pos]), Stop);
    else {
      setNodeSize(this->map->height - 1, NewSize[Pos]);
      setNodeStop(this->map->height - 1, Stop);
    }
    if (Pos + 1 == Nodes)
      break;
    this->treeIncrement();
    ++Pos;
  }

  // Where was I? Find NewOffset.
  while(Pos != NewOffset.first) {
    this->treeDecrement();
    --Pos;
  }
  this->treeLeafOffset() = NewOffset.second;
}

} // namespace llvm

#endif
