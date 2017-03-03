//===--- HFSort.h - Cluster functions by hotness --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Cluster functions by hotness.  There are four clustering algorithms:
// 1. clusterize
// 2. HFsort+
// 3. pettisAndHansen
// 4. randomClusters
//
// See original code in hphp/utils/hfsort.[h,cpp]
//===----------------------------------------------------------------------===//

// TODO: copyright/license msg.

/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2016 Facebook, Inc. (http://www.facebook.com)     |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | http://www.php.net/license/3_01.txt                                  |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
*/

#ifndef LLVM_TOOLS_LLVM_BOLT_HFSORT_H
#define LLVM_TOOLS_LLVM_BOLT_HFSORT_H

#include <string>
#include <unordered_set>
#include <vector>
#include <functional>

#if defined(__x86_64__) && !defined(_MSC_VER)
#  if (!defined USE_SSECRC)
#    define USE_SSECRC
#  endif
#else
#  undef USE_SSECRC
#endif

namespace llvm {
namespace bolt {

using TargetId = size_t;
constexpr TargetId InvalidId = -1;

class Arc {
public:
  Arc(TargetId S, TargetId D, double W = 0)
      : Src(S)
      , Dst(D)
      , Weight(W)
  {}
  Arc(const Arc&) = delete;

  friend bool operator==(const Arc &Lhs, const Arc &Rhs) {
    return Lhs.Src == Rhs.Src && Lhs.Dst == Rhs.Dst;
  }

  const TargetId Src;
  const TargetId Dst;
  mutable double Weight;
  mutable double NormalizedWeight{0};
  mutable double AvgCallOffset{0};
};

namespace {

inline int64_t hashCombine(const int64_t Seed, const int64_t Val) {
  std::hash<int64_t> Hasher;
  return Seed ^ (Hasher(Val) + 0x9e3779b9 + (Seed << 6) + (Seed >> 2));
}

inline size_t hash_int64_fallback(int64_t key) {
  // "64 bit Mix Functions", from Thomas Wang's "Integer Hash Function."
  // http://www.concentric.net/~ttwang/tech/inthash.htm
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ ((unsigned long long)key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ ((unsigned long long)key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ ((unsigned long long)key >> 28);
  return static_cast<size_t>(static_cast<uint32_t>(key));
}

inline size_t hash_int64(int64_t k) {
#if defined(USE_SSECRC) && defined(__SSE4_2__)
  size_t h = 0;
  __asm("crc32q %1, %0\n" : "+r"(h) : "rm"(k));
  return h;
#else
  return hash_int64_fallback(k);
#endif
}
  
inline size_t hash_int64_pair(int64_t k1, int64_t k2) {
#if defined(USE_SSECRC) && defined(__SSE4_2__)
  // crc32 is commutative, so we need to perturb k1 so that (k1, k2) hashes
  // differently from (k2, k1).
  k1 += k1;
  __asm("crc32q %1, %0\n" : "+r" (k1) : "rm"(k2));
  return k1;
#else
  return (hash_int64(k1) << 1) ^ hash_int64(k2);
#endif
}
  
}

class ArcHash {
public:
  int64_t operator()(const Arc &Arc) const {
#ifdef USE_STD_HASH
    std::hash<int64_t> Hasher;
    return hashCombine(Hasher(Arc.Src), Arc.Dst);
#else
    return hash_int64_pair(int64_t(Arc.Src), int64_t(Arc.Dst));
#endif
  }
};

class TargetNode {
public:
  explicit TargetNode(uint32_t Size, uint32_t Samples = 0)
    : Size(Size), Samples(Samples)
  {}

  uint32_t Size;
  uint32_t Samples;

  // preds and succs contain no duplicate elements and self arcs are not allowed
  std::vector<TargetId> Preds;
  std::vector<TargetId> Succs;
};

class TargetGraph {
public:
  TargetId addTarget(uint32_t Size, uint32_t Samples = 0);
  const Arc &incArcWeight(TargetId Src, TargetId Dst, double W = 1.0);

  std::vector<TargetNode> Targets;
  std::unordered_set<Arc, ArcHash> Arcs;
};

class Cluster {
public:
  Cluster(TargetId Id, const TargetNode &F);

  std::string toString() const;
  double density() const {
    return (double)Samples / Size;
  }

  std::vector<TargetId> Targets;
  uint32_t Samples;
  uint32_t Size;
  bool Frozen; // not a candidate for merging
};

/*
 * Cluster functions in order to minimize call distance.
 */
std::vector<Cluster> clusterize(const TargetGraph &Cg);

/*
 * Optimize function placement for iTLB cache and i-cache.
 */
std::vector<Cluster> hfsortPlus(const TargetGraph &Cg);

/*
 * Pettis-Hansen code layout algorithm
 * reference: K. Pettis and R. C. Hansen, "Profile Guided Code Positioning",
 * PLDI '90
 */
std::vector<Cluster> pettisAndHansen(const TargetGraph &Cg);

/* Group functions into clusters randomly. */
std::vector<Cluster> randomClusters(const TargetGraph &Cg);

}
}

#endif
