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

#include "CallGraph.h"

#include <string>
#include <vector>

namespace llvm {
namespace bolt {

class Cluster {
public:
  Cluster(CallGraph::NodeId Id, const CallGraph::Node &F);

  std::string toString() const;
  double density() const { return Density; }
  uint64_t samples() const { return Samples; }
  uint32_t size() const { return Size; }
  bool frozen() const { return Frozen; }
  void freeze() { Frozen = true; }
  void merge(const Cluster &Other, const double Aw = 0);
  void clear();
  size_t numTargets() const {
    return Targets.size();
  }
  const std::vector<CallGraph::NodeId> &targets() const {
    return Targets;
  }
  CallGraph::NodeId target(size_t N) const {
    return Targets[N];
  }
  void reverseTargets();
  bool hasId() const { return Id != -1u; }
  void setId(uint32_t NewId) {
    assert(!hasId());
    Id = NewId;
  }
  uint32_t id() const {
    assert(hasId());
    return Id;
  }
private:
  uint32_t Id{-1u};
  std::vector<CallGraph::NodeId> Targets;
  uint64_t Samples{0};
  uint32_t Size{0};
  double Density{0.0};
  bool Frozen{false}; // not a candidate for merging
};

// Maximum size of a cluster, in bytes.
constexpr uint32_t MaxClusterSize = 1 << 20;

// Size of a huge page in bytes.
constexpr uint32_t HugePageSize = 2 << 20;

inline bool compareClustersDensity(const Cluster &C1, const Cluster &C2) {
  return C1.density() > C2.density();
}

/*
 * Cluster functions in order to minimize call distance.
 */
std::vector<Cluster> clusterize(const CallGraph &Cg);

/*
 * Optimize function placement for iTLB cache and i-cache.
 */
std::vector<Cluster> hfsortPlus(CallGraph &Cg,
                                bool UseGainCache = true,
                                bool UseShortCallCache = true);

/*
 * Pettis-Hansen code layout algorithm
 * reference: K. Pettis and R. C. Hansen, "Profile Guided Code Positioning",
 * PLDI '90
 */
std::vector<Cluster> pettisAndHansen(const CallGraph &Cg);

/* Group functions into clusters randomly. */
std::vector<Cluster> randomClusters(const CallGraph &Cg);

}
}

#endif
