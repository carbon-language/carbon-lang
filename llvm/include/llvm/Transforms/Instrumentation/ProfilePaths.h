//===-- ProfilePaths.h - interface to insert instrumentation -----*- C++ -*--=//
//
// This inserts intrumentation for counting
// execution of paths though a given method
// Its implemented as a "Method" Pass, and called using opt
//
// This pass is implemented by using algorithms similar to 
// 1."Efficient Path Profiling": Ball, T. and Larus, J. R., 
// Proceedings of Micro-29, Dec 1996, Paris, France.
// 2."Efficiently Counting Program events with support for on-line
//   "queries": Ball T., ACM Transactions on Programming Languages
//   and systems, Sep 1994.
//
// The algorithms work on a Graph constructed over the nodes
// made from Basic Blocks: The transformations then take place on
// the constucted graph (implementation in Graph.cpp and GraphAuxillary.cpp)
// and finally, appropriate instrumentation is placed over suitable edges.
// (code inserted through EdgeCode.cpp).
// 
// The algorithm inserts code such that every acyclic path in the CFG
// of a method is identified through a unique number. the code insertion
// is optimal in the sense that its inserted over a minimal set of edges. Also,
// the algorithm makes sure than initialization, path increment and counter
// update can be collapsed into minmimum number of edges.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_PROFILE_PATHS_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_PROFILE_PATHS_H

#include "llvm/Pass.h"

class ProfilePaths: public MethodPass {
 public:
  bool runOnMethod(Method *M);

  // getAnalysisUsageInfo - transform cfg to have just one exit node
  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided);
};

#endif
    
