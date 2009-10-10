//===-- SSAUpdater.h - Unstructured SSA Update Tool -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SSAUpdater class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SSAUPDATER_H
#define LLVM_TRANSFORMS_UTILS_SSAUPDATER_H

namespace llvm {
  class Value;
  class BasicBlock;
  class Use;
  
/// SSAUpdater - This class updates SSA form for a set of values defined in
/// multiple blocks.  This is used when code duplication or another unstructured
/// transformation wants to rewrite a set of uses of one value with uses of a
/// set of values.
class SSAUpdater {
  /// AvailableVals - This keeps track of which value to use on a per-block
  /// basis.  When we insert PHI nodes, we keep track of them here.  We use
  /// WeakVH's for the value of the map because we RAUW PHI nodes when we
  /// eliminate them, and want the WeakVH to track this.
  //typedef DenseMap<BasicBlock*, TrackingVH<Value> > AvailableValsTy;
  void *AV;
  
  /// PrototypeValue is an arbitrary representative value, which we derive names
  /// and a type for PHI nodes.
  Value *PrototypeValue;
  
  /// IncomingPredInfo - We use this as scratch space when doing our recursive
  /// walk.  This should only be used in GetValueInBlockInternal, normally it
  /// should be empty.
  //std::vector<std::pair<BasicBlock*, TrackingVH<Value> > > IncomingPredInfo;
  void *IPI;
public:
  SSAUpdater();
  ~SSAUpdater();
  
  /// Initialize - Reset this object to get ready for a new set of SSA
  /// updates.  ProtoValue is the value used to name PHI nodes.
  void Initialize(Value *ProtoValue);
  
  /// AddAvailableValue - Indicate that a rewritten value is available in the
  /// specified block with the specified value.
  void AddAvailableValue(BasicBlock *BB, Value *V);
  
  /// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
  /// live at the end of the specified block.
  Value *GetValueAtEndOfBlock(BasicBlock *BB);
  
  /// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
  /// which use their value in the corresponding predecessor.
  void RewriteUse(Use &U);
  
private:
  Value *GetValueAtEndOfBlockInternal(BasicBlock *BB);
  void operator=(const SSAUpdater&); // DO NOT IMPLEMENT
  SSAUpdater(const SSAUpdater&);     // DO NOT IMPLEMENT
};

} // End llvm namespace

#endif
