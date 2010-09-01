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
  class PHINode;
  template<typename T> class SmallVectorImpl;
  template<typename T> class SSAUpdaterTraits;
  class BumpPtrAllocator;

/// SSAUpdater - This class updates SSA form for a set of values defined in
/// multiple blocks.  This is used when code duplication or another unstructured
/// transformation wants to rewrite a set of uses of one value with uses of a
/// set of values.
class SSAUpdater {
  friend class SSAUpdaterTraits<SSAUpdater>;

private:
  /// AvailableVals - This keeps track of which value to use on a per-block
  /// basis.  When we insert PHI nodes, we keep track of them here.
  //typedef DenseMap<BasicBlock*, Value*> AvailableValsTy;
  void *AV;

  /// ProtoType holds the type of the values being rewritten.
  const Type *ProtoType;

  // PHI nodes are given a name based on ProtoName.
  std::string ProtoName;

  /// InsertedPHIs - If this is non-null, the SSAUpdater adds all PHI nodes that
  /// it creates to the vector.
  SmallVectorImpl<PHINode*> *InsertedPHIs;

public:
  /// SSAUpdater constructor.  If InsertedPHIs is specified, it will be filled
  /// in with all PHI Nodes created by rewriting.
  explicit SSAUpdater(SmallVectorImpl<PHINode*> *InsertedPHIs = 0);
  ~SSAUpdater();

  /// Initialize - Reset this object to get ready for a new set of SSA
  /// updates with type 'Ty'.  PHI nodes get a name based on 'Name'.
  void Initialize(const Type *Ty, StringRef Name);

  /// AddAvailableValue - Indicate that a rewritten value is available at the
  /// end of the specified block with the specified value.
  void AddAvailableValue(BasicBlock *BB, Value *V);

  /// HasValueForBlock - Return true if the SSAUpdater already has a value for
  /// the specified block.
  bool HasValueForBlock(BasicBlock *BB) const;

  /// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
  /// live at the end of the specified block.
  Value *GetValueAtEndOfBlock(BasicBlock *BB);

  /// GetValueInMiddleOfBlock - Construct SSA form, materializing a value that
  /// is live in the middle of the specified block.
  ///
  /// GetValueInMiddleOfBlock is the same as GetValueAtEndOfBlock except in one
  /// important case: if there is a definition of the rewritten value after the
  /// 'use' in BB.  Consider code like this:
  ///
  ///      X1 = ...
  ///   SomeBB:
  ///      use(X)
  ///      X2 = ...
  ///      br Cond, SomeBB, OutBB
  ///
  /// In this case, there are two values (X1 and X2) added to the AvailableVals
  /// set by the client of the rewriter, and those values are both live out of
  /// their respective blocks.  However, the use of X happens in the *middle* of
  /// a block.  Because of this, we need to insert a new PHI node in SomeBB to
  /// merge the appropriate values, and this value isn't live out of the block.
  ///
  Value *GetValueInMiddleOfBlock(BasicBlock *BB);

  /// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
  /// which use their value in the corresponding predecessor.  Note that this
  /// will not work if the use is supposed to be rewritten to a value defined in
  /// the same block as the use, but above it.  Any 'AddAvailableValue's added
  /// for the use's block will be considered to be below it.
  void RewriteUse(Use &U);

  /// RewriteUseAfterInsertions - Rewrite a use, just like RewriteUse.  However,
  /// this version of the method can rewrite uses in the same block as a
  /// definition, because it assumes that all uses of a value are below any
  /// inserted values.
  void RewriteUseAfterInsertions(Use &U);

private:
  Value *GetValueAtEndOfBlockInternal(BasicBlock *BB);

  void operator=(const SSAUpdater&); // DO NOT IMPLEMENT
  SSAUpdater(const SSAUpdater&);     // DO NOT IMPLEMENT
};

} // End llvm namespace

#endif
