//===-- llvm/CFGdecls.h - CFG forward declarations ---------------*- C++ -*--=//
//
// This file contains forward declarations for CFG functions and data
// structures.  This is used to reduce compile time dependencies among files.
// Any users of these functions must include CFG.h to get their full 
// definitions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CFG_DECLS_H
#define LLVM_CFG_DECLS_H

#include "llvm/Value.h"
class TerminatorInst;
class BasicBlock;
class Method;

//===----------------------------------------------------------------------===//
//                                Interface
//===----------------------------------------------------------------------===//

namespace cfg {

//===--------------------------------------------------------------------===//
// Predecessor iterator code
//===--------------------------------------------------------------------===//
// 
// This is used to figure out what basic blocks we could be coming from.
//

// Forward declare iterator class template...
template <class _Ptr, class _USE_iterator> class PredIterator;

typedef PredIterator<BasicBlock, Value::use_iterator> pred_iterator;
typedef PredIterator<const BasicBlock, 
		     Value::use_const_iterator> pred_const_iterator;

inline pred_iterator       pred_begin(      BasicBlock *BB);
inline pred_const_iterator pred_begin(const BasicBlock *BB);
inline pred_iterator       pred_end  (      BasicBlock *BB);
inline pred_const_iterator pred_end  (const BasicBlock *BB);


//===--------------------------------------------------------------------===//
// Successor iterator code
//===--------------------------------------------------------------------===//
// 
// This is used to figure out what basic blocks we could be going to...
//

// Forward declare iterator class template...
template <class _Term, class _BB> class SuccIterator;

typedef SuccIterator<TerminatorInst*, BasicBlock> succ_iterator;
typedef SuccIterator<const TerminatorInst*, 
		     const BasicBlock> succ_const_iterator;

inline succ_iterator       succ_begin(      BasicBlock *BB);
inline succ_const_iterator succ_begin(const BasicBlock *BB);
inline succ_iterator       succ_end  (      BasicBlock *BB);
inline succ_const_iterator succ_end  (const BasicBlock *BB);


//===--------------------------------------------------------------------===//
// <Reverse> Depth First CFG iterator code
//===--------------------------------------------------------------------===//
// 
// This is used to visit basic blocks in a method in either depth first, or 
// reverse depth first ordering, depending on the value passed to the df_begin
// method.
//
struct      BasicBlockGraph;
struct ConstBasicBlockGraph;
struct      InverseBasicBlockGraph;
struct ConstInverseBasicBlockGraph;

// Forward declare iterator class template...
template<class GraphInfo> class DFIterator;

// Normal Depth First Iterator Definitions (Forward and Reverse)
typedef DFIterator<     BasicBlockGraph> df_iterator;
typedef DFIterator<ConstBasicBlockGraph> df_const_iterator;

inline df_iterator       df_begin(      Method *M, bool Reverse = false);
inline df_const_iterator df_begin(const Method *M, bool Reverse = false);
inline df_iterator       df_end  (      Method *M);
inline df_const_iterator df_end  (const Method *M);

inline df_iterator       df_begin(      BasicBlock *BB, bool Reverse = false);
inline df_const_iterator df_begin(const BasicBlock *BB, bool Reverse = false);
inline df_iterator       df_end  (      BasicBlock *BB);
inline df_const_iterator df_end  (const BasicBlock *BB);


// Inverse Depth First Iterator Definitions (Forward and Reverse) - Traverse
// predecessors instead of successors...
//
typedef DFIterator<     InverseBasicBlockGraph> idf_iterator;
typedef DFIterator<ConstInverseBasicBlockGraph> idf_const_iterator;

inline idf_iterator       idf_begin(      BasicBlock *BB, bool Reverse = false);
inline idf_const_iterator idf_begin(const BasicBlock *BB, bool Reverse = false);
inline idf_iterator       idf_end  (      BasicBlock *BB);
inline idf_const_iterator idf_end  (const BasicBlock *BB);


//===--------------------------------------------------------------------===//
// Post Order CFG iterator code
//===--------------------------------------------------------------------===//
// 
// This is used to visit basic blocks in a method in standard post order.
//

// Forward declare iterator class template...
template<class BBType, class SuccItTy> class POIterator;

typedef POIterator<BasicBlock, succ_iterator> po_iterator;
typedef POIterator<const BasicBlock, 
		   succ_const_iterator> po_const_iterator;

inline po_iterator       po_begin(      Method *M);
inline po_const_iterator po_begin(const Method *M);
inline po_iterator       po_end  (      Method *M);
inline po_const_iterator po_end  (const Method *M);

inline po_iterator       po_begin(      BasicBlock *BB);
inline po_const_iterator po_begin(const BasicBlock *BB);
inline po_iterator       po_end  (      BasicBlock *BB);
inline po_const_iterator po_end  (const BasicBlock *BB);


//===--------------------------------------------------------------------===//
// Reverse Post Order CFG iterator code
//===--------------------------------------------------------------------===//
// 
// This is used to visit basic blocks in a method in reverse post order.  This
// class is awkward to use because I don't know a good incremental algorithm to
// computer RPO from a graph.  Because of this, the construction of the 
// ReversePostOrderTraversal object is expensive (it must walk the entire graph
// with a postorder iterator to build the data structures).  The moral of this
// story is: Don't create more ReversePostOrderTraversal classes than neccesary.
//
// This class should be used like this:
// {
//   cfg::ReversePostOrderTraversal RPOT(MethodPtr);   // Expensive to create
//   for (cfg::rpo_iterator I = RPOT.begin(); I != RPOT.end(); ++I) {
//      ...
//   }
//   for (cfg::rpo_iterator I = RPOT.begin(); I != RPOT.end(); ++I) {
//      ...
//   }
// }
//

//typedef reverse_iterator<vector<BasicBlock*>::const_iterator> 
// rpo_const_iterator;
typedef reverse_iterator<vector<BasicBlock*>::iterator> rpo_iterator;

class ReversePostOrderTraversal;

}    // End namespace cfg

#endif
