//===- DSSupport.h - Support for datastructure graphs -----------*- C++ -*-===//
//
// Support for graph nodes, call sites, and types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DSSUPPORT_H
#define LLVM_ANALYSIS_DSSUPPORT_H

#include <vector>
#include <functional>
#include <string>
#include "Support/HashExtras.h"
#include "Support/hash_set"

class Function;
class CallInst;
class Value;
class GlobalValue;
class Type;

class DSNode;                  // Each node in the graph
class DSGraph;                 // A graph for a function

namespace DS { // FIXME: After the paper, this should get cleaned up
  enum { PointerShift = 3,     // 64bit ptrs = 3, 32 bit ptrs = 2
         PointerSize = 1 << PointerShift
  };

  // isPointerType - Return true if this first class type is big enough to hold
  // a pointer.
  //
  bool isPointerType(const Type *Ty);
};

//===----------------------------------------------------------------------===//
/// DSNodeHandle - Implement a "handle" to a data structure node that takes care
/// of all of the add/un'refing of the node to prevent the backpointers in the
/// graph from getting out of date.  This class represents a "pointer" in the
/// graph, whose destination is an indexed offset into a node.
///
/// Note: some functions that are marked as inline in DSNodeHandle are actually
/// defined in DSNode.h because they need knowledge of DSNode operation. Putting
/// them in a CPP file wouldn't help making them inlined and keeping DSNode and
/// DSNodeHandle (and friends) in one file complicates things.
///
class DSNodeHandle {
  DSNode *N;
  unsigned Offset;
public:
  // Allow construction, destruction, and assignment...
  DSNodeHandle(DSNode *n = 0, unsigned offs = 0) : N(0), Offset(offs) {
    setNode(n);
  }
  DSNodeHandle(const DSNodeHandle &H) : N(0), Offset(H.Offset) { setNode(H.N); }
  ~DSNodeHandle() { setNode((DSNode*)0); }
  DSNodeHandle &operator=(const DSNodeHandle &H) {
    setNode(H.N); Offset = H.Offset;
    return *this;
  }

  bool operator<(const DSNodeHandle &H) const {  // Allow sorting
    return N < H.N || (N == H.N && Offset < H.Offset);
  }
  bool operator>(const DSNodeHandle &H) const { return H < *this; }
  bool operator==(const DSNodeHandle &H) const { // Allow comparison
    return N == H.N && Offset == H.Offset;
  }
  bool operator!=(const DSNodeHandle &H) const { return !operator==(H); }

  inline void swap(DSNodeHandle &H);

  // Allow explicit conversion to DSNode...
  DSNode *getNode() const { return N; }
  unsigned getOffset() const { return Offset; }

  inline void setNode(DSNode *N);  // Defined inline later...
  void setOffset(unsigned O) { Offset = O; }

  void addEdgeTo(unsigned LinkNo, const DSNodeHandle &N);
  void addEdgeTo(const DSNodeHandle &N) { addEdgeTo(0, N); }

  /// mergeWith - Merge the logical node pointed to by 'this' with the node
  /// pointed to by 'N'.
  ///
  void mergeWith(const DSNodeHandle &N);

  // hasLink - Return true if there is a link at the specified offset...
  inline bool hasLink(unsigned Num) const;

  /// getLink - Treat this current node pointer as a pointer to a structure of
  /// some sort.  This method will return the pointer a mem[this+Num]
  ///
  inline const DSNodeHandle &getLink(unsigned Num) const;
  inline DSNodeHandle &getLink(unsigned Num);

  inline void setLink(unsigned Num, const DSNodeHandle &NH);
};

inline void swap(DSNodeHandle &NH1, DSNodeHandle &NH2) { NH1.swap(NH2); }

//===----------------------------------------------------------------------===//
/// DSCallSite - Representation of a call site via its call instruction,
/// the DSNode handle for the callee function (or function pointer), and
/// the DSNode handles for the function arguments.
///
/// One unusual aspect of this callsite record is the ResolvingCaller member.
/// If this is non-null, then it indicates the function that allowed a call-site
/// to finally be resolved.  Because of indirect calls, this function may not
/// actually be the function that contains the Call instruction itself.  This is
/// used by the BU and TD passes to communicate.
/// 
class DSCallSite {
  CallInst    *Inst;                    // Actual call site
  DSNodeHandle RetVal;                  // Returned value
  DSNodeHandle Callee;                  // The function node called
  std::vector<DSNodeHandle> CallArgs;   // The pointer arguments
  Function    *ResolvingCaller;         // See comments above

  static void InitNH(DSNodeHandle &NH, const DSNodeHandle &Src,
                     const hash_map<const DSNode*, DSNode*> &NodeMap) {
    if (DSNode *N = Src.getNode()) {
      hash_map<const DSNode*, DSNode*>::const_iterator I = NodeMap.find(N);
      assert(I != NodeMap.end() && "Not not in mapping!");

      NH.setOffset(Src.getOffset());
      NH.setNode(I->second);
    }
  }

  static void InitNH(DSNodeHandle &NH, const DSNodeHandle &Src,
                     const hash_map<const DSNode*, DSNodeHandle> &NodeMap) {
    if (DSNode *N = Src.getNode()) {
      hash_map<const DSNode*, DSNodeHandle>::const_iterator I = NodeMap.find(N);
      assert(I != NodeMap.end() && "Not not in mapping!");

      NH.setOffset(Src.getOffset()+I->second.getOffset());
      NH.setNode(I->second.getNode());
    }
  }

  DSCallSite();                         // DO NOT IMPLEMENT
public:
  /// Constructor.  Note - This ctor destroys the argument vector passed in.  On
  /// exit, the argument vector is empty.
  ///
  DSCallSite(CallInst &inst, const DSNodeHandle &rv, const DSNodeHandle &callee,
             std::vector<DSNodeHandle> &Args)
    : Inst(&inst), RetVal(rv), Callee(callee), ResolvingCaller(0) {
    Args.swap(CallArgs);
  }

  DSCallSite(const DSCallSite &DSCS)   // Simple copy ctor
    : Inst(DSCS.Inst), RetVal(DSCS.RetVal),
      Callee(DSCS.Callee), CallArgs(DSCS.CallArgs),
      ResolvingCaller(DSCS.ResolvingCaller) {}

  /// Mapping copy constructor - This constructor takes a preexisting call site
  /// to copy plus a map that specifies how the links should be transformed.
  /// This is useful when moving a call site from one graph to another.
  ///
  template<typename MapTy>
  DSCallSite(const DSCallSite &FromCall, const MapTy &NodeMap) {
    Inst = FromCall.Inst;
    InitNH(RetVal, FromCall.RetVal, NodeMap);
    InitNH(Callee, FromCall.Callee, NodeMap);

    CallArgs.resize(FromCall.CallArgs.size());
    for (unsigned i = 0, e = FromCall.CallArgs.size(); i != e; ++i)
      InitNH(CallArgs[i], FromCall.CallArgs[i], NodeMap);
    ResolvingCaller = FromCall.ResolvingCaller;
  }

  // Accessor functions...
  Function           &getCaller()     const;
  CallInst           &getCallInst()   const { return *Inst; }
        DSNodeHandle &getRetVal()           { return RetVal; }
        DSNodeHandle &getCallee()           { return Callee; }
  const DSNodeHandle &getRetVal()     const { return RetVal; }
  const DSNodeHandle &getCallee()     const { return Callee; }
  void setCallee(const DSNodeHandle &H) { Callee = H; }

  unsigned            getNumPtrArgs() const { return CallArgs.size(); }

  Function           *getResolvingCaller() const { return ResolvingCaller; }
  void setResolvingCaller(Function *F) { ResolvingCaller = F; }

  DSNodeHandle &getPtrArg(unsigned i) {
    assert(i < CallArgs.size() && "Argument to getPtrArgNode is out of range!");
    return CallArgs[i];
  }
  const DSNodeHandle &getPtrArg(unsigned i) const {
    assert(i < CallArgs.size() && "Argument to getPtrArgNode is out of range!");
    return CallArgs[i];
  }

  void swap(DSCallSite &CS) {
    if (this != &CS) {
      std::swap(Inst, CS.Inst);
      std::swap(RetVal, CS.RetVal);
      std::swap(Callee, CS.Callee);
      std::swap(CallArgs, CS.CallArgs);
      std::swap(ResolvingCaller, CS.ResolvingCaller);
    }
  }

  // MergeWith - Merge the return value and parameters of the these two call
  // sites.
  void mergeWith(DSCallSite &CS) {
    getRetVal().mergeWith(CS.getRetVal());
    unsigned MinArgs = getNumPtrArgs();
    if (CS.getNumPtrArgs() < MinArgs) MinArgs = CS.getNumPtrArgs();

    for (unsigned a = 0; a != MinArgs; ++a)
      getPtrArg(a).mergeWith(CS.getPtrArg(a));
  }

  /// markReachableNodes - This method recursively traverses the specified
  /// DSNodes, marking any nodes which are reachable.  All reachable nodes it
  /// adds to the set, which allows it to only traverse visited nodes once.
  ///
  void markReachableNodes(hash_set<DSNode*> &Nodes);

  bool operator<(const DSCallSite &CS) const {
    if (Callee < CS.Callee) return true;   // This must sort by callee first!
    if (Callee > CS.Callee) return false;
    if (RetVal < CS.RetVal) return true;
    if (RetVal > CS.RetVal) return false;
    return CallArgs < CS.CallArgs;
  }

  bool operator==(const DSCallSite &CS) const {
    return RetVal == CS.RetVal && Callee == CS.Callee &&
           CallArgs == CS.CallArgs;
  }
};

inline void swap(DSCallSite &CS1, DSCallSite &CS2) { CS1.swap(CS2); }

#endif
