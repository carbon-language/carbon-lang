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
#include <cassert>
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
  mutable DSNode *N;
  mutable unsigned Offset;
  void operator==(const DSNode *N);  // DISALLOW, use to promote N to nodehandle
public:
  // Allow construction, destruction, and assignment...
  DSNodeHandle(DSNode *n = 0, unsigned offs = 0) : N(0), Offset(offs) {
    setNode(n);
  }
  DSNodeHandle(const DSNodeHandle &H) : N(0), Offset(0) {
    setNode(H.getNode());
    Offset = H.Offset;      // Must read offset AFTER the getNode()
  }
  ~DSNodeHandle() { setNode((DSNode*)0); }
  DSNodeHandle &operator=(const DSNodeHandle &H) {
    if (&H == this) return *this;  // Don't set offset to 0 if self assigning.
    Offset = 0; setNode(H.getNode()); Offset = H.Offset;
    return *this;
  }

  bool operator<(const DSNodeHandle &H) const {  // Allow sorting
    return getNode() < H.getNode() || (N == H.N && Offset < H.Offset);
  }
  bool operator>(const DSNodeHandle &H) const { return H < *this; }
  bool operator==(const DSNodeHandle &H) const { // Allow comparison
    return getNode() == H.getNode() && Offset == H.Offset;
  }
  bool operator!=(const DSNodeHandle &H) const { return !operator==(H); }

  inline void swap(DSNodeHandle &NH) {
    std::swap(Offset, NH.Offset);
    std::swap(N, NH.N);
  }

  /// isNull - Check to see if getNode() == 0, without going through the trouble
  /// of checking to see if we are forwarding...
  bool isNull() const { return N == 0; }

  // Allow explicit conversion to DSNode...
  inline DSNode *getNode() const;  // Defined inline in DSNode.h
  unsigned getOffset() const { return Offset; }

  inline void setNode(DSNode *N);  // Defined inline in DSNode.h
  void setOffset(unsigned O) {
    //assert((!N || Offset < N->Size || (N->Size == 0 && Offset == 0) ||
    //       !N->ForwardNH.isNull()) && "Node handle offset out of range!");
    //assert((!N || O < N->Size || (N->Size == 0 && O == 0) ||
    //       !N->ForwardNH.isNull()) && "Node handle offset out of range!");
    Offset = O;
  }

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
private:
  DSNode *HandleForwarding() const;
};

namespace std {
  inline void swap(DSNodeHandle &NH1, DSNodeHandle &NH2) { NH1.swap(NH2); }
}

//===----------------------------------------------------------------------===//
/// DSCallSite - Representation of a call site via its call instruction,
/// the DSNode handle for the callee function (or function pointer), and
/// the DSNode handles for the function arguments.
/// 
class DSCallSite {
  CallInst    *Inst;                 // Actual call site
  Function    *CalleeF;              // The function called (direct call)
  DSNodeHandle CalleeN;              // The function node called (indirect call)
  DSNodeHandle RetVal;               // Returned value
  std::vector<DSNodeHandle> CallArgs;// The pointer arguments

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
  DSCallSite(CallInst &inst, const DSNodeHandle &rv, DSNode *Callee,
             std::vector<DSNodeHandle> &Args)
    : Inst(&inst), CalleeF(0), CalleeN(Callee), RetVal(rv) {
    assert(Callee && "Null callee node specified for call site!");
    Args.swap(CallArgs);
  }
  DSCallSite(CallInst &inst, const DSNodeHandle &rv, Function *Callee,
             std::vector<DSNodeHandle> &Args)
    : Inst(&inst), CalleeF(Callee), RetVal(rv) {
    assert(Callee && "Null callee function specified for call site!");
    Args.swap(CallArgs);
  }

  DSCallSite(const DSCallSite &DSCS)   // Simple copy ctor
    : Inst(DSCS.Inst), CalleeF(DSCS.CalleeF), CalleeN(DSCS.CalleeN),
      RetVal(DSCS.RetVal), CallArgs(DSCS.CallArgs) {}

  /// Mapping copy constructor - This constructor takes a preexisting call site
  /// to copy plus a map that specifies how the links should be transformed.
  /// This is useful when moving a call site from one graph to another.
  ///
  template<typename MapTy>
  DSCallSite(const DSCallSite &FromCall, const MapTy &NodeMap) {
    Inst = FromCall.Inst;
    InitNH(RetVal, FromCall.RetVal, NodeMap);
    InitNH(CalleeN, FromCall.CalleeN, NodeMap);
    CalleeF = FromCall.CalleeF;

    CallArgs.resize(FromCall.CallArgs.size());
    for (unsigned i = 0, e = FromCall.CallArgs.size(); i != e; ++i)
      InitNH(CallArgs[i], FromCall.CallArgs[i], NodeMap);
  }

  const DSCallSite &operator=(const DSCallSite &RHS) {
    Inst     = RHS.Inst;
    CalleeF  = RHS.CalleeF;
    CalleeN  = RHS.CalleeN;
    RetVal   = RHS.RetVal;
    CallArgs = RHS.CallArgs;
    return *this;
  }

  /// isDirectCall - Return true if this call site is a direct call of the
  /// function specified by getCalleeFunc.  If not, it is an indirect call to
  /// the node specified by getCalleeNode.
  ///
  bool isDirectCall() const { return CalleeF != 0; }
  bool isIndirectCall() const { return !isDirectCall(); }


  // Accessor functions...
  Function           &getCaller()     const;
  CallInst           &getCallInst()   const { return *Inst; }
        DSNodeHandle &getRetVal()           { return RetVal; }
  const DSNodeHandle &getRetVal()     const { return RetVal; }

  DSNode *getCalleeNode() const {
    assert(!CalleeF && CalleeN.getNode()); return CalleeN.getNode();
  }
  Function *getCalleeFunc() const {
    assert(!CalleeN.getNode() && CalleeF); return CalleeF;
  }

  unsigned            getNumPtrArgs() const { return CallArgs.size(); }

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
      std::swap(CalleeN, CS.CalleeN);
      std::swap(CalleeF, CS.CalleeF);
      std::swap(CallArgs, CS.CallArgs);
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
    if (isDirectCall()) {      // This must sort by callee first!
      if (CS.isIndirectCall()) return true;
      if (CalleeF < CS.CalleeF) return true;
      if (CalleeF > CS.CalleeF) return false;
    } else {
      if (CS.isDirectCall()) return false;
      if (CalleeN < CS.CalleeN) return true;
      if (CalleeN > CS.CalleeN) return false;
    }
    if (RetVal < CS.RetVal) return true;
    if (RetVal > CS.RetVal) return false;
    return CallArgs < CS.CallArgs;
  }

  bool operator==(const DSCallSite &CS) const {
    return RetVal == CS.RetVal && CalleeN == CS.CalleeN &&
           CalleeF == CS.CalleeF && CallArgs == CS.CallArgs;
  }
};

namespace std {
  inline void swap(DSCallSite &CS1, DSCallSite &CS2) { CS1.swap(CS2); }
}
#endif
