//===-- llvm/CodeGen/MachineFunction.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect native machine code for a function.  This class contains a list of
// MachineBasicBlock instances that make up the current compiled function.
//
// This class also contains pointers to various classes which hold
// target-specific information about the generated code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTION_H
#define LLVM_CODEGEN_MACHINEFUNCTION_H

#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/Annotation.h"

namespace llvm {

class Function;
class TargetMachine;
class SSARegMap;
class MachineFrameInfo;
class MachineConstantPool;
class MachineJumpTableInfo;

// ilist_traits
template <>
struct ilist_traits<MachineBasicBlock> {
  // this is only set by the MachineFunction owning the ilist
  friend class MachineFunction;
  MachineFunction* Parent;

public:
  ilist_traits<MachineBasicBlock>() : Parent(0) { }

  static MachineBasicBlock* getPrev(MachineBasicBlock* N) { return N->Prev; }
  static MachineBasicBlock* getNext(MachineBasicBlock* N) { return N->Next; }

  static const MachineBasicBlock*
  getPrev(const MachineBasicBlock* N) { return N->Prev; }

  static const MachineBasicBlock*
  getNext(const MachineBasicBlock* N) { return N->Next; }

  static void setPrev(MachineBasicBlock* N, MachineBasicBlock* prev) {
    N->Prev = prev;
  }
  static void setNext(MachineBasicBlock* N, MachineBasicBlock* next) {
    N->Next = next;
  }

  static MachineBasicBlock* createSentinel();
  static void destroySentinel(MachineBasicBlock *MBB) { delete MBB; }
  void addNodeToList(MachineBasicBlock* N);
  void removeNodeFromList(MachineBasicBlock* N);
  void transferNodesFromList(iplist<MachineBasicBlock,
                                    ilist_traits<MachineBasicBlock> > &toList,
                             ilist_iterator<MachineBasicBlock> first,
                             ilist_iterator<MachineBasicBlock> last);
};

/// MachineFunctionInfo - This class can be derived from and used by targets to
/// hold private target-specific information for each MachineFunction.  Objects
/// of type are accessed/created with MF::getInfo and destroyed when the
/// MachineFunction is destroyed.
struct MachineFunctionInfo {
  virtual ~MachineFunctionInfo() {};
};

class MachineFunction : private Annotation {
  const Function *Fn;
  const TargetMachine &Target;

  // List of machine basic blocks in function
  ilist<MachineBasicBlock> BasicBlocks;

  // Keeping track of mapping from SSA values to registers
  SSARegMap *SSARegMapping;

  // Used to keep track of target-specific per-machine function information for
  // the target implementation.
  MachineFunctionInfo *MFInfo;

  // Keep track of objects allocated on the stack.
  MachineFrameInfo *FrameInfo;

  // Keep track of constants which are spilled to memory
  MachineConstantPool *ConstantPool;
  
  // Keep track of jump tables for switch instructions
  MachineJumpTableInfo *JumpTableInfo;

  // Function-level unique numbering for MachineBasicBlocks.  When a
  // MachineBasicBlock is inserted into a MachineFunction is it automatically
  // numbered and this vector keeps track of the mapping from ID's to MBB's.
  std::vector<MachineBasicBlock*> MBBNumbering;

  /// UsedPhysRegs - This is a new[]'d array of bools that is computed and set
  /// by the register allocator, and must be kept up to date by passes that run
  /// after register allocation (though most don't modify this).  This is used
  /// so that the code generator knows which callee save registers to save and
  /// for other target specific uses.
  bool *UsedPhysRegs;

  /// LiveIns/LiveOuts - Keep track of the physical registers that are
  /// livein/liveout of the function.  Live in values are typically arguments in
  /// registers, live out values are typically return values in registers.
  /// LiveIn values are allowed to have virtual registers associated with them,
  /// stored in the second element.
  std::vector<std::pair<unsigned, unsigned> > LiveIns;
  std::vector<unsigned> LiveOuts;
  
public:
  MachineFunction(const Function *Fn, const TargetMachine &TM);
  ~MachineFunction();

  /// getFunction - Return the LLVM function that this machine code represents
  ///
  const Function *getFunction() const { return Fn; }

  /// getTarget - Return the target machine this machine code is compiled with
  ///
  const TargetMachine &getTarget() const { return Target; }

  /// SSARegMap Interface... Keep track of information about each SSA virtual
  /// register, such as which register class it belongs to.
  ///
  SSARegMap *getSSARegMap() const { return SSARegMapping; }
  void clearSSARegMap();

  /// getFrameInfo - Return the frame info object for the current function.
  /// This object contains information about objects allocated on the stack
  /// frame of the current function in an abstract way.
  ///
  MachineFrameInfo *getFrameInfo() const { return FrameInfo; }

  /// getJumpTableInfo - Return the jump table info object for the current 
  /// function.  This object contains information about jump tables for switch
  /// instructions in the current function.
  ///
  MachineJumpTableInfo *getJumpTableInfo() const { return JumpTableInfo; }
  
  /// getConstantPool - Return the constant pool object for the current
  /// function.
  ///
  MachineConstantPool *getConstantPool() const { return ConstantPool; }

  /// MachineFunctionInfo - Keep track of various per-function pieces of
  /// information for backends that would like to do so.
  ///
  template<typename Ty>
  Ty *getInfo() {
    if (!MFInfo) MFInfo = new Ty(*this);

    assert((void*)dynamic_cast<Ty*>(MFInfo) == (void*)MFInfo &&
           "Invalid concrete type or multiple inheritence for getInfo");
    return static_cast<Ty*>(MFInfo);
  }

  template<typename Ty>
  const Ty *getInfo() const {
     return const_cast<MachineFunction*>(this)->getInfo<Ty>();
  }

  /// setUsedPhysRegs - The register allocator should call this to initialized
  /// the UsedPhysRegs set.  This should be passed a new[]'d array with entries
  /// for all of the physical registers that the target supports.  Each array
  /// entry should be set to true iff the physical register is used within the
  /// function.
  void setUsedPhysRegs(bool *UPR) { UsedPhysRegs = UPR; }

  /// getUsedPhysregs - This returns the UsedPhysRegs array.  This returns null
  /// before register allocation.
  bool *getUsedPhysregs() { return UsedPhysRegs; }
  const bool *getUsedPhysregs() const { return UsedPhysRegs; }

  /// isPhysRegUsed - Return true if the specified register is used in this
  /// function.  This only works after register allocation.
  bool isPhysRegUsed(unsigned Reg) { return UsedPhysRegs[Reg]; }

  /// changePhyRegUsed - This method allows code that runs after register
  /// allocation to keep the PhysRegsUsed array up-to-date.
  void changePhyRegUsed(unsigned Reg, bool State) { UsedPhysRegs[Reg] = State; }


  // LiveIn/LiveOut management methods.

  /// addLiveIn/Out - Add the specified register as a live in/out.  Note that it
  /// is an error to add the same register to the same set more than once.
  void addLiveIn(unsigned Reg, unsigned vreg = 0) {
    LiveIns.push_back(std::make_pair(Reg, vreg));
  }
  void addLiveOut(unsigned Reg) { LiveOuts.push_back(Reg); }

  // Iteration support for live in/out sets.  These sets are kept in sorted
  // order by their register number.
  typedef std::vector<std::pair<unsigned,unsigned> >::const_iterator
  livein_iterator;
  typedef std::vector<unsigned>::const_iterator liveout_iterator;
  livein_iterator livein_begin() const { return LiveIns.begin(); }
  livein_iterator livein_end()   const { return LiveIns.end(); }
  bool            livein_empty() const { return LiveIns.empty(); }
  liveout_iterator liveout_begin() const { return LiveOuts.begin(); }
  liveout_iterator liveout_end()   const { return LiveOuts.end(); }
  bool             liveout_empty() const { return LiveOuts.empty(); }

  /// getBlockNumbered - MachineBasicBlocks are automatically numbered when they
  /// are inserted into the machine function.  The block number for a machine
  /// basic block can be found by using the MBB::getBlockNumber method, this
  /// method provides the inverse mapping.
  ///
  MachineBasicBlock *getBlockNumbered(unsigned N) {
    assert(N < MBBNumbering.size() && "Illegal block number");
    assert(MBBNumbering[N] && "Block was removed from the machine function!");
    return MBBNumbering[N];
  }

  /// getLastBlock - Returns the MachineBasicBlock with the greatest number
  MachineBasicBlock *getLastBlock() {
    return MBBNumbering.back();
  }
  const MachineBasicBlock *getLastBlock() const {
    return MBBNumbering.back();
  }
  
  /// print - Print out the MachineFunction in a format suitable for debugging
  /// to the specified stream.
  ///
  void print(std::ostream &OS) const;

  /// viewCFG - This function is meant for use from the debugger.  You can just
  /// say 'call F->viewCFG()' and a ghostview window should pop up from the
  /// program, displaying the CFG of the current function with the code for each
  /// basic block inside.  This depends on there being a 'dot' and 'gv' program
  /// in your path.
  ///
  void viewCFG() const;

  /// viewCFGOnly - This function is meant for use from the debugger.  It works
  /// just like viewCFG, but it does not include the contents of basic blocks
  /// into the nodes, just the label.  If you are only interested in the CFG
  /// this can make the graph smaller.
  ///
  void viewCFGOnly() const;

  /// dump - Print the current MachineFunction to cerr, useful for debugger use.
  ///
  void dump() const;

  /// construct - Allocate and initialize a MachineFunction for a given Function
  /// and Target
  ///
  static MachineFunction& construct(const Function *F, const TargetMachine &TM);

  /// destruct - Destroy the MachineFunction corresponding to a given Function
  ///
  static void destruct(const Function *F);

  /// get - Return a handle to a MachineFunction corresponding to the given
  /// Function.  This should not be called before "construct()" for a given
  /// Function.
  ///
  static MachineFunction& get(const Function *F);

  // Provide accessors for the MachineBasicBlock list...
  typedef ilist<MachineBasicBlock> BasicBlockListType;
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  // Provide accessors for basic blocks...
  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }

  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  unsigned                  size() const { return BasicBlocks.size(); }
  bool                     empty() const { return BasicBlocks.empty(); }
  const MachineBasicBlock &front() const { return BasicBlocks.front(); }
        MachineBasicBlock &front()       { return BasicBlocks.front(); }
  const MachineBasicBlock & back() const { return BasicBlocks.back(); }
        MachineBasicBlock & back()       { return BasicBlocks.back(); }

  //===--------------------------------------------------------------------===//
  // Internal functions used to automatically number MachineBasicBlocks
  //

  /// getNextMBBNumber - Returns the next unique number to be assigned
  /// to a MachineBasicBlock in this MachineFunction.
  ///
  unsigned addToMBBNumbering(MachineBasicBlock *MBB) {
    MBBNumbering.push_back(MBB);
    return MBBNumbering.size()-1;
  }

  /// removeFromMBBNumbering - Remove the specific machine basic block from our
  /// tracker, this is only really to be used by the MachineBasicBlock
  /// implementation.
  void removeFromMBBNumbering(unsigned N) {
    assert(N < MBBNumbering.size() && "Illegal basic block #");
    MBBNumbering[N] = 0;
  }
};

//===--------------------------------------------------------------------===//
// GraphTraits specializations for function basic block graphs (CFGs)
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a
// machine function as a graph of machine basic blocks... these are
// the same as the machine basic block iterators, except that the root
// node is implicitly the first node of the function.
//
template <> struct GraphTraits<MachineFunction*> :
  public GraphTraits<MachineBasicBlock*> {
  static NodeType *getEntryNode(MachineFunction *F) {
    return &F->front();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef MachineFunction::iterator nodes_iterator;
  static nodes_iterator nodes_begin(MachineFunction *F) { return F->begin(); }
  static nodes_iterator nodes_end  (MachineFunction *F) { return F->end(); }
};
template <> struct GraphTraits<const MachineFunction*> :
  public GraphTraits<const MachineBasicBlock*> {
  static NodeType *getEntryNode(const MachineFunction *F) {
    return &F->front();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef MachineFunction::const_iterator nodes_iterator;
  static nodes_iterator nodes_begin(const MachineFunction *F) { return F->begin(); }
  static nodes_iterator nodes_end  (const MachineFunction *F) { return F->end(); }
};


// Provide specializations of GraphTraits to be able to treat a function as a
// graph of basic blocks... and to walk it in inverse order.  Inverse order for
// a function is considered to be when traversing the predecessor edges of a BB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<MachineFunction*> > :
  public GraphTraits<Inverse<MachineBasicBlock*> > {
  static NodeType *getEntryNode(Inverse<MachineFunction*> G) {
    return &G.Graph->front();
  }
};
template <> struct GraphTraits<Inverse<const MachineFunction*> > :
  public GraphTraits<Inverse<const MachineBasicBlock*> > {
  static NodeType *getEntryNode(Inverse<const MachineFunction *> G) {
    return &G.Graph->front();
  }
};

} // End llvm namespace

#endif
