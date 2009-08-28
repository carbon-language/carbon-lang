//===-- llvm/CodeGen/MachineFunction.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Recycler.h"
#include <map>

namespace llvm {

class Function;
class MachineRegisterInfo;
class MachineFrameInfo;
class MachineConstantPool;
class MachineJumpTableInfo;
class TargetMachine;
class TargetRegisterClass;

template <>
struct ilist_traits<MachineBasicBlock>
    : public ilist_default_traits<MachineBasicBlock> {
  mutable ilist_half_node<MachineBasicBlock> Sentinel;
public:
  MachineBasicBlock *createSentinel() const {
    return static_cast<MachineBasicBlock*>(&Sentinel);
  }
  void destroySentinel(MachineBasicBlock *) const {}

  MachineBasicBlock *provideInitialHead() const { return createSentinel(); }
  MachineBasicBlock *ensureHead(MachineBasicBlock*) const {
    return createSentinel();
  }
  static void noteHead(MachineBasicBlock*, MachineBasicBlock*) {}

  void addNodeToList(MachineBasicBlock* MBB);
  void removeNodeFromList(MachineBasicBlock* MBB);
  void deleteNode(MachineBasicBlock *MBB);
private:
  void createNode(const MachineBasicBlock &);
};

/// MachineFunctionInfo - This class can be derived from and used by targets to
/// hold private target-specific information for each MachineFunction.  Objects
/// of type are accessed/created with MF::getInfo and destroyed when the
/// MachineFunction is destroyed.
struct MachineFunctionInfo {
  virtual ~MachineFunctionInfo() {}
};

class MachineFunction {
  Function *Fn;
  const TargetMachine &Target;

  // RegInfo - Information about each register in use in the function.
  MachineRegisterInfo *RegInfo;

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

  // Pool-allocate MachineFunction-lifetime and IR objects.
  BumpPtrAllocator Allocator;

  // Allocation management for instructions in function.
  Recycler<MachineInstr> InstructionRecycler;

  // Allocation management for basic blocks in function.
  Recycler<MachineBasicBlock> BasicBlockRecycler;

  // List of machine basic blocks in function
  typedef ilist<MachineBasicBlock> BasicBlockListType;
  BasicBlockListType BasicBlocks;

  // Default debug location. Used to print out the debug label at the beginning
  // of a function.
  DebugLoc DefaultDebugLoc;

  // Tracks debug locations.
  DebugLocTracker DebugLocInfo;

  // The alignment of the function.
  unsigned Alignment;

public:
  MachineFunction(Function *Fn, const TargetMachine &TM);
  ~MachineFunction();

  /// getFunction - Return the LLVM function that this machine code represents
  ///
  Function *getFunction() const { return Fn; }

  /// getTarget - Return the target machine this machine code is compiled with
  ///
  const TargetMachine &getTarget() const { return Target; }

  /// getRegInfo - Return information about the registers currently in use.
  ///
  MachineRegisterInfo &getRegInfo() { return *RegInfo; }
  const MachineRegisterInfo &getRegInfo() const { return *RegInfo; }

  /// getFrameInfo - Return the frame info object for the current function.
  /// This object contains information about objects allocated on the stack
  /// frame of the current function in an abstract way.
  ///
  MachineFrameInfo *getFrameInfo() { return FrameInfo; }
  const MachineFrameInfo *getFrameInfo() const { return FrameInfo; }

  /// getJumpTableInfo - Return the jump table info object for the current 
  /// function.  This object contains information about jump tables for switch
  /// instructions in the current function.
  ///
  MachineJumpTableInfo *getJumpTableInfo() { return JumpTableInfo; }
  const MachineJumpTableInfo *getJumpTableInfo() const { return JumpTableInfo; }
  
  /// getConstantPool - Return the constant pool object for the current
  /// function.
  ///
  MachineConstantPool *getConstantPool() { return ConstantPool; }
  const MachineConstantPool *getConstantPool() const { return ConstantPool; }

  /// getAlignment - Return the alignment (log2, not bytes) of the function.
  ///
  unsigned getAlignment() const { return Alignment; }

  /// setAlignment - Set the alignment (log2, not bytes) of the function.
  ///
  void setAlignment(unsigned A) { Alignment = A; }

  /// MachineFunctionInfo - Keep track of various per-function pieces of
  /// information for backends that would like to do so.
  ///
  template<typename Ty>
  Ty *getInfo() {
    if (!MFInfo) {
        // This should be just `new (Allocator.Allocate<Ty>()) Ty(*this)', but
        // that apparently breaks GCC 3.3.
        Ty *Loc = static_cast<Ty*>(Allocator.Allocate(sizeof(Ty),
                                                      AlignOf<Ty>::Alignment));
        MFInfo = new (Loc) Ty(*this);
    }

    assert((void*)dynamic_cast<Ty*>(MFInfo) == (void*)MFInfo &&
           "Invalid concrete type or multiple inheritence for getInfo");
    return static_cast<Ty*>(MFInfo);
  }

  template<typename Ty>
  const Ty *getInfo() const {
     return const_cast<MachineFunction*>(this)->getInfo<Ty>();
  }

  /// getBlockNumbered - MachineBasicBlocks are automatically numbered when they
  /// are inserted into the machine function.  The block number for a machine
  /// basic block can be found by using the MBB::getBlockNumber method, this
  /// method provides the inverse mapping.
  ///
  MachineBasicBlock *getBlockNumbered(unsigned N) const {
    assert(N < MBBNumbering.size() && "Illegal block number");
    assert(MBBNumbering[N] && "Block was removed from the machine function!");
    return MBBNumbering[N];
  }

  /// getNumBlockIDs - Return the number of MBB ID's allocated.
  ///
  unsigned getNumBlockIDs() const { return (unsigned)MBBNumbering.size(); }
  
  /// RenumberBlocks - This discards all of the MachineBasicBlock numbers and
  /// recomputes them.  This guarantees that the MBB numbers are sequential,
  /// dense, and match the ordering of the blocks within the function.  If a
  /// specific MachineBasicBlock is specified, only that block and those after
  /// it are renumbered.
  void RenumberBlocks(MachineBasicBlock *MBBFrom = 0);
  
  /// print - Print out the MachineFunction in a format suitable for debugging
  /// to the specified stream.
  ///
  void print(raw_ostream &OS) const;

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

  // Provide accessors for the MachineBasicBlock list...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  /// addLiveIn - Add the specified physical register as a live-in value and
  /// create a corresponding virtual register for it.
  unsigned addLiveIn(unsigned PReg, const TargetRegisterClass *RC);

  //===--------------------------------------------------------------------===//
  // BasicBlock accessor functions.
  //
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  unsigned                  size() const { return (unsigned)BasicBlocks.size();}
  bool                     empty() const { return BasicBlocks.empty(); }
  const MachineBasicBlock &front() const { return BasicBlocks.front(); }
        MachineBasicBlock &front()       { return BasicBlocks.front(); }
  const MachineBasicBlock & back() const { return BasicBlocks.back(); }
        MachineBasicBlock & back()       { return BasicBlocks.back(); }

  void push_back (MachineBasicBlock *MBB) { BasicBlocks.push_back (MBB); }
  void push_front(MachineBasicBlock *MBB) { BasicBlocks.push_front(MBB); }
  void insert(iterator MBBI, MachineBasicBlock *MBB) {
    BasicBlocks.insert(MBBI, MBB);
  }
  void splice(iterator InsertPt, iterator MBBI) {
    BasicBlocks.splice(InsertPt, BasicBlocks, MBBI);
  }

  void remove(iterator MBBI) {
    BasicBlocks.remove(MBBI);
  }
  void erase(iterator MBBI) {
    BasicBlocks.erase(MBBI);
  }

  //===--------------------------------------------------------------------===//
  // Internal functions used to automatically number MachineBasicBlocks
  //

  /// getNextMBBNumber - Returns the next unique number to be assigned
  /// to a MachineBasicBlock in this MachineFunction.
  ///
  unsigned addToMBBNumbering(MachineBasicBlock *MBB) {
    MBBNumbering.push_back(MBB);
    return (unsigned)MBBNumbering.size()-1;
  }

  /// removeFromMBBNumbering - Remove the specific machine basic block from our
  /// tracker, this is only really to be used by the MachineBasicBlock
  /// implementation.
  void removeFromMBBNumbering(unsigned N) {
    assert(N < MBBNumbering.size() && "Illegal basic block #");
    MBBNumbering[N] = 0;
  }

  /// CreateMachineInstr - Allocate a new MachineInstr. Use this instead
  /// of `new MachineInstr'.
  ///
  MachineInstr *CreateMachineInstr(const TargetInstrDesc &TID,
                                   DebugLoc DL,
                                   bool NoImp = false);

  /// CloneMachineInstr - Create a new MachineInstr which is a copy of the
  /// 'Orig' instruction, identical in all ways except the the instruction
  /// has no parent, prev, or next.
  ///
  MachineInstr *CloneMachineInstr(const MachineInstr *Orig);

  /// DeleteMachineInstr - Delete the given MachineInstr.
  ///
  void DeleteMachineInstr(MachineInstr *MI);

  /// CreateMachineBasicBlock - Allocate a new MachineBasicBlock. Use this
  /// instead of `new MachineBasicBlock'.
  ///
  MachineBasicBlock *CreateMachineBasicBlock(const BasicBlock *bb = 0);

  /// DeleteMachineBasicBlock - Delete the given MachineBasicBlock.
  ///
  void DeleteMachineBasicBlock(MachineBasicBlock *MBB);

  //===--------------------------------------------------------------------===//
  // Debug location.
  //

  /// getOrCreateDebugLocID - Look up the DebugLocTuple index with the given
  /// source file, line, and column. If none currently exists, create a new
  /// DebugLocTuple, and insert it into the DebugIdMap.
  unsigned getOrCreateDebugLocID(MDNode *CompileUnit,
                                 unsigned Line, unsigned Col);

  /// getDebugLocTuple - Get the DebugLocTuple for a given DebugLoc object.
  DebugLocTuple getDebugLocTuple(DebugLoc DL) const;

  /// getDefaultDebugLoc - Get the default debug location for the machine
  /// function.
  DebugLoc getDefaultDebugLoc() const { return DefaultDebugLoc; }

  /// setDefaultDebugLoc - Get the default debug location for the machine
  /// function.
  void setDefaultDebugLoc(DebugLoc DL) { DefaultDebugLoc = DL; }

  /// getDebugLocInfo - Get the debug info location tracker.
  DebugLocTracker &getDebugLocInfo() { return DebugLocInfo; }
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
  static nodes_iterator nodes_begin(const MachineFunction *F) {
    return F->begin();
  }
  static nodes_iterator nodes_end  (const MachineFunction *F) {
    return F->end();
  }
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
