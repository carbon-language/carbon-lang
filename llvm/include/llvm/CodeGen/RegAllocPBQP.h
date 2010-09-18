//===-- RegAllocPBQP.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PBQPBuilder interface, for classes which build PBQP
// instances to represent register allocation problems, and the RegAllocPBQP
// interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCPBQP_H
#define LLVM_CODEGEN_REGALLOCPBQP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/PBQP/Graph.h"
#include "llvm/CodeGen/PBQP/Solution.h"

#include <map>

namespace llvm {

  class LiveInterval;
  class MachineFunction;

  /// This class wraps up a PBQP instance representing a register allocation
  /// problem, plus the structures necessary to map back from the PBQP solution
  /// to a register allocation solution. (i.e. The PBQP-node <--> vreg map,
  /// and the PBQP option <--> storage location map).

  class PBQPRAProblem {
  public:

    typedef SmallVector<unsigned, 16> AllowedSet;

    PBQP::Graph& getGraph() { return graph; }

    const PBQP::Graph& getGraph() const { return graph; }

    /// Record the mapping between the given virtual register and PBQP node,
    /// and the set of allowed pregs for the vreg.
    ///
    /// If you are extending
    /// PBQPBuilder you are unlikely to need this: Nodes and options for all
    /// vregs will already have been set up for you by the base class. 
    template <typename AllowedRegsItr>
    void recordVReg(unsigned vreg, PBQP::Graph::NodeItr node,
                    AllowedRegsItr arBegin, AllowedRegsItr arEnd) {
      assert(node2VReg.find(node) == node2VReg.end() && "Re-mapping node.");
      assert(vreg2Node.find(vreg) == vreg2Node.end() && "Re-mapping vreg.");
      assert(allowedSets[vreg].empty() && "vreg already has pregs.");

      node2VReg[node] = vreg;
      vreg2Node[vreg] = node;
      std::copy(arBegin, arEnd, std::back_inserter(allowedSets[vreg]));
    }

    /// Get the virtual register corresponding to the given PBQP node.
    unsigned getVRegForNode(PBQP::Graph::ConstNodeItr node) const;

    /// Get the PBQP node corresponding to the given virtual register.
    PBQP::Graph::NodeItr getNodeForVReg(unsigned vreg) const;

    /// Returns true if the given PBQP option represents a physical register,
    /// false otherwise.
    bool isPRegOption(unsigned vreg, unsigned option) const {
      // At present we only have spills or pregs, so anything that's not a
      // spill is a preg. (This might be extended one day to support remat).
      return !isSpillOption(vreg, option);
    }

    /// Returns true if the given PBQP option represents spilling, false
    /// otherwise.
    bool isSpillOption(unsigned vreg, unsigned option) const {
      // We hardcode option zero as the spill option.
      return option == 0;
    }

    /// Returns the allowed set for the given virtual register.
    const AllowedSet& getAllowedSet(unsigned vreg) const;

    /// Get PReg for option.
    unsigned getPRegForOption(unsigned vreg, unsigned option) const;

  private:

    typedef std::map<PBQP::Graph::ConstNodeItr, unsigned,
                     PBQP::NodeItrComparator>  Node2VReg;
    typedef DenseMap<unsigned, PBQP::Graph::NodeItr> VReg2Node;
    typedef std::map<unsigned, AllowedSet> AllowedSetMap;

    PBQP::Graph graph;
    Node2VReg node2VReg;
    VReg2Node vreg2Node;

    AllowedSetMap allowedSets;
    
  };

  /// Builds PBQP instances to represent register allocation problems. Includes
  /// spill, interference and coalescing costs by default. You can extend this
  /// class to support additional constraints for your architecture.
  class PBQPBuilder {
  private:
    PBQPBuilder(const PBQPBuilder&) {}
    void operator=(const PBQPBuilder&) {}
  public:

    typedef std::set<unsigned> RegSet;
 

    /// Default constructor.
    PBQPBuilder() {}

    /// Clean up a PBQPBuilder.
    virtual ~PBQPBuilder() {}

    /// Build a PBQP instance to represent the register allocation problem for
    /// the given MachineFunction.
    virtual std::auto_ptr<PBQPRAProblem> build(
                                              MachineFunction *mf,
                                              const LiveIntervals *lis,
                                              const RegSet &vregs);
  private:

    void addSpillCosts(PBQP::Vector &costVec, PBQP::PBQPNum spillCost);

    void addInterferenceCosts(PBQP::Matrix &costMat,
                              const PBQPRAProblem::AllowedSet &vr1Allowed,
                              const PBQPRAProblem::AllowedSet &vr2Allowed,
                              const TargetRegisterInfo *tri);
  };

  ///
  /// PBQP based allocators solve the register allocation problem by mapping
  /// register allocation problems to Partitioned Boolean Quadratic
  /// Programming problems.
  class RegAllocPBQP : public MachineFunctionPass {
  public:

    static char ID;

    /// Construct a PBQP register allocator.
    RegAllocPBQP(std::auto_ptr<PBQPBuilder> b) : MachineFunctionPass(ID), builder(b) {}

    /// Return the pass name.
    virtual const char* getPassName() const {
      return "PBQP Register Allocator";
    }

    /// PBQP analysis usage.
    virtual void getAnalysisUsage(AnalysisUsage &au) const;

    /// Perform register allocation
    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:

    typedef std::map<const LiveInterval*, unsigned> LI2NodeMap;
    typedef std::vector<const LiveInterval*> Node2LIMap;
    typedef std::vector<unsigned> AllowedSet;
    typedef std::vector<AllowedSet> AllowedSetMap;
    typedef std::pair<unsigned, unsigned> RegPair;
    typedef std::map<RegPair, PBQP::PBQPNum> CoalesceMap;
    typedef std::vector<PBQP::Graph::NodeItr> NodeVector;
    typedef std::set<unsigned> RegSet;


    std::auto_ptr<PBQPBuilder> builder;

    MachineFunction *mf;
    const TargetMachine *tm;
    const TargetRegisterInfo *tri;
    const TargetInstrInfo *tii;
    const MachineLoopInfo *loopInfo;
    MachineRegisterInfo *mri;
    RenderMachineFunction *rmf;

    LiveIntervals *lis;
    LiveStacks *lss;
    VirtRegMap *vrm;

    LI2NodeMap li2Node;
    Node2LIMap node2LI;
    AllowedSetMap allowedSets;
    RegSet vregsToAlloc, emptyIntervalVRegs;
    NodeVector problemNodes;


    /// Builds a PBQP cost vector.
    template <typename RegContainer>
    PBQP::Vector buildCostVector(unsigned vReg,
                                 const RegContainer &allowed,
                                 const CoalesceMap &cealesces,
                                 PBQP::PBQPNum spillCost) const;

    /// \brief Builds a PBQP interference matrix.
    ///
    /// @return Either a pointer to a non-zero PBQP matrix representing the
    ///         allocation option costs, or a null pointer for a zero matrix.
    ///
    /// Expects allowed sets for two interfering LiveIntervals. These allowed
    /// sets should contain only allocable registers from the LiveInterval's
    /// register class, with any interfering pre-colored registers removed.
    template <typename RegContainer>
    PBQP::Matrix* buildInterferenceMatrix(const RegContainer &allowed1,
                                          const RegContainer &allowed2) const;

    ///
    /// Expects allowed sets for two potentially coalescable LiveIntervals,
    /// and an estimated benefit due to coalescing. The allowed sets should
    /// contain only allocable registers from the LiveInterval's register
    /// classes, with any interfering pre-colored registers removed.
    template <typename RegContainer>
    PBQP::Matrix* buildCoalescingMatrix(const RegContainer &allowed1,
                                        const RegContainer &allowed2,
                                        PBQP::PBQPNum cBenefit) const;

    /// \brief Finds coalescing opportunities and returns them as a map.
    ///
    /// Any entries in the map are guaranteed coalescable, even if their
    /// corresponding live intervals overlap.
    CoalesceMap findCoalesces();

    /// \brief Finds the initial set of vreg intervals to allocate.
    void findVRegIntervalsToAlloc();

    /// \brief Constructs a PBQP problem representation of the register
    /// allocation problem for this function.
    ///
    /// @return a PBQP solver object for the register allocation problem.
    PBQP::Graph constructPBQPProblem();

    /// \brief Adds a stack interval if the given live interval has been
    /// spilled. Used to support stack slot coloring.
    void addStackInterval(const LiveInterval *spilled,MachineRegisterInfo* mri);

    /// \brief Given a solved PBQP problem maps this solution back to a register
    /// assignment.
    bool mapPBQPToRegAlloc(const PBQP::Solution &solution);

    /// \brief Given a solved PBQP problem maps this solution back to a register
    /// assignment.
    bool mapPBQPToRegAlloc2(const PBQPRAProblem &problem,
                            const PBQP::Solution &solution);

    /// \brief Postprocessing before final spilling. Sets basic block "live in"
    /// variables.
    void finalizeAlloc() const;

  };

}

#endif /* LLVM_CODEGEN_REGALLOCPBQP_H */
