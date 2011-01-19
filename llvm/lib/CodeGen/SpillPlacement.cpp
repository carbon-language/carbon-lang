//===-- SpillPlacement.cpp - Optimal Spill Code Placement -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the spill code placement analysis.
//
// Each edge bundle corresponds to a node in a Hopfield network. Constraints on
// basic blocks are weighted by the block frequency and added to become the node
// bias.
//
// Transparent basic blocks have the variable live through, but don't care if it
// is spilled or in a register. These blocks become connections in the Hopfield
// network, again weighted by block frequency.
//
// The Hopfield network minimizes (possibly locally) its energy function:
//
//   E = -sum_n V_n * ( B_n + sum_{n, m linked by b} V_m * F_b )
//
// The energy function represents the expected spill code execution frequency,
// or the cost of spilling. This is a Lyapunov function which never increases
// when a node is updated. It is guaranteed to converge to a local minimum.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "spillplacement"
#include "SpillPlacement.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

using namespace llvm;

char SpillPlacement::ID = 0;
INITIALIZE_PASS_BEGIN(SpillPlacement, "spill-code-placement",
                      "Spill Code Placement Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(EdgeBundles)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(SpillPlacement, "spill-code-placement",
                    "Spill Code Placement Analysis", true, true)

char &llvm::SpillPlacementID = SpillPlacement::ID;

void SpillPlacement::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<EdgeBundles>();
  AU.addRequiredTransitive<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// Node - Each edge bundle corresponds to a Hopfield node.
///
/// The node contains precomputed frequency data that only depends on the CFG,
/// but Bias and Links are computed each time placeSpills is called.
///
/// The node Value is positive when the variable should be in a register. The
/// value can change when linked nodes change, but convergence is very fast
/// because all weights are positive.
///
struct SpillPlacement::Node {
  /// Frequency - Total block frequency feeding into[0] or out of[1] the bundle.
  /// Ideally, these two numbers should be identical, but inaccuracies in the
  /// block frequency estimates means that we need to normalize ingoing and
  /// outgoing frequencies separately so they are commensurate.
  float Frequency[2];

  /// Bias - Normalized contributions from non-transparent blocks.
  /// A bundle connected to a MustSpill block has a huge negative bias,
  /// otherwise it is a number in the range [-2;2].
  float Bias;

  /// Value - Output value of this node computed from the Bias and links.
  /// This is always in the range [-1;1]. A positive number means the variable
  /// should go in a register through this bundle.
  float Value;

  typedef SmallVector<std::pair<float, unsigned>, 4> LinkVector;

  /// Links - (Weight, BundleNo) for all transparent blocks connecting to other
  /// bundles. The weights are all positive and add up to at most 2, weights
  /// from ingoing and outgoing nodes separately add up to a most 1. The weight
  /// sum can be less than 2 when the variable is not live into / out of some
  /// connected basic blocks.
  LinkVector Links;

  /// preferReg - Return true when this node prefers to be in a register.
  bool preferReg() const {
    // Undecided nodes (Value==0) go on the stack.
    return Value > 0;
  }

  /// mustSpill - Return True if this node is so biased that it must spill.
  bool mustSpill() const {
    // Actually, we must spill if Bias < sum(weights).
    // It may be worth it to compute the weight sum here?
    return Bias < -2.0f;
  }

  /// Node - Create a blank Node.
  Node() {
    Frequency[0] = Frequency[1] = 0;
  }

  /// clear - Reset per-query data, but preserve frequencies that only depend on
  // the CFG.
  void clear() {
    Bias = Value = 0;
    Links.clear();
  }

  /// addLink - Add a link to bundle b with weight w.
  /// out=0 for an ingoing link, and 1 for an outgoing link.
  void addLink(unsigned b, float w, bool out) {
    // Normalize w relative to all connected blocks from that direction.
    w /= Frequency[out];

    // There can be multiple links to the same bundle, add them up.
    for (LinkVector::iterator I = Links.begin(), E = Links.end(); I != E; ++I)
      if (I->second == b) {
        I->first += w;
        return;
      }
    // This must be the first link to b.
    Links.push_back(std::make_pair(w, b));
  }

  /// addBias - Bias this node from an ingoing[0] or outgoing[1] link.
  void addBias(float w, bool out) {
    // Normalize w relative to all connected blocks from that direction.
    w /= Frequency[out];
    Bias += w;
  }

  /// update - Recompute Value from Bias and Links. Return true when node
  /// preference changes.
  bool update(const Node nodes[]) {
    // Compute the weighted sum of inputs.
    float Sum = Bias;
    for (LinkVector::iterator I = Links.begin(), E = Links.end(); I != E; ++I)
      Sum += I->first * nodes[I->second].Value;

    // The weighted sum is going to be in the range [-2;2]. Ideally, we should
    // simply set Value = sign(Sum), but we will add a dead zone around 0 for
    // two reasons:
    //  1. It avoids arbitrary bias when all links are 0 as is possible during
    //     initial iterations.
    //  2. It helps tame rounding errors when the links nominally sum to 0.
    const float Thres = 1e-4;
    bool Before = preferReg();
    if (Sum < -Thres)
      Value = -1;
    else if (Sum > Thres)
      Value = 1;
    else
      Value = 0;
    return Before != preferReg();
  }
};

bool SpillPlacement::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  bundles = &getAnalysis<EdgeBundles>();
  loops = &getAnalysis<MachineLoopInfo>();

  assert(!nodes && "Leaking node array");
  nodes = new Node[bundles->getNumBundles()];

  // Compute total ingoing and outgoing block frequencies for all bundles.
  for (MachineFunction::iterator I = mf.begin(), E = mf.end(); I != E; ++I) {
    float Freq = getBlockFrequency(I);
    unsigned Num = I->getNumber();
    nodes[bundles->getBundle(Num, 1)].Frequency[0] += Freq;
    nodes[bundles->getBundle(Num, 0)].Frequency[1] += Freq;
  }

  // We never change the function.
  return false;
}

void SpillPlacement::releaseMemory() {
  delete[] nodes;
  nodes = 0;
}

/// activate - mark node n as active if it wasn't already.
void SpillPlacement::activate(unsigned n) {
  if (ActiveNodes->test(n))
    return;
  ActiveNodes->set(n);
  nodes[n].clear();
}


/// prepareNodes - Compute node biases and weights from a set of constraints.
/// Set a bit in NodeMask for each active node.
void SpillPlacement::
prepareNodes(const SmallVectorImpl<BlockConstraint> &LiveBlocks) {
  DEBUG(dbgs() << "Building Hopfield network from " << LiveBlocks.size()
               << " constraint blocks:\n");
  for (SmallVectorImpl<BlockConstraint>::const_iterator I = LiveBlocks.begin(),
       E = LiveBlocks.end(); I != E; ++I) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(I->Number);
    float Freq = getBlockFrequency(MBB);
    DEBUG(dbgs() << "  BB#" << I->Number << format(", Freq = %.1f", Freq));

    // Is this a transparent block? Link ingoing and outgoing bundles.
    if (I->Entry == DontCare && I->Exit == DontCare) {
      unsigned ib = bundles->getBundle(I->Number, 0);
      unsigned ob = bundles->getBundle(I->Number, 1);
      DEBUG(dbgs() << ", transparent EB#" << ib << " -> EB#" << ob << '\n');

      // Ignore self-loops.
      if (ib == ob)
        continue;
      activate(ib);
      activate(ob);
      nodes[ib].addLink(ob, Freq, 1);
      nodes[ob].addLink(ib, Freq, 0);
      continue;
    }

    // This block is not transparent, but it can still add bias.
    const float Bias[] = {
      0,           // DontCare,
      1,           // PrefReg,
      -1,          // PrefSpill
      -HUGE_VALF   // MustSpill
    };

    // Live-in to block?
    if (I->Entry != DontCare) {
      unsigned ib = bundles->getBundle(I->Number, 0);
      activate(ib);
      nodes[ib].addBias(Freq * Bias[I->Entry], 1);
      DEBUG(dbgs() << format(", entry EB#%u %+.1f", ib, Freq * Bias[I->Entry]));
    }

    // Live-out from block?
    if (I->Exit != DontCare) {
      unsigned ob = bundles->getBundle(I->Number, 1);
      activate(ob);
      nodes[ob].addBias(Freq * Bias[I->Exit], 0);
      DEBUG(dbgs() << format(", exit EB#%u %+.1f", ob, Freq * Bias[I->Exit]));
    }

    DEBUG(dbgs() << '\n');
  }
}

/// iterate - Repeatedly update the Hopfield nodes until stability or the
/// maximum number of iterations is reached.
/// @param Linked - Numbers of linked nodes that need updating.
void SpillPlacement::iterate(const SmallVectorImpl<unsigned> &Linked) {
  DEBUG(dbgs() << "Iterating over " << Linked.size() << " linked nodes:\n");
  if (Linked.empty())
    return;

  // Run up to 10 iterations. The edge bundle numbering is closely related to
  // basic block numbering, so there is a strong tendency towards chains of
  // linked nodes with sequential numbers. By scanning the linked nodes
  // backwards and forwards, we make it very likely that a single node can
  // affect the entire network in a single iteration. That means very fast
  // convergence, usually in a single iteration.
  for (unsigned iteration = 0; iteration != 10; ++iteration) {
    // Scan backwards, skipping the last node which was just updated.
    bool Changed = false;
    for (SmallVectorImpl<unsigned>::const_reverse_iterator I =
           llvm::next(Linked.rbegin()), E = Linked.rend(); I != E; ++I) {
      unsigned n = *I;
      bool C = nodes[n].update(nodes);
      Changed |= C;
      DEBUG(dbgs() << " \\EB#" << n << format(" = %+2.0f", nodes[n].Value)
                   << (C ? " *\n" : "\n"));
    }
    if (!Changed)
      return;

    // Scan forwards, skipping the first node which was just updated.
    Changed = false;
    for (SmallVectorImpl<unsigned>::const_iterator I =
           llvm::next(Linked.begin()), E = Linked.end(); I != E; ++I) {
      unsigned n = *I;
      bool C = nodes[n].update(nodes);
      Changed |= C;
      DEBUG(dbgs() << " /EB#" << n << format(" = %+2.0f", nodes[n].Value)
                   << (C ? " *\n" : "\n"));
    }
    if (!Changed)
      return;
  }
}

bool
SpillPlacement::placeSpills(const SmallVectorImpl<BlockConstraint> &LiveBlocks,
                            BitVector &RegBundles) {
  // Reuse RegBundles as our ActiveNodes vector.
  ActiveNodes = &RegBundles;
  ActiveNodes->clear();
  ActiveNodes->resize(bundles->getNumBundles());

  // Compute active nodes, links and biases.
  prepareNodes(LiveBlocks);

  // Update all active nodes, and find the ones that are actually linked to
  // something so their value may change when iterating.
  DEBUG(dbgs() << "Network has " << RegBundles.count() << " active nodes:\n");
  SmallVector<unsigned, 8> Linked;
  for (int n = RegBundles.find_first(); n>=0; n = RegBundles.find_next(n)) {
    nodes[n].update(nodes);
    // A node that must spill, or a node without any links is not going to
    // change its value ever again, so exclude it from iterations.
    if (!nodes[n].Links.empty() && !nodes[n].mustSpill())
      Linked.push_back(n);

    DEBUG({
      dbgs() << "  EB#" << n << format(" = %+2.0f", nodes[n].Value)
             << format(", Bias %+.2f", nodes[n].Bias)
             << format(", Freq %.1f/%.1f", nodes[n].Frequency[0],
                                           nodes[n].Frequency[1]);
      for (unsigned i = 0, e = nodes[n].Links.size(); i != e; ++i)
        dbgs() << format(", %.2f -> EB#%u", nodes[n].Links[i].first,
                                            nodes[n].Links[i].second);
      dbgs() << '\n';
    });
  }

  // Iterate the network to convergence.
  iterate(Linked);

  // Write preferences back to RegBundles.
  bool Perfect = true;
  for (int n = RegBundles.find_first(); n>=0; n = RegBundles.find_next(n))
    if (!nodes[n].preferReg()) {
      RegBundles.reset(n);
      Perfect = false;
    }
  return Perfect;
}

/// getBlockFrequency - Return our best estimate of the block frequency which is
/// the expected number of block executions per function invocation.
float SpillPlacement::getBlockFrequency(const MachineBasicBlock *MBB) {
  // Use the unnormalized spill weight for real block frequencies.
  return LiveIntervals::getSpillWeight(true, false, loops->getLoopDepth(MBB));
}

