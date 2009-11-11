//===-- BranchFolding.h - Fold machine code branch instructions --*- C++ -*===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BRANCHFOLDING_HPP
#define LLVM_CODEGEN_BRANCHFOLDING_HPP

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <vector>

namespace llvm {
  class MachineFunction;
  class MachineModuleInfo;
  class RegScavenger;
  class TargetInstrInfo;
  class TargetRegisterInfo;

  class BranchFolder {
  public:
    explicit BranchFolder(bool defaultEnableTailMerge);

    bool OptimizeFunction(MachineFunction &MF,
                          const TargetInstrInfo *tii,
                          const TargetRegisterInfo *tri,
                          MachineModuleInfo *mmi);
  private:
    class MergePotentialsElt {
      unsigned Hash;
      MachineBasicBlock *Block;
    public:
      MergePotentialsElt(unsigned h, MachineBasicBlock *b)
        : Hash(h), Block(b) {}

      unsigned getHash() const { return Hash; }
      MachineBasicBlock *getBlock() const { return Block; }

      void setBlock(MachineBasicBlock *MBB) {
        Block = MBB;
      }

      bool operator<(const MergePotentialsElt &) const;
    };
    typedef std::vector<MergePotentialsElt>::iterator MPIterator;
    std::vector<MergePotentialsElt> MergePotentials;

    class SameTailElt {
      MPIterator MPIter;
      MachineBasicBlock::iterator TailStartPos;
    public:
      SameTailElt(MPIterator mp, MachineBasicBlock::iterator tsp)
        : MPIter(mp), TailStartPos(tsp) {}

      MPIterator getMPIter() const {
        return MPIter;
      }
      MergePotentialsElt &getMergePotentialsElt() const {
        return *getMPIter();
      }
      MachineBasicBlock::iterator getTailStartPos() const {
        return TailStartPos;
      }
      unsigned getHash() const {
        return getMergePotentialsElt().getHash();
      }
      MachineBasicBlock *getBlock() const {
        return getMergePotentialsElt().getBlock();
      }
      bool tailIsWholeBlock() const {
        return TailStartPos == getBlock()->begin();
      }

      void setBlock(MachineBasicBlock *MBB) {
        getMergePotentialsElt().setBlock(MBB);
      }
      void setTailStartPos(MachineBasicBlock::iterator Pos) {
        TailStartPos = Pos;
      }
    };
    std::vector<SameTailElt> SameTails;

    bool EnableTailMerge;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineModuleInfo *MMI;
    RegScavenger *RS;

    bool TailMergeBlocks(MachineFunction &MF);
    bool TryTailMergeBlocks(MachineBasicBlock* SuccBB,
                       MachineBasicBlock* PredBB);
    void ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                 MachineBasicBlock *NewDest);
    MachineBasicBlock *SplitMBBAt(MachineBasicBlock &CurMBB,
                                  MachineBasicBlock::iterator BBI1);
    unsigned ComputeSameTails(unsigned CurHash, unsigned minCommonTailLength,
                              MachineBasicBlock *SuccBB,
                              MachineBasicBlock *PredBB);
    void RemoveBlocksWithHash(unsigned CurHash, MachineBasicBlock* SuccBB,
                                                MachineBasicBlock* PredBB);
    unsigned CreateCommonTailOnlyBlock(MachineBasicBlock *&PredBB,
                                       unsigned maxCommonTailLength);

    bool TailDuplicate(MachineBasicBlock *TailBB,
                       bool PrevFallsThrough,
                       MachineFunction &MF);
    
    bool OptimizeBranches(MachineFunction &MF);
    bool OptimizeBlock(MachineBasicBlock *MBB);
    void RemoveDeadBlock(MachineBasicBlock *MBB);
    bool OptimizeImpDefsBlock(MachineBasicBlock *MBB);
    
    bool CanFallThrough(MachineBasicBlock *CurBB);
    bool CanFallThrough(MachineBasicBlock *CurBB, bool BranchUnAnalyzable,
                        MachineBasicBlock *TBB, MachineBasicBlock *FBB,
                        const SmallVectorImpl<MachineOperand> &Cond);
  };


  /// BranchFolderPass - Wrap branch folder in a machine function pass.
  class BranchFolderPass : public MachineFunctionPass,
                           public BranchFolder {
  public:
    static char ID;
    explicit BranchFolderPass(bool defaultEnableTailMerge)
      :  MachineFunctionPass(&ID), BranchFolder(defaultEnableTailMerge) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "Control Flow Optimizer"; }
  };
}

#endif /* LLVM_CODEGEN_BRANCHFOLDING_HPP */
