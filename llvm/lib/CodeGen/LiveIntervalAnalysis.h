//===-- LiveIntervalAnalysis.h - Live Interval Analysis ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass.  Given some numbering of
// each the machine instructions (in this implemention depth-first order) an
// interval [i, j) is said to be a live interval for register v if there is no
// instruction with number j' > j such that v is live at j' abd there is no
// instruction with number i' < i such that v is live at i'. In this
// implementation intervals can have holes, i.e. an interval might look like
// [1,20), [50,65), [1000,1001).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_ANALYSIS_H
#define LLVM_CODEGEN_LIVEINTERVAL_ANALYSIS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "LiveInterval.h"
#include <list>

namespace llvm {

    class LiveVariables;
    class MRegisterInfo;
    class VirtRegMap;

    class LiveIntervals : public MachineFunctionPass
    {
    public:
        typedef std::list<LiveInterval> Intervals;

    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        MachineBasicBlock* currentMbb_;
        MachineBasicBlock::iterator currentInstr_;
        LiveVariables* lv_;

        typedef std::map<MachineInstr*, unsigned> Mi2IndexMap;
        Mi2IndexMap mi2iMap_;

        typedef std::vector<MachineInstr*> Index2MiMap;
        Index2MiMap i2miMap_;

        typedef std::map<unsigned, Intervals::iterator> Reg2IntervalMap;
        Reg2IntervalMap r2iMap_;

        typedef std::map<unsigned, unsigned> Reg2RegMap;
        Reg2RegMap r2rMap_;

        Intervals intervals_;

    public:
        struct InstrSlots
        {
            enum {
                LOAD  = 0,
                USE   = 1,
                DEF   = 2,
                STORE = 3,
                NUM   = 4,
            };
        };

        static unsigned getBaseIndex(unsigned index) {
            return index - (index % InstrSlots::NUM);
        }
        static unsigned getBoundaryIndex(unsigned index) {
            return getBaseIndex(index + InstrSlots::NUM - 1);
        }
        static unsigned getLoadIndex(unsigned index) {
            return getBaseIndex(index) + InstrSlots::LOAD;
        }
        static unsigned getUseIndex(unsigned index) {
            return getBaseIndex(index) + InstrSlots::USE;
        }
        static unsigned getDefIndex(unsigned index) {
            return getBaseIndex(index) + InstrSlots::DEF;
        }
        static unsigned getStoreIndex(unsigned index) {
            return getBaseIndex(index) + InstrSlots::STORE;
        }

        virtual void getAnalysisUsage(AnalysisUsage &AU) const;
        virtual void releaseMemory();

        /// runOnMachineFunction - pass entry point
        virtual bool runOnMachineFunction(MachineFunction&);

        LiveInterval& getInterval(unsigned reg) {
            assert(r2iMap_.count(reg)&& "Interval does not exist for register");
            return *r2iMap_.find(reg)->second;
        }

        /// getInstructionIndex - returns the base index of instr
        unsigned getInstructionIndex(MachineInstr* instr) const;

        /// getInstructionFromIndex - given an index in any slot of an
        /// instruction return a pointer the instruction
        MachineInstr* getInstructionFromIndex(unsigned index) const;

        Intervals& getIntervals() { return intervals_; }

        std::vector<LiveInterval*> addIntervalsForSpills(const LiveInterval& i,
                                                         VirtRegMap& vrm,
                                                         int slot);

    private:
        /// computeIntervals - compute live intervals
        void computeIntervals();

        /// joinIntervals - join compatible live intervals
        void joinIntervals();

        /// joinIntervalsInMachineBB - Join intervals based on move
        /// instructions in the specified basic block.
        void joinIntervalsInMachineBB(MachineBasicBlock *MBB);

        /// handleRegisterDef - update intervals for a register def
        /// (calls handlePhysicalRegisterDef and
        /// handleVirtualRegisterDef)
        void handleRegisterDef(MachineBasicBlock* mbb,
                               MachineBasicBlock::iterator mi,
                               unsigned reg);

        /// handleVirtualRegisterDef - update intervals for a virtual
        /// register def
        void handleVirtualRegisterDef(MachineBasicBlock* mbb,
                                      MachineBasicBlock::iterator mi,
                                      LiveInterval& interval);

        /// handlePhysicalRegisterDef - update intervals for a
        /// physical register def
        void handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                       MachineBasicBlock::iterator mi,
                                       LiveInterval& interval);

        bool overlapsAliases(const LiveInterval& lhs, 
                             const LiveInterval& rhs) const;


        LiveInterval& getOrCreateInterval(unsigned reg);

        /// rep - returns the representative of this register
        unsigned rep(unsigned reg);

        void printRegName(unsigned reg) const;
    };

} // End llvm namespace

#endif
