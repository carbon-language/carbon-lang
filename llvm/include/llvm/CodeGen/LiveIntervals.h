//===-- llvm/CodeGen/LiveInterval.h - Live Interval Analysis ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass.  Given some
// numbering of each the machine instructions (in this implemention
// depth-first order) an interval [i, j) is said to be a live interval
// for register v if there is no instruction with number j' > j such
// that v is live at j' abd there is no instruction with number i' < i
// such that v is live at i'. In this implementation intervals can
// have holes, i.e. an interval might look like [1,20), [50,65),
// [1000,1001)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVALS_H
#define LLVM_CODEGEN_LIVEINTERVALS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include <list>

namespace llvm {

    class LiveVariables;
    class MRegisterInfo;

    class LiveIntervals : public MachineFunctionPass
    {
    public:
        struct Interval {
            typedef std::pair<unsigned, unsigned> Range;
            typedef std::vector<Range> Ranges;
            unsigned reg;   // the register of this interval
            float weight;   // weight of this interval (number of uses
                            // * 10^loopDepth)
            Ranges ranges;  // the ranges in which this register is live

            Interval(unsigned r);

            bool empty() const { return ranges.empty(); }

            bool spilled() const;

            unsigned start() const {
                assert(!empty() && "empty interval for register");
                return ranges.front().first;
            }

            unsigned end() const {
                assert(!empty() && "empty interval for register");
                return ranges.back().second;
            }

            bool expiredAt(unsigned index) const {
                return end() <= (index + 1);
            }

            bool liveAt(unsigned index) const;

            bool overlaps(const Interval& other) const;

            void addRange(unsigned start, unsigned end);

            void join(const Interval& other);

        private:
            Ranges::iterator mergeRangesForward(Ranges::iterator it);

            Ranges::iterator mergeRangesBackward(Ranges::iterator it);
        };

        struct StartPointComp {
            bool operator()(const Interval& lhs, const Interval& rhs) {
                return lhs.ranges.front().first < rhs.ranges.front().first;
            }
        };

        struct EndPointComp {
            bool operator()(const Interval& lhs, const Interval& rhs) {
                return lhs.ranges.back().second < rhs.ranges.back().second;
            }
        };

        typedef std::list<Interval> Intervals;

    private:
        MachineFunction* mf_;
        const TargetMachine* tm_;
        const MRegisterInfo* mri_;
        MachineBasicBlock* currentMbb_;
        MachineBasicBlock::iterator currentInstr_;
        LiveVariables* lv_;

        typedef std::map<unsigned, MachineBasicBlock*> MbbIndex2MbbMap;
        MbbIndex2MbbMap mbbi2mbbMap_;

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

        Interval& getInterval(unsigned reg) {
            assert(r2iMap_.count(reg)&& "Interval does not exist for register");
            return *r2iMap_.find(reg)->second;
        }

        /// getInstructionIndex - returns the base index of instr
        unsigned getInstructionIndex(MachineInstr* instr) const;

        /// getInstructionFromIndex - given an index in any slot of an
        /// instruction return a pointer the instruction
        MachineInstr* getInstructionFromIndex(unsigned index) const;

        Intervals& getIntervals() { return intervals_; }

        void updateSpilledInterval(Interval& i, int slot);

    private:
        /// computeIntervals - compute live intervals
        void computeIntervals();

        /// joinIntervals - join compatible live intervals
        void joinIntervals();

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
                                      unsigned reg);

        /// handlePhysicalRegisterDef - update intervals for a
        /// physical register def
        void handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                       MachineBasicBlock::iterator mi,
                                       unsigned reg);

        bool overlapsAliases(const Interval& lhs, const Interval& rhs) const;

        /// rep - returns the representative of this register
        unsigned rep(unsigned reg);

        void printRegName(unsigned reg) const;
    };

    inline bool operator==(const LiveIntervals::Interval& lhs,
                           const LiveIntervals::Interval& rhs) {
        return lhs.reg == rhs.reg;
    }

    std::ostream& operator<<(std::ostream& os,
                             const LiveIntervals::Interval& li);

} // End llvm namespace

#endif
