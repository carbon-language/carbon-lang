//===-- llvm/CodeGen/RenderMachineFunction.h - MF->HTML -*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_RENDERMACHINEFUNCTION_H
#define LLVM_CODEGEN_RENDERMACHINEFUNCTION_H

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Target/TargetRegisterInfo.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>

namespace llvm {

  class LiveInterval;
  class LiveIntervals;
  class MachineInstr;
  class MachineRegisterInfo;
  class TargetRegisterClass;
  class TargetRegisterInfo;
  class VirtRegMap;
  class raw_ostream;

  /// \brief Provide extra information about the physical and virtual registers
  ///        in the function being compiled.
  class TargetRegisterExtraInfo {
  public:
    TargetRegisterExtraInfo();

    /// \brief Set up TargetRegisterExtraInfo with pointers to necessary
    ///        sources of information.
    void setup(MachineFunction *mf, MachineRegisterInfo *mri,
               const TargetRegisterInfo *tri, LiveIntervals *lis);

    /// \brief Recompute tables for changed function.
    void reset(); 

    /// \brief Free all tables in TargetRegisterExtraInfo.
    void clear();

    /// \brief Maximum number of registers from trc which alias reg.
    unsigned getWorst(unsigned reg, const TargetRegisterClass *trc) const;

    /// \brief Returns the number of allocable registers in trc.
    unsigned getCapacity(const TargetRegisterClass *trc) const;

    /// \brief Return the number of registers of class trc that may be
    ///        needed at slot i.
    unsigned getPressureAtSlot(const TargetRegisterClass *trc,
                               SlotIndex i) const;

    /// \brief Return true if the number of registers of type trc that may be
    ///        needed at slot i is greater than the capacity of trc.
    bool classOverCapacityAtSlot(const TargetRegisterClass *trc,
                                 SlotIndex i) const;

  private:

    MachineFunction *mf;
    MachineRegisterInfo *mri;
    const TargetRegisterInfo *tri;
    LiveIntervals *lis;

    typedef std::map<const TargetRegisterClass*, unsigned> WorstMapLine;
    typedef std::map<const TargetRegisterClass*, WorstMapLine> VRWorstMap;
    VRWorstMap vrWorst;

    typedef std::map<unsigned, WorstMapLine> PRWorstMap;
    PRWorstMap prWorst;

    typedef std::map<const TargetRegisterClass*, unsigned> CapacityMap;
    CapacityMap capacityMap;

    typedef std::map<const TargetRegisterClass*, unsigned> PressureMapLine;
    typedef std::map<SlotIndex, PressureMapLine> PressureMap;
    PressureMap pressureMap;

    bool mapsPopulated;

    /// \brief Initialise the 'worst' table.
    void initWorst();
 
    /// \brief Initialise the 'capacity' table.
    void initCapacity();

    /// \brief Initialise/Reset the 'pressure' and live states tables.
    void resetPressureAndLiveStates();
  };

  /// \brief Helper class to process rendering options. Tries to be as lazy as
  ///        possible.
  class MFRenderingOptions {
  public:

    struct RegClassComp {
      bool operator()(const TargetRegisterClass *trc1,
                      const TargetRegisterClass *trc2) const {
        std::string trc1Name(trc1->getName()), trc2Name(trc2->getName());
        return std::lexicographical_compare(trc1Name.begin(), trc1Name.end(),
                                            trc2Name.begin(), trc2Name.end());
      }
    };

    typedef std::set<const TargetRegisterClass*, RegClassComp> RegClassSet;

    struct IntervalComp {
      bool operator()(const LiveInterval *li1, const LiveInterval *li2) const {
        return li1->reg < li2->reg;
      }
    };

    typedef std::set<const LiveInterval*, IntervalComp> IntervalSet;

    /// Initialise the rendering options.
    void setup(MachineFunction *mf, const TargetRegisterInfo *tri,
               LiveIntervals *lis);

    /// Clear translations of options to the current function.
    void clear();

    /// Reset any options computed for this specific rendering.
    void resetRenderSpecificOptions();

    /// Should we render the current function.
    bool shouldRenderCurrentMachineFunction() const;

    /// Return the set of register classes to render pressure for.
    const RegClassSet& regClasses() const;

    /// Return the set of live intervals to render liveness for.
    const IntervalSet& intervals() const;

    /// Render indexes which are not associated with instructions / MBB starts.
    bool renderEmptyIndexes() const;

    /// Return whether or not to render using SVG for fancy vertical text.
    bool fancyVerticals() const;

  private:

    static bool renderingOptionsProcessed;
    static std::set<std::string> mfNamesToRender;
    static bool renderAllMFs;

    static std::set<std::string> classNamesToRender;
    static bool renderAllClasses;


    static std::set<std::pair<unsigned, unsigned> > intervalNumsToRender;
    typedef enum { ExplicitOnly     = 0,
                   VirtPlusExplicit = 1,
                   PhysPlusExplicit = 2,
                   All              = 3 }
      IntervalTypesToRender;
    static unsigned intervalTypesToRender;

    template <typename OutputItr>
    static void splitComaSeperatedList(const std::string &s, OutputItr outItr);

    static void processOptions();

    static void processFuncNames();
    static void processRegClassNames();
    static void processIntervalNumbers();

    static void processIntervalRange(const std::string &intervalRangeStr);

    MachineFunction *mf;
    const TargetRegisterInfo *tri;
    LiveIntervals *lis;

    mutable bool regClassesTranslatedToCurrentFunction;
    mutable RegClassSet regClassSet;

    mutable bool intervalsTranslatedToCurrentFunction;
    mutable IntervalSet intervalSet;

    void translateRegClassNamesToCurrentFunction() const;

    void translateIntervalNumbersToCurrentFunction() const;
  };

  /// \brief Render MachineFunction objects and related information to a HTML
  ///        page.
  class RenderMachineFunction : public MachineFunctionPass {
  public:
    static char ID;

    RenderMachineFunction() : MachineFunctionPass(ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &au) const;

    virtual bool runOnMachineFunction(MachineFunction &fn);

    virtual void releaseMemory();

    /// \brief Render this machine function to HTML.
    /// 
    /// @param renderContextStr This parameter will be included in the top of
    ///                         the html file to explain where (in the
    ///                         codegen pipeline) this function was rendered
    ///                         from. Set it to something like
    ///                         "Pre-register-allocation".
    /// @param vrm              If non-null the VRM will be queried to determine
    ///                         whether a virtual register was allocated to a
    ///                         physical register or spilled.
    /// @param renderFilePrefix This string will be appended to the function
    ///                         name (before the output file suffix) to enable
    ///                         multiple renderings from the same function.
    void renderMachineFunction(const char *renderContextStr,
                               const VirtRegMap *vrm = 0,
                               const char *renderSuffix = 0);

  private:
    class Spacer;

    friend raw_ostream& operator<<(raw_ostream &os, const Spacer &s);


    std::string fqn;

    MachineFunction *mf;
    MachineRegisterInfo *mri;
    const TargetRegisterInfo *tri;
    LiveIntervals *lis;
    SlotIndexes *sis;
    const VirtRegMap *vrm;

    TargetRegisterExtraInfo trei;
    MFRenderingOptions ro;

    // Utilities.
    typedef enum { Dead, Defined, Used, AliveReg, AliveStack } LiveState;
    LiveState getLiveStateAt(const LiveInterval *li, SlotIndex i) const;

    typedef enum { Zero, Low, High } PressureState;
    PressureState getPressureStateAt(const TargetRegisterClass *trc,
                                     SlotIndex i) const;

    // ---------- Rendering methods ----------

    /// For inserting spaces when pretty printing.
    class Spacer {
    public:
      explicit Spacer(unsigned numSpaces) : ns(numSpaces) {}
      Spacer operator+(const Spacer &o) const { return Spacer(ns + o.ns); }
      void print(raw_ostream &os) const;
    private:
      unsigned ns;
    };

    Spacer s(unsigned ns) const;

    template <typename Iterator>
    std::string escapeChars(Iterator sBegin, Iterator sEnd) const;

    /// \brief Render a machine instruction.
    void renderMachineInstr(raw_ostream &os,
                            const MachineInstr *mi) const;

    /// \brief Render vertical text.
    template <typename T>
    void renderVertical(const Spacer &indent,
                        raw_ostream &os,
                        const T &t) const;

    /// \brief Insert CSS layout info.
    void insertCSS(const Spacer &indent,
                   raw_ostream &os) const;

    /// \brief Render a brief summary of the function (including rendering
    ///        context).
    void renderFunctionSummary(const Spacer &indent,
                               raw_ostream &os,
                               const char * const renderContextStr) const;

    /// \brief Render a legend for the pressure table.
    void renderPressureTableLegend(const Spacer &indent,
                                   raw_ostream &os) const;

    /// \brief Render a consecutive set of HTML cells of the same class using
    /// the colspan attribute for run-length encoding.
    template <typename CellType>
    void renderCellsWithRLE(
                     const Spacer &indent, raw_ostream &os,
                     const std::pair<CellType, unsigned> &rleAccumulator,
                     const std::map<CellType, std::string> &cellTypeStrs) const;

    /// \brief Render code listing, potentially with register pressure
    ///        and live intervals shown alongside.
    void renderCodeTablePlusPI(const Spacer &indent,
                               raw_ostream &os) const;

    /// \brief Render the HTML page representing the MachineFunction.
    void renderFunctionPage(raw_ostream &os,
                            const char * const renderContextStr) const;

    std::string escapeChars(const std::string &s) const;
  };
}

#endif /* LLVM_CODEGEN_RENDERMACHINEFUNCTION_H */
