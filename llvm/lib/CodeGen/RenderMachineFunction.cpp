//===-- llvm/CodeGen/RenderMachineFunction.cpp - MF->HTML -----s-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "rendermf"

#include "RenderMachineFunction.h"

#include "VirtRegMap.h"

#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <sstream>

using namespace llvm;

char RenderMachineFunction::ID = 0;
INITIALIZE_PASS_BEGIN(RenderMachineFunction, "rendermf",
                "Render machine functions (and related info) to HTML pages",
                false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(RenderMachineFunction, "rendermf",
                "Render machine functions (and related info) to HTML pages",
                false, false)

static cl::opt<std::string>
outputFileSuffix("rmf-file-suffix",
                 cl::desc("Appended to function name to get output file name "
                          "(default: \".html\")"),
                 cl::init(".html"), cl::Hidden);

static cl::opt<std::string>
machineFuncsToRender("rmf-funcs",
                     cl::desc("Comma separated list of functions to render"
                              ", or \"*\"."),
                     cl::init(""), cl::Hidden);

static cl::opt<std::string>
pressureClasses("rmf-classes",
                cl::desc("Register classes to render pressure for."),
                cl::init(""), cl::Hidden);

static cl::opt<std::string>
showIntervals("rmf-intervals",
              cl::desc("Live intervals to show alongside code."),
              cl::init(""), cl::Hidden);

static cl::opt<bool>
filterEmpty("rmf-filter-empty-intervals",
            cl::desc("Don't display empty intervals."),
            cl::init(true), cl::Hidden);

static cl::opt<bool>
showEmptyIndexes("rmf-empty-indexes",
                 cl::desc("Render indexes not associated with instructions or "
                          "MBB starts."),
                 cl::init(false), cl::Hidden);

static cl::opt<bool>
useFancyVerticals("rmf-fancy-verts",
                  cl::desc("Use SVG for vertical text."),
                  cl::init(true), cl::Hidden);

static cl::opt<bool>
prettyHTML("rmf-pretty-html",
           cl::desc("Pretty print HTML. For debugging the renderer only.."),
           cl::init(false), cl::Hidden);


namespace llvm {

  bool MFRenderingOptions::renderingOptionsProcessed;
  std::set<std::string> MFRenderingOptions::mfNamesToRender;
  bool MFRenderingOptions::renderAllMFs = false;

  std::set<std::string> MFRenderingOptions::classNamesToRender;
  bool MFRenderingOptions::renderAllClasses = false;

  std::set<std::pair<unsigned, unsigned> >
    MFRenderingOptions::intervalNumsToRender;
  unsigned MFRenderingOptions::intervalTypesToRender = ExplicitOnly;

  template <typename OutputItr>
  void MFRenderingOptions::splitComaSeperatedList(const std::string &s,
                                                         OutputItr outItr) {
    std::string::const_iterator curPos = s.begin();
    std::string::const_iterator nextComa = std::find(curPos, s.end(), ',');
    while (nextComa != s.end()) {
      std::string elem;
      std::copy(curPos, nextComa, std::back_inserter(elem));
      *outItr = elem;
      ++outItr;
      curPos = llvm::next(nextComa);
      nextComa = std::find(curPos, s.end(), ',');
    }

    if (curPos != s.end()) {
      std::string elem;
      std::copy(curPos, s.end(), std::back_inserter(elem));
      *outItr = elem;
      ++outItr;
    }
  }

  void MFRenderingOptions::processOptions() {
    if (!renderingOptionsProcessed) {
      processFuncNames();
      processRegClassNames();
      processIntervalNumbers();
      renderingOptionsProcessed = true;
    }
  }

  void MFRenderingOptions::processFuncNames() {
    if (machineFuncsToRender == "*") {
      renderAllMFs = true;
    } else {
      splitComaSeperatedList(machineFuncsToRender,
                             std::inserter(mfNamesToRender,
                                           mfNamesToRender.begin()));
    }
  }

  void MFRenderingOptions::processRegClassNames() {
    if (pressureClasses == "*") {
      renderAllClasses = true;
    } else {
      splitComaSeperatedList(pressureClasses,
                             std::inserter(classNamesToRender,
                                           classNamesToRender.begin()));
    }
  }

  void MFRenderingOptions::processIntervalNumbers() {
    std::set<std::string> intervalRanges;
    splitComaSeperatedList(showIntervals,
                           std::inserter(intervalRanges,
                                         intervalRanges.begin()));
    std::for_each(intervalRanges.begin(), intervalRanges.end(),
                  processIntervalRange);
  }

  void MFRenderingOptions::processIntervalRange(
                                          const std::string &intervalRangeStr) {
    if (intervalRangeStr == "*") {
      intervalTypesToRender |= All;
    } else if (intervalRangeStr == "virt-nospills*") {
      intervalTypesToRender |= VirtNoSpills;
    } else if (intervalRangeStr == "spills*") {
      intervalTypesToRender |= VirtSpills;
    } else if (intervalRangeStr == "virt*") {
      intervalTypesToRender |= AllVirt;
    } else if (intervalRangeStr == "phys*") {
      intervalTypesToRender |= AllPhys;
    } else {
      std::istringstream iss(intervalRangeStr);
      unsigned reg1, reg2;
      if ((iss >> reg1 >> std::ws)) {
        if (iss.eof()) {
          intervalNumsToRender.insert(std::make_pair(reg1, reg1 + 1));
        } else {
          char c;
          iss >> c;
          if (c == '-' && (iss >> reg2)) {
            intervalNumsToRender.insert(std::make_pair(reg1, reg2 + 1));
          } else {
            dbgs() << "Warning: Invalid interval range \""
                   << intervalRangeStr << "\" in -rmf-intervals. Skipping.\n";
          }
        }
      } else {
        dbgs() << "Warning: Invalid interval number \""
               << intervalRangeStr << "\" in -rmf-intervals. Skipping.\n";
      }
    }
  }

  void MFRenderingOptions::setup(MachineFunction *mf,
                                 const TargetRegisterInfo *tri,
                                 LiveIntervals *lis,
                                 const RenderMachineFunction *rmf) {
    this->mf = mf;
    this->tri = tri;
    this->lis = lis;
    this->rmf = rmf;

    clear();
  }

  void MFRenderingOptions::clear() {
    regClassesTranslatedToCurrentFunction = false;
    regClassSet.clear();

    intervalsTranslatedToCurrentFunction = false;
    intervalSet.clear();
  }

  void MFRenderingOptions::resetRenderSpecificOptions() {
    intervalSet.clear();
    intervalsTranslatedToCurrentFunction = false;
  }

  bool MFRenderingOptions::shouldRenderCurrentMachineFunction() const {
    processOptions();

    return (renderAllMFs ||
            mfNamesToRender.find(mf->getFunction()->getName()) !=
              mfNamesToRender.end());    
  }

  const MFRenderingOptions::RegClassSet& MFRenderingOptions::regClasses() const{
    translateRegClassNamesToCurrentFunction();
    return regClassSet;
  }

  const MFRenderingOptions::IntervalSet& MFRenderingOptions::intervals() const {
    translateIntervalNumbersToCurrentFunction();
    return intervalSet;
  }

  bool MFRenderingOptions::renderEmptyIndexes() const {
    return showEmptyIndexes;
  }

  bool MFRenderingOptions::fancyVerticals() const {
    return useFancyVerticals;
  }

  void MFRenderingOptions::translateRegClassNamesToCurrentFunction() const {
    if (!regClassesTranslatedToCurrentFunction) {
      processOptions();
      for (TargetRegisterInfo::regclass_iterator rcItr = tri->regclass_begin(),
                                                 rcEnd = tri->regclass_end();
           rcItr != rcEnd; ++rcItr) {
        const TargetRegisterClass *trc = *rcItr;
        if (renderAllClasses ||
            classNamesToRender.find(trc->getName()) !=
              classNamesToRender.end()) {
          regClassSet.insert(trc);
        }
      }
      regClassesTranslatedToCurrentFunction = true;
    }
  }

  void MFRenderingOptions::translateIntervalNumbersToCurrentFunction() const {
    if (!intervalsTranslatedToCurrentFunction) {
      processOptions();

      // If we're not just doing explicit then do a copy over all matching
      // types.
      if (intervalTypesToRender != ExplicitOnly) {
        for (LiveIntervals::iterator liItr = lis->begin(), liEnd = lis->end();
             liItr != liEnd; ++liItr) {
          LiveInterval *li = liItr->second;

          if (filterEmpty && li->empty())
            continue;

          if ((TargetRegisterInfo::isPhysicalRegister(li->reg) &&
               (intervalTypesToRender & AllPhys))) {
            intervalSet.insert(li);
          } else if (TargetRegisterInfo::isVirtualRegister(li->reg)) {
            if (((intervalTypesToRender & VirtNoSpills) && !rmf->isSpill(li)) || 
                ((intervalTypesToRender & VirtSpills) && rmf->isSpill(li))) {
              intervalSet.insert(li);
            }
          }
        }
      }

      // If we need to process the explicit list...
      if (intervalTypesToRender != All) {
        for (std::set<std::pair<unsigned, unsigned> >::const_iterator
               regRangeItr = intervalNumsToRender.begin(),
               regRangeEnd = intervalNumsToRender.end();
             regRangeItr != regRangeEnd; ++regRangeItr) {
          const std::pair<unsigned, unsigned> &range = *regRangeItr;
          for (unsigned reg = range.first; reg != range.second; ++reg) {
            if (lis->hasInterval(reg)) {
              intervalSet.insert(&lis->getInterval(reg));
            }
          }
        }
      }

      intervalsTranslatedToCurrentFunction = true;
    }
  }

  // ---------- TargetRegisterExtraInformation implementation ----------

  TargetRegisterExtraInfo::TargetRegisterExtraInfo()
    : mapsPopulated(false) {
  }

  void TargetRegisterExtraInfo::setup(MachineFunction *mf,
                                      MachineRegisterInfo *mri,
                                      const TargetRegisterInfo *tri,
                                      LiveIntervals *lis) {
    this->mf = mf;
    this->mri = mri;
    this->tri = tri;
    this->lis = lis;
  }

  void TargetRegisterExtraInfo::reset() {
    if (!mapsPopulated) {
      initWorst();
      //initBounds();
      initCapacity();
      mapsPopulated = true;
    }

    resetPressureAndLiveStates();
  }

  void TargetRegisterExtraInfo::clear() {
    prWorst.clear();
    vrWorst.clear();
    capacityMap.clear();
    pressureMap.clear();
    //liveStatesMap.clear();
    mapsPopulated = false;
  }

  void TargetRegisterExtraInfo::initWorst() {
    assert(!mapsPopulated && prWorst.empty() && vrWorst.empty() &&
           "Worst map already initialised?");

    // Start with the physical registers.
    for (unsigned preg = 1; preg < tri->getNumRegs(); ++preg) {
      WorstMapLine &pregLine = prWorst[preg];

      for (TargetRegisterInfo::regclass_iterator rcItr = tri->regclass_begin(),
                                                 rcEnd = tri->regclass_end();
           rcItr != rcEnd; ++rcItr) {
        const TargetRegisterClass *trc = *rcItr;

        unsigned numOverlaps = 0;
        for (TargetRegisterClass::iterator rItr = trc->begin(),
                                           rEnd = trc->end();
             rItr != rEnd; ++rItr) {
          unsigned trcPReg = *rItr;
          if (tri->regsOverlap(preg, trcPReg))
            ++numOverlaps;
        }
        
        pregLine[trc] = numOverlaps;
      }
    }

    // Now the register classes.
    for (TargetRegisterInfo::regclass_iterator rc1Itr = tri->regclass_begin(),
                                               rcEnd = tri->regclass_end();
         rc1Itr != rcEnd; ++rc1Itr) {
      const TargetRegisterClass *trc1 = *rc1Itr;
      WorstMapLine &classLine = vrWorst[trc1];

      for (TargetRegisterInfo::regclass_iterator rc2Itr = tri->regclass_begin();
           rc2Itr != rcEnd; ++rc2Itr) {
        const TargetRegisterClass *trc2 = *rc2Itr;

        unsigned worst = 0;

        for (TargetRegisterClass::iterator trc1Itr = trc1->begin(),
                                           trc1End = trc1->end();
             trc1Itr != trc1End; ++trc1Itr) {
          unsigned trc1Reg = *trc1Itr;
          unsigned trc1RegWorst = 0;

          for (TargetRegisterClass::iterator trc2Itr = trc2->begin(),
                                             trc2End = trc2->end();
               trc2Itr != trc2End; ++trc2Itr) {
            unsigned trc2Reg = *trc2Itr;
            if (tri->regsOverlap(trc1Reg, trc2Reg))
              ++trc1RegWorst;
          }
          if (trc1RegWorst > worst) {
            worst = trc1RegWorst;
          }    
        }

        if (worst != 0) {
          classLine[trc2] = worst;
        }
      }
    }
  }

  unsigned TargetRegisterExtraInfo::getWorst(
                                        unsigned reg,
                                        const TargetRegisterClass *trc) const {
    const WorstMapLine *wml = 0;
    if (TargetRegisterInfo::isPhysicalRegister(reg)) {
      PRWorstMap::const_iterator prwItr = prWorst.find(reg);
      assert(prwItr != prWorst.end() && "Missing prWorst entry.");
      wml = &prwItr->second;
    } else {
      const TargetRegisterClass *regTRC = mri->getRegClass(reg);
      VRWorstMap::const_iterator vrwItr = vrWorst.find(regTRC);
      assert(vrwItr != vrWorst.end() && "Missing vrWorst entry.");
      wml = &vrwItr->second;
    }
    
    WorstMapLine::const_iterator wmlItr = wml->find(trc);
    if (wmlItr == wml->end())
      return 0;

    return wmlItr->second;
  }

  void TargetRegisterExtraInfo::initCapacity() {
    assert(!mapsPopulated && capacityMap.empty() &&
           "Capacity map already initialised?");

    for (TargetRegisterInfo::regclass_iterator rcItr = tri->regclass_begin(),
           rcEnd = tri->regclass_end();
         rcItr != rcEnd; ++rcItr) {
      const TargetRegisterClass *trc = *rcItr;
      unsigned capacity = std::distance(trc->allocation_order_begin(*mf),
                                        trc->allocation_order_end(*mf));

      if (capacity != 0)
        capacityMap[trc] = capacity;
    }
  }

  unsigned TargetRegisterExtraInfo::getCapacity(
                                         const TargetRegisterClass *trc) const {
    CapacityMap::const_iterator cmItr = capacityMap.find(trc);
    assert(cmItr != capacityMap.end() &&
           "vreg with unallocable register class");
    return cmItr->second;
  }

  void TargetRegisterExtraInfo::resetPressureAndLiveStates() {
    pressureMap.clear();
    //liveStatesMap.clear();

    // Iterate over all slots.
    

    // Iterate over all live intervals.
    for (LiveIntervals::iterator liItr = lis->begin(),
           liEnd = lis->end();
         liItr != liEnd; ++liItr) {
      LiveInterval *li = liItr->second;

      if (TargetRegisterInfo::isPhysicalRegister(li->reg))
        continue;
      
      // For all ranges in the current interal.
      for (LiveInterval::iterator lrItr = li->begin(),
             lrEnd = li->end();
           lrItr != lrEnd; ++lrItr) {
        LiveRange *lr = &*lrItr;
        
        // For all slots in the current range.
        for (SlotIndex i = lr->start; i != lr->end; i = i.getNextSlot()) {

          // Record increased pressure at index for all overlapping classes.
          for (TargetRegisterInfo::regclass_iterator
                 rcItr = tri->regclass_begin(),
                 rcEnd = tri->regclass_end();
               rcItr != rcEnd; ++rcItr) {
            const TargetRegisterClass *trc = *rcItr;

            if (trc->allocation_order_begin(*mf) ==
                trc->allocation_order_end(*mf))
              continue;

            unsigned worstAtI = getWorst(li->reg, trc);

            if (worstAtI != 0) {
              pressureMap[i][trc] += worstAtI;
            }
          }
        }
      }
    } 
  }

  unsigned TargetRegisterExtraInfo::getPressureAtSlot(
                                                 const TargetRegisterClass *trc,
                                                 SlotIndex i) const {
    PressureMap::const_iterator pmItr = pressureMap.find(i);
    if (pmItr == pressureMap.end())
      return 0;
    const PressureMapLine &pmLine = pmItr->second;
    PressureMapLine::const_iterator pmlItr = pmLine.find(trc);
    if (pmlItr == pmLine.end())
      return 0;
    return pmlItr->second;
  }

  bool TargetRegisterExtraInfo::classOverCapacityAtSlot(
                                                 const TargetRegisterClass *trc,
                                                 SlotIndex i) const {
    return (getPressureAtSlot(trc, i) > getCapacity(trc));
  }

  // ---------- MachineFunctionRenderer implementation ----------

  void RenderMachineFunction::Spacer::print(raw_ostream &os) const {
    if (!prettyHTML)
      return;
    for (unsigned i = 0; i < ns; ++i) {
      os << " ";
    }
  }

  RenderMachineFunction::Spacer RenderMachineFunction::s(unsigned ns) const {
    return Spacer(ns);
  }

  raw_ostream& operator<<(raw_ostream &os, const RenderMachineFunction::Spacer &s) {
    s.print(os);
    return os;
  }

  template <typename Iterator>
  std::string RenderMachineFunction::escapeChars(Iterator sBegin, Iterator sEnd) const {
    std::string r;

    for (Iterator sItr = sBegin; sItr != sEnd; ++sItr) {
      char c = *sItr;

      switch (c) {
        case '<': r.append("&lt;"); break;
        case '>': r.append("&gt;"); break;
        case '&': r.append("&amp;"); break;
        case ' ': r.append("&nbsp;"); break;
        case '\"': r.append("&quot;"); break;
        default: r.push_back(c); break;
      }
    }

    return r;
  }

  RenderMachineFunction::LiveState
  RenderMachineFunction::getLiveStateAt(const LiveInterval *li,
                                        SlotIndex i) const {
    const MachineInstr *mi = sis->getInstructionFromIndex(i);

    // For uses/defs recorded use/def indexes override current liveness and
    // instruction operands (Only for the interval which records the indexes).
    if (i.isUse() || i.isDef()) {
      UseDefs::const_iterator udItr = useDefs.find(li);
      if (udItr != useDefs.end()) {
        const SlotSet &slotSet = udItr->second;
        if (slotSet.count(i)) {
          if (i.isUse()) {
            return Used;
          }
          // else
          return Defined;
        }
      }
    }

    // If the slot is a load/store, or there's no info in the use/def set then
    // use liveness and instruction operand info.
    if (li->liveAt(i)) {

      if (mi == 0) {
        if (vrm == 0 || 
            (vrm->getStackSlot(li->reg) == VirtRegMap::NO_STACK_SLOT)) {
          return AliveReg;
        } else {
          return AliveStack;
        }
      } else {
        if (i.isDef() && mi->definesRegister(li->reg, tri)) {
          return Defined;
        } else if (i.isUse() && mi->readsRegister(li->reg)) {
          return Used;
        } else {
          if (vrm == 0 || 
              (vrm->getStackSlot(li->reg) == VirtRegMap::NO_STACK_SLOT)) {
            return AliveReg;
          } else {
            return AliveStack;
          }
        }
      }
    }
    return Dead;
  }

  RenderMachineFunction::PressureState
  RenderMachineFunction::getPressureStateAt(const TargetRegisterClass *trc,
                                              SlotIndex i) const {
    if (trei.getPressureAtSlot(trc, i) == 0) {
      return Zero;
    } else if (trei.classOverCapacityAtSlot(trc, i)){
      return High;
    }
    return Low;
  }

  /// \brief Render a machine instruction.
  void RenderMachineFunction::renderMachineInstr(raw_ostream &os,
                                                 const MachineInstr *mi) const {
    std::string s;
    raw_string_ostream oss(s);
    oss << *mi;

    os << escapeChars(oss.str());
  }

  template <typename T>
  void RenderMachineFunction::renderVertical(const Spacer &indent,
                                             raw_ostream &os,
                                             const T &t) const {
    if (ro.fancyVerticals()) {
      os << indent << "<object\n"
         << indent + s(2) << "class=\"obj\"\n"
         << indent + s(2) << "type=\"image/svg+xml\"\n"
         << indent + s(2) << "width=\"14px\"\n"
         << indent + s(2) << "height=\"55px\"\n"
         << indent + s(2) << "data=\"data:image/svg+xml,\n"
         << indent + s(4) << "<svg xmlns='http://www.w3.org/2000/svg'>\n"
         << indent + s(6) << "<text x='-55' y='10' "
                             "font-family='Courier' font-size='12' "
                             "transform='rotate(-90)' "
                             "text-rendering='optimizeSpeed' "
                             "fill='#000'>" << t << "</text>\n"
         << indent + s(4) << "</svg>\">\n"
         << indent << "</object>\n";
    } else {
      std::ostringstream oss;
      oss << t;
      std::string tStr(oss.str());

      os << indent;
      for (std::string::iterator tStrItr = tStr.begin(), tStrEnd = tStr.end();
           tStrItr != tStrEnd; ++tStrItr) {
        os << *tStrItr << "<br/>";
      }
      os << "\n";
    }
  }

  void RenderMachineFunction::insertCSS(const Spacer &indent,
                                        raw_ostream &os) const {
    os << indent << "<style type=\"text/css\">\n"
       << indent + s(2) << "body { font-color: black; }\n"
       << indent + s(2) << "table.code td { font-family: monospace; "
                    "border-width: 0px; border-style: solid; "
                    "border-bottom: 1px solid #dddddd; white-space: nowrap; }\n"
       << indent + s(2) << "table.code td.p-z { background-color: #000000; }\n"
       << indent + s(2) << "table.code td.p-l { background-color: #00ff00; }\n"
       << indent + s(2) << "table.code td.p-h { background-color: #ff0000; }\n"
       << indent + s(2) << "table.code td.l-n { background-color: #ffffff; }\n"
       << indent + s(2) << "table.code td.l-d { background-color: #ff0000; }\n"
       << indent + s(2) << "table.code td.l-u { background-color: #ffff00; }\n"
       << indent + s(2) << "table.code td.l-r { background-color: #000000; }\n"
       << indent + s(2) << "table.code td.l-s { background-color: #770000; }\n"
       << indent + s(2) << "table.code th { border-width: 0px; "
                    "border-style: solid; }\n"
       << indent << "</style>\n";
  }

  void RenderMachineFunction::renderFunctionSummary(
                                    const Spacer &indent, raw_ostream &os,
                                    const char * const renderContextStr) const {
    os << indent << "<h1>Function: " << mf->getFunction()->getName()
                 << "</h1>\n"
       << indent << "<h2>Rendering context: " << renderContextStr << "</h2>\n";
  }


  void RenderMachineFunction::renderPressureTableLegend(
                                                      const Spacer &indent,
                                                      raw_ostream &os) const {
    os << indent << "<h2>Rendering Pressure Legend:</h2>\n"
       << indent << "<table class=\"code\">\n"
       << indent + s(2) << "<tr>\n"
       << indent + s(4) << "<th>Pressure</th><th>Description</th>"
                    "<th>Appearance</th>\n"
       << indent + s(2) << "</tr>\n"
       << indent + s(2) << "<tr>\n"
       << indent + s(4) << "<td>No Pressure</td>"
                    "<td>No physical registers of this class requested.</td>"
                    "<td class=\"p-z\">&nbsp;&nbsp;</td>\n"
       << indent + s(2) << "</tr>\n"
       << indent + s(2) << "<tr>\n"
       << indent + s(4) << "<td>Low Pressure</td>"
                    "<td>Sufficient physical registers to meet demand.</td>"
                    "<td class=\"p-l\">&nbsp;&nbsp;</td>\n"
       << indent + s(2) << "</tr>\n"
       << indent + s(2) << "<tr>\n"
       << indent + s(4) << "<td>High Pressure</td>"
                    "<td>Potentially insufficient physical registers to meet demand.</td>"
                    "<td class=\"p-h\">&nbsp;&nbsp;</td>\n"
       << indent + s(2) << "</tr>\n"
       << indent << "</table>\n";
  }

  template <typename CellType>
  void RenderMachineFunction::renderCellsWithRLE(
                   const Spacer &indent, raw_ostream &os,
                   const std::pair<CellType, unsigned> &rleAccumulator,
                   const std::map<CellType, std::string> &cellTypeStrs) const {

    if (rleAccumulator.second == 0)
      return; 

    typename std::map<CellType, std::string>::const_iterator ctsItr =
      cellTypeStrs.find(rleAccumulator.first);

    assert(ctsItr != cellTypeStrs.end() && "No string for given cell type.");

    os << indent + s(4) << "<td class=\"" << ctsItr->second << "\"";
    if (rleAccumulator.second > 1)
      os << " colspan=" << rleAccumulator.second;
    os << "></td>\n";
  }


  void RenderMachineFunction::renderCodeTablePlusPI(const Spacer &indent,
                                                    raw_ostream &os) const {

    std::map<LiveState, std::string> lsStrs;
    lsStrs[Dead] = "l-n";
    lsStrs[Defined] = "l-d";
    lsStrs[Used] = "l-u";
    lsStrs[AliveReg] = "l-r";
    lsStrs[AliveStack] = "l-s";

    std::map<PressureState, std::string> psStrs;
    psStrs[Zero] = "p-z";
    psStrs[Low] = "p-l";
    psStrs[High] = "p-h";

    // Open the table... 

    os << indent << "<table cellpadding=0 cellspacing=0 class=\"code\">\n"
       << indent + s(2) << "<tr>\n";

    // Render the header row...

    os << indent + s(4) << "<th>index</th>\n"
       << indent + s(4) << "<th>instr</th>\n";

    // Render class names if necessary...
    if (!ro.regClasses().empty()) {
      for (MFRenderingOptions::RegClassSet::const_iterator
             rcItr = ro.regClasses().begin(),
             rcEnd = ro.regClasses().end();
           rcItr != rcEnd; ++rcItr) {
        const TargetRegisterClass *trc = *rcItr;
        os << indent + s(4) << "<th>\n";
        renderVertical(indent + s(6), os, trc->getName());
        os << indent + s(4) << "</th>\n";
      }
    }

    // FIXME: Is there a nicer way to insert space between columns in HTML?
    if (!ro.regClasses().empty() && !ro.intervals().empty())
      os << indent + s(4) << "<th>&nbsp;&nbsp;</th>\n";

    // Render interval numbers if necessary...
    if (!ro.intervals().empty()) {
      for (MFRenderingOptions::IntervalSet::const_iterator
             liItr = ro.intervals().begin(),
             liEnd = ro.intervals().end();
           liItr != liEnd; ++liItr) {

        const LiveInterval *li = *liItr;
        os << indent + s(4) << "<th>\n";
        renderVertical(indent + s(6), os, li->reg);
        os << indent + s(4) << "</th>\n";
      }
    }

    os << indent + s(2) << "</tr>\n";

    // End header row, start with the data rows...

    MachineInstr *mi = 0;

    // Data rows:
    for (SlotIndex i = sis->getZeroIndex(); i != sis->getLastIndex();
         i = i.getNextSlot()) {
     
      // Render the slot column. 
      os << indent + s(2) << "<tr height=6ex>\n";
      
      // Render the code column.
      if (i.isLoad()) {
        MachineBasicBlock *mbb = sis->getMBBFromIndex(i);
        mi = sis->getInstructionFromIndex(i);

        if (i == sis->getMBBStartIdx(mbb) || mi != 0 ||
            ro.renderEmptyIndexes()) {
          os << indent + s(4) << "<td rowspan=4>" << i << "&nbsp;</td>\n"
             << indent + s(4) << "<td rowspan=4>\n";

          if (i == sis->getMBBStartIdx(mbb)) {
            os << indent + s(6) << "BB#" << mbb->getNumber() << ":&nbsp;\n";
          } else if (mi != 0) {
            os << indent + s(6) << "&nbsp;&nbsp;";
            renderMachineInstr(os, mi);
          } else {
            // Empty interval - leave blank.
          }
          os << indent + s(4) << "</td>\n";
        } else {
          i = i.getStoreIndex(); // <- Will be incremented to the next index.
          continue;
        }
      }

      // Render the class columns.
      if (!ro.regClasses().empty()) {
        std::pair<PressureState, unsigned> psRLEAccumulator(Zero, 0);
        for (MFRenderingOptions::RegClassSet::const_iterator
               rcItr = ro.regClasses().begin(),
               rcEnd = ro.regClasses().end();
             rcItr != rcEnd; ++rcItr) {
          const TargetRegisterClass *trc = *rcItr;
          PressureState newPressure = getPressureStateAt(trc, i);

          if (newPressure == psRLEAccumulator.first) {
            ++psRLEAccumulator.second;
          } else {
            renderCellsWithRLE(indent + s(4), os, psRLEAccumulator, psStrs);
            psRLEAccumulator.first = newPressure;
            psRLEAccumulator.second = 1;
          }
        }
        renderCellsWithRLE(indent + s(4), os, psRLEAccumulator, psStrs);
      }
  
      // FIXME: Is there a nicer way to insert space between columns in HTML?
      if (!ro.regClasses().empty() && !ro.intervals().empty())
        os << indent + s(4) << "<td width=2em></td>\n";

      if (!ro.intervals().empty()) {
        std::pair<LiveState, unsigned> lsRLEAccumulator(Dead, 0);
        for (MFRenderingOptions::IntervalSet::const_iterator
               liItr = ro.intervals().begin(),
               liEnd = ro.intervals().end();
             liItr != liEnd; ++liItr) {
          const LiveInterval *li = *liItr;
          LiveState newLiveness = getLiveStateAt(li, i);

          if (newLiveness == lsRLEAccumulator.first) {
            ++lsRLEAccumulator.second;
          } else {
            renderCellsWithRLE(indent + s(4), os, lsRLEAccumulator, lsStrs);
            lsRLEAccumulator.first = newLiveness;
            lsRLEAccumulator.second = 1;
          }
        }
        renderCellsWithRLE(indent + s(4), os, lsRLEAccumulator, lsStrs);
      }
      os << indent + s(2) << "</tr>\n";
    }

    os << indent << "</table>\n";

    if (!ro.regClasses().empty())
      renderPressureTableLegend(indent, os);
  }

  void RenderMachineFunction::renderFunctionPage(
                                    raw_ostream &os,
                                    const char * const renderContextStr) const {
    os << "<html>\n"
       << s(2) << "<head>\n"
       << s(4) << "<title>" << fqn << "</title>\n";

    insertCSS(s(4), os);

    os << s(2) << "<head>\n"
       << s(2) << "<body >\n";

    renderFunctionSummary(s(4), os, renderContextStr);

    os << s(4) << "<br/><br/><br/>\n";

    //renderLiveIntervalInfoTable("    ", os);

    os << s(4) << "<br/><br/><br/>\n";

    renderCodeTablePlusPI(s(4), os);

    os << s(2) << "</body>\n"
       << "</html>\n";
  }

  void RenderMachineFunction::getAnalysisUsage(AnalysisUsage &au) const {
    au.addRequired<SlotIndexes>();
    au.addRequired<LiveIntervals>();
    au.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(au);
  }

  bool RenderMachineFunction::runOnMachineFunction(MachineFunction &fn) {

    mf = &fn;
    mri = &mf->getRegInfo();
    tri = mf->getTarget().getRegisterInfo();
    lis = &getAnalysis<LiveIntervals>();
    sis = &getAnalysis<SlotIndexes>();

    trei.setup(mf, mri, tri, lis);
    ro.setup(mf, tri, lis, this);
    spillIntervals.clear();
    spillFor.clear();
    useDefs.clear();

    fqn = mf->getFunction()->getParent()->getModuleIdentifier() + "." +
          mf->getFunction()->getName().str();

    return false;
  }

  void RenderMachineFunction::releaseMemory() {
    trei.clear();
    ro.clear();
    spillIntervals.clear();
    spillFor.clear();
    useDefs.clear();
  }

  void RenderMachineFunction::rememberUseDefs(const LiveInterval *li) {

    if (!ro.shouldRenderCurrentMachineFunction())
      return; 

    for (MachineRegisterInfo::reg_iterator rItr = mri->reg_begin(li->reg),
                                           rEnd = mri->reg_end();
         rItr != rEnd; ++rItr) {
      const MachineInstr *mi = &*rItr;
      if (mi->readsRegister(li->reg)) {
        useDefs[li].insert(lis->getInstructionIndex(mi).getUseIndex());
      }
      if (mi->definesRegister(li->reg)) {
        useDefs[li].insert(lis->getInstructionIndex(mi).getDefIndex());
      }
    }
  }

  void RenderMachineFunction::rememberSpills(
                                     const LiveInterval *li,
                                     const std::vector<LiveInterval*> &spills) {

    if (!ro.shouldRenderCurrentMachineFunction())
      return; 

    for (std::vector<LiveInterval*>::const_iterator siItr = spills.begin(),
                                                    siEnd = spills.end();
         siItr != siEnd; ++siItr) {
      const LiveInterval *spill = *siItr;
      spillIntervals[li].insert(spill);
      spillFor[spill] = li;
    }
  }

  bool RenderMachineFunction::isSpill(const LiveInterval *li) const {
    SpillForMap::const_iterator sfItr = spillFor.find(li);
    if (sfItr == spillFor.end())
      return false;
    return true;
  }

  void RenderMachineFunction::renderMachineFunction(
                                                   const char *renderContextStr,
                                                   const VirtRegMap *vrm,
                                                   const char *renderSuffix) {
    if (!ro.shouldRenderCurrentMachineFunction())
      return; 

    this->vrm = vrm;
    trei.reset();

    std::string rpFileName(mf->getFunction()->getName().str() +
                           (renderSuffix ? renderSuffix : "") +
                           outputFileSuffix);

    std::string errMsg;
    raw_fd_ostream outFile(rpFileName.c_str(), errMsg, raw_fd_ostream::F_Binary);

    renderFunctionPage(outFile, renderContextStr);

    ro.resetRenderSpecificOptions();
  }

  std::string RenderMachineFunction::escapeChars(const std::string &s) const {
    return escapeChars(s.begin(), s.end());
  }

}
