//===- PTXMachineFuctionInfo.h - PTX machine function info -------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares PTX-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_MACHINE_FUNCTION_INFO_H
#define PTX_MACHINE_FUNCTION_INFO_H

#include "PTX.h"
#include "PTXParamManager.h"
#include "PTXRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// PTXMachineFunctionInfo - This class is derived from MachineFunction and
/// contains private PTX target-specific information for each MachineFunction.
///
class PTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  bool IsKernel;
  DenseSet<unsigned> RegArgs;
  DenseSet<unsigned> RegRets;

  typedef std::vector<unsigned> RegisterList;
  typedef DenseMap<const TargetRegisterClass*, RegisterList> RegisterMap;
  typedef DenseMap<unsigned, std::string> RegisterNameMap;
  typedef DenseMap<int, std::string> FrameMap;

  RegisterMap UsedRegs;
  RegisterNameMap RegNames;
  FrameMap FrameSymbols;

  PTXParamManager ParamManager;

public:
  typedef DenseSet<unsigned>::const_iterator reg_iterator;

  PTXMachineFunctionInfo(MachineFunction &MF)
    : IsKernel(false) {
      UsedRegs[PTX::RegPredRegisterClass] = RegisterList();
      UsedRegs[PTX::RegI16RegisterClass] = RegisterList();
      UsedRegs[PTX::RegI32RegisterClass] = RegisterList();
      UsedRegs[PTX::RegI64RegisterClass] = RegisterList();
      UsedRegs[PTX::RegF32RegisterClass] = RegisterList();
      UsedRegs[PTX::RegF64RegisterClass] = RegisterList();
    }

  /// getParamManager - Returns the PTXParamManager instance for this function.
  PTXParamManager& getParamManager() { return ParamManager; }
  const PTXParamManager& getParamManager() const { return ParamManager; }

  /// setKernel/isKernel - Gets/sets a flag that indicates if this function is
  /// a PTX kernel function.
  void setKernel(bool _IsKernel=true) { IsKernel = _IsKernel; }
  bool isKernel() const { return IsKernel; }

  /// argreg_begin/argreg_end - Returns iterators to the set of registers
  /// containing function arguments.
  reg_iterator argreg_begin() const { return RegArgs.begin(); }
  reg_iterator argreg_end()   const { return RegArgs.end(); }

  /// retreg_begin/retreg_end - Returns iterators to the set of registers
  /// containing the function return values.
  reg_iterator retreg_begin() const { return RegRets.begin(); }
  reg_iterator retreg_end()   const { return RegRets.end(); }

  /// addRetReg - Adds a register to the set of return-value registers.
  void addRetReg(unsigned Reg) {
    if (!RegRets.count(Reg)) {
      RegRets.insert(Reg);
      std::string name;
      name = "%ret";
      name += utostr(RegRets.size() - 1);
      RegNames[Reg] = name;
    }
  }

  /// addArgReg - Adds a register to the set of function argument registers.
  void addArgReg(unsigned Reg) {
    RegArgs.insert(Reg);
    std::string name;
    name = "%param";
    name += utostr(RegArgs.size() - 1);
    RegNames[Reg] = name;
  }

  /// addVirtualRegister - Adds a virtual register to the set of all used
  /// registers in the function.
  void addVirtualRegister(const TargetRegisterClass *TRC, unsigned Reg) {
    std::string name;

    // Do not count registers that are argument/return registers.
    if (!RegRets.count(Reg) && !RegArgs.count(Reg)) {
      UsedRegs[TRC].push_back(Reg);
      if (TRC == PTX::RegPredRegisterClass)
        name = "%p";
      else if (TRC == PTX::RegI16RegisterClass)
        name = "%rh";
      else if (TRC == PTX::RegI32RegisterClass)
        name = "%r";
      else if (TRC == PTX::RegI64RegisterClass)
        name = "%rd";
      else if (TRC == PTX::RegF32RegisterClass)
        name = "%f";
      else if (TRC == PTX::RegF64RegisterClass)
        name = "%fd";
      else
        llvm_unreachable("Invalid register class");

      name += utostr(UsedRegs[TRC].size() - 1);
      RegNames[Reg] = name;
    }
  }

  /// getRegisterName - Returns the name of the specified virtual register. This
  /// name is used during PTX emission.
  const char *getRegisterName(unsigned Reg) const {
    if (RegNames.count(Reg))
      return RegNames.find(Reg)->second.c_str();
    else if (Reg == PTX::NoRegister)
      return "%noreg";
    else
      llvm_unreachable("Register not in register name map");
  }

  /// getNumRegistersForClass - Returns the number of virtual registers that are
  /// used for the specified register class.
  unsigned getNumRegistersForClass(const TargetRegisterClass *TRC) const {
    return UsedRegs.lookup(TRC).size();
  }

  /// getOffsetForRegister - Returns the offset of the virtual register
  unsigned getOffsetForRegister(const TargetRegisterClass *TRC,
                                unsigned Reg) const {
    const RegisterList &RegList = UsedRegs.lookup(TRC);
    for (unsigned i = 0, e = RegList.size(); i != e; ++i) {
      if (RegList[i] == Reg)
        return i;
    }
    //llvm_unreachable("Unknown virtual register");
    return 0;
  }

  /// getFrameSymbol - Returns the symbol name for the given FrameIndex.
  const char* getFrameSymbol(int FrameIndex) {
    if (FrameSymbols.count(FrameIndex)) {
      return FrameSymbols.lookup(FrameIndex).c_str();
    } else {
      std::string Name          = "__local";
      Name                     += utostr(FrameIndex);
      // The whole point of caching this name is to ensure the pointer we pass
      // to any getExternalSymbol() calls will remain valid for the lifetime of
      // the back-end instance. This is to work around an issue in SelectionDAG
      // where symbol names are expected to be life-long strings.
      FrameSymbols[FrameIndex]  = Name;
      return FrameSymbols[FrameIndex].c_str();
    }
  }
}; // class PTXMachineFunctionInfo
} // namespace llvm

#endif // PTX_MACHINE_FUNCTION_INFO_H
