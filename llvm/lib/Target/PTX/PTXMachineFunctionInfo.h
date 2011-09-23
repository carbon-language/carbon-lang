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
  bool is_kernel;
  DenseSet<unsigned> reg_local_var;
  DenseSet<unsigned> reg_arg;
  DenseSet<unsigned> reg_ret;
  std::vector<unsigned> call_params;
  bool _isDoneAddArg;

  typedef std::vector<unsigned> RegisterList;
  typedef DenseMap<const TargetRegisterClass*, RegisterList> RegisterMap;
  typedef DenseMap<unsigned, std::string> RegisterNameMap;

  RegisterMap usedRegs;
  RegisterNameMap regNames;

  SmallVector<unsigned, 8> argParams;

  unsigned retParamSize;

  PTXParamManager ParamManager;

public:
  PTXMachineFunctionInfo(MachineFunction &MF)
    : is_kernel(false), reg_ret(PTX::NoRegister), _isDoneAddArg(false) {
      usedRegs[PTX::RegPredRegisterClass] = RegisterList();
      usedRegs[PTX::RegI16RegisterClass] = RegisterList();
      usedRegs[PTX::RegI32RegisterClass] = RegisterList();
      usedRegs[PTX::RegI64RegisterClass] = RegisterList();
      usedRegs[PTX::RegF32RegisterClass] = RegisterList();
      usedRegs[PTX::RegF64RegisterClass] = RegisterList();

      retParamSize = 0;
    }

  PTXParamManager& getParamManager() { return ParamManager; }
  const PTXParamManager& getParamManager() const { return ParamManager; }

  void setKernel(bool _is_kernel=true) { is_kernel = _is_kernel; }


  void addLocalVarReg(unsigned reg) { reg_local_var.insert(reg); }


  void doneAddArg(void) {
    _isDoneAddArg = true;
  }
  void doneAddLocalVar(void) {}

  bool isKernel() const { return is_kernel; }

  typedef DenseSet<unsigned>::const_iterator         reg_iterator;
  //typedef DenseSet<unsigned>::const_reverse_iterator reg_reverse_iterator;
  typedef DenseSet<unsigned>::const_iterator         ret_iterator;
  typedef std::vector<unsigned>::const_iterator         param_iterator;
  typedef SmallVector<unsigned, 8>::const_iterator    argparam_iterator;

  bool         argRegEmpty() const { return reg_arg.empty(); }
  int          getNumArg() const { return reg_arg.size(); }
  reg_iterator argRegBegin() const { return reg_arg.begin(); }
  reg_iterator argRegEnd()   const { return reg_arg.end(); }
  argparam_iterator argParamBegin() const { return argParams.begin(); }
  argparam_iterator argParamEnd() const { return argParams.end(); }
  //reg_reverse_iterator argRegReverseBegin() const { return reg_arg.rbegin(); }
  //reg_reverse_iterator argRegReverseEnd() const { return reg_arg.rend(); }

  bool         localVarRegEmpty() const { return reg_local_var.empty(); }
  reg_iterator localVarRegBegin() const { return reg_local_var.begin(); }
  reg_iterator localVarRegEnd()   const { return reg_local_var.end(); }

  bool         retRegEmpty() const { return reg_ret.empty(); }
  int          getNumRet() const { return reg_ret.size(); }
  ret_iterator retRegBegin() const { return reg_ret.begin(); }
  ret_iterator retRegEnd()   const { return reg_ret.end(); }

  param_iterator paramBegin() const { return call_params.begin(); }
  param_iterator paramEnd() const { return call_params.end(); }
  unsigned       getNextParam(unsigned size) {
    call_params.push_back(size);
    return call_params.size()-1;
  }

  bool isArgReg(unsigned reg) const {
    return std::find(reg_arg.begin(), reg_arg.end(), reg) != reg_arg.end();
  }

  /*bool isRetReg(unsigned reg) const {
    return std::find(reg_ret.begin(), reg_ret.end(), reg) != reg_ret.end();
  }*/

  bool isLocalVarReg(unsigned reg) const {
    return std::find(reg_local_var.begin(), reg_local_var.end(), reg)
      != reg_local_var.end();
  }

  void addRetReg(unsigned Reg) {
    if (!reg_ret.count(Reg)) {
      reg_ret.insert(Reg);
      std::string name;
      name = "%ret";
      name += utostr(reg_ret.size() - 1);
      regNames[Reg] = name;
    }
  }

  void setRetParamSize(unsigned SizeInBits) {
    retParamSize = SizeInBits;
  }

  unsigned getRetParamSize() const {
    return retParamSize;
  }

  void addArgReg(unsigned Reg) {
    reg_arg.insert(Reg);
    std::string name;
    name = "%param";
    name += utostr(reg_arg.size() - 1);
    regNames[Reg] = name;
  }

  void addArgParam(unsigned SizeInBits) {
    argParams.push_back(SizeInBits);
  }

  void addVirtualRegister(const TargetRegisterClass *TRC, unsigned Reg) {
    std::string name;

    if (!reg_ret.count(Reg) && !reg_arg.count(Reg)) {
      usedRegs[TRC].push_back(Reg);
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

      name += utostr(usedRegs[TRC].size() - 1);
      regNames[Reg] = name;
    }
  }

  std::string getRegisterName(unsigned Reg) const {
    if (regNames.count(Reg))
      return regNames.lookup(Reg);
    else if (Reg == PTX::NoRegister)
      return "%noreg";
    else
      llvm_unreachable("Register not in register name map");
  }

  unsigned getNumRegistersForClass(const TargetRegisterClass *TRC) const {
    return usedRegs.lookup(TRC).size();
  }

}; // class PTXMachineFunctionInfo
} // namespace llvm

#endif // PTX_MACHINE_FUNCTION_INFO_H
