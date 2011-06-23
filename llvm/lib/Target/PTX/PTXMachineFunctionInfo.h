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
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {
/// PTXMachineFunctionInfo - This class is derived from MachineFunction and
/// contains private PTX target-specific information for each MachineFunction.
///
class PTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  bool is_kernel;
  std::vector<unsigned> reg_arg, reg_local_var;
  std::vector<unsigned> reg_ret;
  bool _isDoneAddArg;

public:
  PTXMachineFunctionInfo(MachineFunction &MF)
    : is_kernel(false), reg_ret(PTX::NoRegister), _isDoneAddArg(false) {
      reg_arg.reserve(8);
      reg_local_var.reserve(32);
    }

  void setKernel(bool _is_kernel=true) { is_kernel = _is_kernel; }

  void addArgReg(unsigned reg) { reg_arg.push_back(reg); }
  void addLocalVarReg(unsigned reg) { reg_local_var.push_back(reg); }
  void addRetReg(unsigned reg) {
    if (!isRetReg(reg)) {
      reg_ret.push_back(reg);
    }
  }

  void doneAddArg(void) {
    _isDoneAddArg = true;
  }
  void doneAddLocalVar(void) {}

  bool isKernel() const { return is_kernel; }

  typedef std::vector<unsigned>::const_iterator         reg_iterator;
  typedef std::vector<unsigned>::const_reverse_iterator reg_reverse_iterator;
  typedef std::vector<unsigned>::const_iterator         ret_iterator;

  bool         argRegEmpty() const { return reg_arg.empty(); }
  int          getNumArg() const { return reg_arg.size(); }
  reg_iterator argRegBegin() const { return reg_arg.begin(); }
  reg_iterator argRegEnd()   const { return reg_arg.end(); }
  reg_reverse_iterator argRegReverseBegin() const { return reg_arg.rbegin(); }
  reg_reverse_iterator argRegReverseEnd() const { return reg_arg.rend(); }

  bool         localVarRegEmpty() const { return reg_local_var.empty(); }
  reg_iterator localVarRegBegin() const { return reg_local_var.begin(); }
  reg_iterator localVarRegEnd()   const { return reg_local_var.end(); }

  bool         retRegEmpty() const { return reg_ret.empty(); }
  int          getNumRet() const { return reg_ret.size(); }
  ret_iterator retRegBegin() const { return reg_ret.begin(); }
  ret_iterator retRegEnd()   const { return reg_ret.end(); }

  bool isArgReg(unsigned reg) const {
    return std::find(reg_arg.begin(), reg_arg.end(), reg) != reg_arg.end();
  }

  bool isRetReg(unsigned reg) const {
    return std::find(reg_ret.begin(), reg_ret.end(), reg) != reg_ret.end();
  }

  bool isLocalVarReg(unsigned reg) const {
    return std::find(reg_local_var.begin(), reg_local_var.end(), reg)
      != reg_local_var.end();
  }
}; // class PTXMachineFunctionInfo
} // namespace llvm

#endif // PTX_MACHINE_FUNCTION_INFO_H
