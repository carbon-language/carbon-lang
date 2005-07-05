//===-- IA64ISelPattern.cpp - A pattern matching inst selector for IA64 ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for IA64.
//
//===----------------------------------------------------------------------===//

#include "IA64.h"
#include "IA64InstrBuilder.h"
#include "IA64RegisterInfo.h"
#include "IA64MachineFunctionInfo.h"
#include "llvm/Constants.h"                   // FIXME: REMOVE
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h" // FIXME: REMOVE
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <map>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  IA64TargetLowering - IA64 Implementation of the TargetLowering interface
namespace {
  class IA64TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.

    //int ReturnAddrIndex;              // FrameIndex for return slot.
    unsigned GP, SP, RP; // FIXME - clean this mess up
  public:

   unsigned VirtGPR; // this is public so it can be accessed in the selector
   // for ISD::RET down below. add an accessor instead? FIXME

   IA64TargetLowering(TargetMachine &TM) : TargetLowering(TM) {

      // register class for general registers
      addRegisterClass(MVT::i64, IA64::GRRegisterClass);

      // register class for FP registers
      addRegisterClass(MVT::f64, IA64::FPRegisterClass);

      // register class for predicate registers
      addRegisterClass(MVT::i1, IA64::PRRegisterClass);

      setOperationAction(ISD::BRCONDTWOWAY     , MVT::Other, Expand);
      setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);

      setSetCCResultType(MVT::i1);
      setShiftAmountType(MVT::i64);

      setOperationAction(ISD::EXTLOAD          , MVT::i1   , Promote);

      setOperationAction(ISD::ZEXTLOAD         , MVT::i1   , Expand);

      setOperationAction(ISD::SEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i8   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i16  , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i32  , Expand);

      setOperationAction(ISD::SREM             , MVT::f32  , Expand);
      setOperationAction(ISD::SREM             , MVT::f64  , Expand);

      setOperationAction(ISD::UREM             , MVT::f32  , Expand);
      setOperationAction(ISD::UREM             , MVT::f64  , Expand);

      setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
      setOperationAction(ISD::MEMSET           , MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY           , MVT::Other, Expand);

      // We don't support sin/cos/sqrt
      setOperationAction(ISD::FSIN , MVT::f64, Expand);
      setOperationAction(ISD::FCOS , MVT::f64, Expand);
      setOperationAction(ISD::FSQRT, MVT::f64, Expand);
      setOperationAction(ISD::FSIN , MVT::f32, Expand);
      setOperationAction(ISD::FCOS , MVT::f32, Expand);
      setOperationAction(ISD::FSQRT, MVT::f32, Expand);

      //IA64 has these, but they are not implemented
      setOperationAction(ISD::CTTZ , MVT::i64  , Expand);
      setOperationAction(ISD::CTLZ , MVT::i64  , Expand);

      computeRegisterProperties();

      addLegalFPImmediate(+0.0);
      addLegalFPImmediate(+1.0);
      addLegalFPImmediate(-0.0);
      addLegalFPImmediate(-1.0);
    }

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);

    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    virtual std::pair<SDOperand,SDOperand>
      LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                 const Type *ArgTy, SelectionDAG &DAG);

    void restoreGP_SP_RP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::r1).addReg(GP);
      BuildMI(BB, IA64::MOV, 1, IA64::r12).addReg(SP);
      BuildMI(BB, IA64::MOV, 1, IA64::rp).addReg(RP);
    }

    void restoreSP_RP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::r12).addReg(SP);
      BuildMI(BB, IA64::MOV, 1, IA64::rp).addReg(RP);
    }

    void restoreRP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::rp).addReg(RP);
    }

    void restoreGP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::r1).addReg(GP);
    }

  };
}


std::vector<SDOperand>
IA64TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  std::vector<SDOperand> ArgValues;

  //
  // add beautiful description of IA64 stack frame format
  // here (from intel 24535803.pdf most likely)
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  GP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  SP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  RP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));

  MachineBasicBlock& BB = MF.front();

  unsigned args_int[] = {IA64::r32, IA64::r33, IA64::r34, IA64::r35,
                         IA64::r36, IA64::r37, IA64::r38, IA64::r39};

  unsigned args_FP[] = {IA64::F8, IA64::F9, IA64::F10, IA64::F11,
                        IA64::F12,IA64::F13,IA64::F14, IA64::F15};

  unsigned argVreg[8];
  unsigned argPreg[8];
  unsigned argOpc[8];

  unsigned used_FPArgs = 0; // how many FP args have been used so far?

  unsigned ArgOffset = 0;
  int count = 0;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    {
      SDOperand newroot, argt;
      if(count < 8) { // need to fix this logic? maybe.

        switch (getValueType(I->getType())) {
          default:
            std::cerr << "ERROR in LowerArgs: unknown type "
              << getValueType(I->getType()) << "\n";
            abort();
          case MVT::f32:
            // fixme? (well, will need to for weird FP structy stuff,
            // see intel ABI docs)
          case MVT::f64:
//XXX            BuildMI(&BB, IA64::IDEF, 0, args_FP[used_FPArgs]);
            MF.addLiveIn(args_FP[used_FPArgs]); // mark this reg as liveIn
            // floating point args go into f8..f15 as-needed, the increment
            argVreg[count] =                              // is below..:
            MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::f64));
            // FP args go into f8..f15 as needed: (hence the ++)
            argPreg[count] = args_FP[used_FPArgs++];
            argOpc[count] = IA64::FMOV;
            argt = newroot = DAG.getCopyFromReg(argVreg[count],
                getValueType(I->getType()), DAG.getRoot());
            break;
          case MVT::i1: // NOTE: as far as C abi stuff goes,
                        // bools are just boring old ints
          case MVT::i8:
          case MVT::i16:
          case MVT::i32:
          case MVT::i64:
//XXX            BuildMI(&BB, IA64::IDEF, 0, args_int[count]);
            MF.addLiveIn(args_int[count]); // mark this register as liveIn
            argVreg[count] =
            MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
            argPreg[count] = args_int[count];
            argOpc[count] = IA64::MOV;
            argt = newroot =
              DAG.getCopyFromReg(argVreg[count], MVT::i64, DAG.getRoot());
            if ( getValueType(I->getType()) != MVT::i64)
              argt = DAG.getNode(ISD::TRUNCATE, getValueType(I->getType()),
                  newroot);
            break;
        }
      } else { // more than 8 args go into the frame
        // Create the frame index object for this incoming parameter...
        ArgOffset = 16 + 8 * (count - 8);
        int FI = MFI->CreateFixedObject(8, ArgOffset);
        
        // Create the SelectionDAG nodes corresponding to a load
        //from this parameter
        SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
        argt = newroot = DAG.getLoad(getValueType(I->getType()),
                                     DAG.getEntryNode(), FIN, DAG.getSrcValue(NULL));
      }
      ++count;
      DAG.setRoot(newroot.getValue(1));
      ArgValues.push_back(argt);
    }


  // Create a vreg to hold the output of (what will become)
  // the "alloc" instruction
  VirtGPR = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  BuildMI(&BB, IA64::PSEUDO_ALLOC, 0, VirtGPR);
  // we create a PSEUDO_ALLOC (pseudo)instruction for now

  BuildMI(&BB, IA64::IDEF, 0, IA64::r1);

  // hmm:
  BuildMI(&BB, IA64::IDEF, 0, IA64::r12);
  BuildMI(&BB, IA64::IDEF, 0, IA64::rp);
  // ..hmm.

  BuildMI(&BB, IA64::MOV, 1, GP).addReg(IA64::r1);

  // hmm:
  BuildMI(&BB, IA64::MOV, 1, SP).addReg(IA64::r12);
  BuildMI(&BB, IA64::MOV, 1, RP).addReg(IA64::rp);
  // ..hmm.

  unsigned tempOffset=0;

  // if this is a varargs function, we simply lower llvm.va_start by
  // pointing to the first entry
  if(F.isVarArg()) {
    tempOffset=0;
    VarArgsFrameIndex = MFI->CreateFixedObject(8, tempOffset);
  }

  // here we actually do the moving of args, and store them to the stack
  // too if this is a varargs function:
  for (int i = 0; i < count && i < 8; ++i) {
    BuildMI(&BB, argOpc[i], 1, argVreg[i]).addReg(argPreg[i]);
    if(F.isVarArg()) {
      // if this is a varargs function, we copy the input registers to the stack
      int FI = MFI->CreateFixedObject(8, tempOffset);
      tempOffset+=8;   //XXX: is it safe to use r22 like this?
      BuildMI(&BB, IA64::MOV, 1, IA64::r22).addFrameIndex(FI);
      // FIXME: we should use st8.spill here, one day
      BuildMI(&BB, IA64::ST8, 1, IA64::r22).addReg(argPreg[i]);
    }
  }

  // Finally, inform the code generator which regs we return values in.
  // (see the ISD::RET: case down below)
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "i have no idea where to return this type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
    MF.addLiveOut(IA64::r8);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(IA64::F8);
    break;
  }

  return ArgValues;
}

std::pair<SDOperand, SDOperand>
IA64TargetLowering::LowerCallTo(SDOperand Chain,
                                const Type *RetTy, bool isVarArg,
                                unsigned CallingConv, bool isTailCall,
                                SDOperand Callee, ArgListTy &Args, 
                                SelectionDAG &DAG) {

  MachineFunction &MF = DAG.getMachineFunction();

  unsigned NumBytes = 16;
  unsigned outRegsUsed = 0;

  if (Args.size() > 8) {
    NumBytes += (Args.size() - 8) * 8;
    outRegsUsed = 8;
  } else {
    outRegsUsed = Args.size();
  }

  // FIXME? this WILL fail if we ever try to pass around an arg that
  // consumes more than a single output slot (a 'real' double, int128
  // some sort of aggregate etc.), as we'll underestimate how many 'outX'
  // registers we use. Hopefully, the assembler will notice.
  MF.getInfo<IA64FunctionInfo>()->outRegsUsed=
    std::max(outRegsUsed, MF.getInfo<IA64FunctionInfo>()->outRegsUsed);

  Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));

  std::vector<SDOperand> args_to_use;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    {
      switch (getValueType(Args[i].second)) {
      default: assert(0 && "unexpected argument type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
        //promote to 64-bits, sign/zero extending based on type
        //of the argument
        if(Args[i].second->isSigned())
          Args[i].first = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64,
              Args[i].first);
        else
          Args[i].first = DAG.getNode(ISD::ZERO_EXTEND, MVT::i64,
              Args[i].first);
        break;
      case MVT::f32:
        //promote to 64-bits
        Args[i].first = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Args[i].first);
      case MVT::f64:
      case MVT::i64:
        break;
      }
      args_to_use.push_back(Args[i].first);
    }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(RetTyVT);
  RetVals.push_back(MVT::Other);

  SDOperand TheCall = SDOperand(DAG.getCall(RetVals, Chain,
                                            Callee, args_to_use), 0);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

SDOperand
IA64TargetLowering::LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                 Value *VAListV, SelectionDAG &DAG) {
  // vastart just stores the address of the VarArgsFrameIndex slot.
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64);
  return DAG.getNode(ISD::STORE, MVT::Other, Chain, FR,
                     VAListP, DAG.getSrcValue(VAListV));
}

std::pair<SDOperand,SDOperand> IA64TargetLowering::
LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
           const Type *ArgTy, SelectionDAG &DAG) {

  MVT::ValueType ArgVT = getValueType(ArgTy);
  SDOperand Val = DAG.getLoad(MVT::i64, Chain,
                              VAListP, DAG.getSrcValue(VAListV));
  SDOperand Result = DAG.getLoad(ArgVT, DAG.getEntryNode(), Val,
                                 DAG.getSrcValue(NULL));
  unsigned Amt;
  if (ArgVT == MVT::i32 || ArgVT == MVT::f32)
    Amt = 8;
  else {
    assert((ArgVT == MVT::i64 || ArgVT == MVT::f64) &&
           "Other types should have been promoted for varargs!");
    Amt = 8;
  }
  Val = DAG.getNode(ISD::ADD, Val.getValueType(), Val, 
                    DAG.getConstant(Amt, Val.getValueType()));
  Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                      Val, VAListP, DAG.getSrcValue(VAListV));
  return std::make_pair(Result, Chain);
}

namespace {

  //===--------------------------------------------------------------------===//
  /// ISel - IA64 specific code to select IA64 machine instructions for
  /// SelectionDAG operations.
  ///
  class ISel : public SelectionDAGISel {
    /// IA64Lowering - This object fully describes how to lower LLVM code to an
    /// IA64-specific SelectionDAG.
    IA64TargetLowering IA64Lowering;
    SelectionDAG *ISelDAG; // Hack to support us having a dag->dag transform
                           // for sdiv and udiv until it is put into the future
                           // dag combiner

    /// ExprMap - As shared expressions are codegen'd, we keep track of which
    /// vreg the value is produced in, so we only emit one copy of each compiled
    /// tree.
    std::map<SDOperand, unsigned> ExprMap;
    std::set<SDOperand> LoweredTokens;

  public:
    ISel(TargetMachine &TM) : SelectionDAGISel(IA64Lowering), IA64Lowering(TM),
                              ISelDAG(0) { }

    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);

    unsigned SelectExpr(SDOperand N);
    void Select(SDOperand N);
    // a dag->dag to transform mul-by-constant-int to shifts+adds/subs
    SDOperand BuildConstmulSequence(SDOperand N);

  };
}

/// InstructionSelectBasicBlock - This callback is invoked by SelectionDAGISel
/// when it has created a SelectionDAG for us to codegen.
void ISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {

  // Codegen the basic block.
  ISelDAG = &DAG;
  Select(DAG.getRoot());

  // Clear state used for selection.
  ExprMap.clear();
  LoweredTokens.clear();
  ISelDAG = 0;
}

// strip leading '0' characters from a string
void munchLeadingZeros(std::string& inString) {
  while(inString.c_str()[0]=='0') {
    inString.erase(0, 1);
  }
}

// strip trailing '0' characters from a string
void munchTrailingZeros(std::string& inString) {
  int curPos=inString.length()-1;

  while(inString.c_str()[curPos]=='0') {
    inString.erase(curPos, 1);
    curPos--;
  }
}

// return how many consecutive '0' characters are at the end of a string
unsigned int countTrailingZeros(std::string& inString) {
  int curPos=inString.length()-1;
  unsigned int zeroCount=0;
  // assert goes here
  while(inString.c_str()[curPos--]=='0') {
    zeroCount++;
  }
  return zeroCount;
}

// booth encode a string of '1' and '0' characters (returns string of 'P' (+1)
// '0' and 'N' (-1) characters)
void boothEncode(std::string inString, std::string& boothEncodedString) {

  int curpos=0;
  int replacements=0;
  int lim=inString.size();

  while(curpos<lim) {
    if(inString[curpos]=='1') { // if we see a '1', look for a run of them 
      int runlength=0;
      std::string replaceString="N";
     
      // find the run length
      for(;inString[curpos+runlength]=='1';runlength++) ;

      for(int i=0; i<runlength-1; i++)
	replaceString+="0";
      replaceString+="1";

      if(runlength>1) {
	inString.replace(curpos, runlength+1, replaceString);
	curpos+=runlength-1;
      } else
	curpos++;
    } else { // a zero, we just keep chugging along
      curpos++;
    }
  }

  // clean up (trim the string, reverse it and turn '1's into 'P's)
  munchTrailingZeros(inString);
  boothEncodedString="";

  for(int i=inString.size()-1;i>=0;i--)
    if(inString[i]=='1')
      boothEncodedString+="P";
    else
      boothEncodedString+=inString[i];

}

struct shiftaddblob { // this encodes stuff like (x=) "A << B [+-] C << D"
  unsigned firstVal;    // A
  unsigned firstShift;  // B 
  unsigned secondVal;   // C
  unsigned secondShift; // D
  bool isSub;
};

/* this implements Lefevre's "pattern-based" constant multiplication,
 * see "Multiplication by an Integer Constant", INRIA report 1999-06
 *
 * TODO: implement a method to try rewriting P0N<->0PP / N0P<->0NN
 * to get better booth encodings - this does help in practice
 * TODO: weight shifts appropriately (most architectures can't
 * fuse a shift and an add for arbitrary shift amounts) */
unsigned lefevre(const std::string inString,
                 std::vector<struct shiftaddblob> &ops) {
  std::string retstring;
  std::string s = inString;
  munchTrailingZeros(s);

  int length=s.length()-1;

  if(length==0) {
    return(0);
  }

  std::vector<int> p,n;
  
  for(int i=0; i<=length; i++) {
    if (s.c_str()[length-i]=='P') {
      p.push_back(i);
    } else if (s.c_str()[length-i]=='N') {
      n.push_back(i);
    }
  }

  std::string t, u;
  int c;
  bool f;
  std::map<const int, int> w;

  for(unsigned i=0; i<p.size(); i++) {
    for(unsigned j=0; j<i; j++) {
      w[p[i]-p[j]]++;
    }
  }

  for(unsigned i=1; i<n.size(); i++) {
    for(unsigned j=0; j<i; j++) {
      w[n[i]-n[j]]++;
    }
  }

  for(unsigned i=0; i<p.size(); i++) {
    for(unsigned j=0; j<n.size(); j++) {
      w[-abs(p[i]-n[j])]++;
    }
  }

  std::map<const int, int>::const_iterator ii;
  std::vector<int> d;
  std::multimap<int, int> sorted_by_value;

  for(ii = w.begin(); ii!=w.end(); ii++)
    sorted_by_value.insert(std::pair<int, int>((*ii).second,(*ii).first));

  for (std::multimap<int, int>::iterator it = sorted_by_value.begin();
       it != sorted_by_value.end(); ++it) {
    d.push_back((*it).second);
  }

  int int_W=0;
  int int_d;

  while(d.size()>0 && (w[int_d=d.back()] > int_W)) {
    d.pop_back();
    retstring=s; // hmmm
    int x=0;
    int z=abs(int_d)-1;

    if(int_d>0) {
      
      for(unsigned base=0; base<retstring.size(); base++) {
	if( ((base+z+1) < retstring.size()) &&
	   retstring.c_str()[base]=='P' &&
	   retstring.c_str()[base+z+1]=='P')
	{
	  // match
	  x++;
	  retstring.replace(base, 1, "0");
	  retstring.replace(base+z+1, 1, "p");
	}
      }

      for(unsigned base=0; base<retstring.size(); base++) {
	if( ((base+z+1) < retstring.size()) &&
	   retstring.c_str()[base]=='N' &&
	   retstring.c_str()[base+z+1]=='N')
	{
	  // match
	  x++;
	  retstring.replace(base, 1, "0");
	  retstring.replace(base+z+1, 1, "n");
	}
      }

    } else {
      for(unsigned base=0; base<retstring.size(); base++) {
	if( ((base+z+1) < retstring.size()) &&
	    ((retstring.c_str()[base]=='P' &&
	     retstring.c_str()[base+z+1]=='N') ||
	    (retstring.c_str()[base]=='N' &&
	     retstring.c_str()[base+z+1]=='P')) ) {
	  // match
	  x++;
	  
	  if(retstring.c_str()[base]=='P') {
	    retstring.replace(base, 1, "0");
	    retstring.replace(base+z+1, 1, "p");
	  } else { // retstring[base]=='N'
	    retstring.replace(base, 1, "0");
	    retstring.replace(base+z+1, 1, "n");
	  }
	}
      }
    }

    if(x>int_W) {
      int_W = x;
      t = retstring;
      c = int_d; // tofix
    }
    
  } d.pop_back(); // hmm

  u = t;
  
  for(unsigned i=0; i<t.length(); i++) {
    if(t.c_str()[i]=='p' || t.c_str()[i]=='n')
      t.replace(i, 1, "0");
  }

  for(unsigned i=0; i<u.length(); i++) {
    if(u[i]=='P' || u[i]=='N')
      u.replace(i, 1, "0");
    if(u[i]=='p')
      u.replace(i, 1, "P");
    if(u[i]=='n')
      u.replace(i, 1, "N");
  }

  if( c<0 ) {
    f=true;
    c=-c;
  } else
    f=false;
  
  int pos=0;
  while(u[pos]=='0')
    pos++;

  bool hit=(u[pos]=='N');

  int g=0;
  if(hit) {
    g=1;
    for(unsigned p=0; p<u.length(); p++) {
      bool isP=(u[p]=='P');
      bool isN=(u[p]=='N');

      if(isP)
	u.replace(p, 1, "N");
      if(isN)
	u.replace(p, 1, "P");
    }
  }

  munchLeadingZeros(u);

  int i = lefevre(u, ops);

  shiftaddblob blob;
  
  blob.firstVal=i; blob.firstShift=c;
  blob.isSub=f;
  blob.secondVal=i; blob.secondShift=0;

  ops.push_back(blob);

  i = ops.size();

  munchLeadingZeros(t);

  if(t.length()==0)
    return i;

  if(t.c_str()[0]!='P') {
    g=2;
    for(unsigned p=0; p<t.length(); p++) {
      bool isP=(t.c_str()[p]=='P');
      bool isN=(t.c_str()[p]=='N');

      if(isP)
	t.replace(p, 1, "N");
      if(isN)
	t.replace(p, 1, "P");
    }
  }

  int j = lefevre(t, ops);

  int trail=countTrailingZeros(u);
  blob.secondVal=i; blob.secondShift=trail;

  trail=countTrailingZeros(t);
  blob.firstVal=j; blob.firstShift=trail;

  switch(g) {
    case 0:
      blob.isSub=false; // first + second
      break;
    case 1:
      blob.isSub=true; // first - second
      break;
    case 2:
      blob.isSub=true; // second - first
      int tmpval, tmpshift;
      tmpval=blob.firstVal;
      tmpshift=blob.firstShift;
      blob.firstVal=blob.secondVal;
      blob.firstShift=blob.secondShift;
      blob.secondVal=tmpval;
      blob.secondShift=tmpshift;
      break;
      //assert
  }
 
  ops.push_back(blob);
  return ops.size();
}

SDOperand ISel::BuildConstmulSequence(SDOperand N) {
  //FIXME: we should shortcut this stuff for multiplies by 2^n+1
  //       in particular, *3 is nicer as *2+1, not *4-1
  int64_t constant=cast<ConstantSDNode>(N.getOperand(1))->getValue();

  bool flippedSign;
  unsigned preliminaryShift=0;

  assert(constant != 0 && "erk, you're trying to multiply by constant zero\n");

  // first, we make the constant to multiply by positive
  if(constant<0) {
    constant=-constant;
    flippedSign=true;
  } else {
    flippedSign=false;
  }

  // next, we make it odd.
  for(; (constant%2==0); preliminaryShift++)
    constant>>=1;

  //OK, we have a positive, odd number of 64 bits or less. Convert it
  //to a binary string, constantString[0] is the LSB
  char constantString[65];
  for(int i=0; i<64; i++)
    constantString[i]='0'+((constant>>i)&0x1);
  constantString[64]=0;

  // now, Booth encode it
  std::string boothEncodedString;
  boothEncode(constantString, boothEncodedString);

  std::vector<struct shiftaddblob> ops;
  // do the transformation, filling out 'ops'
  lefevre(boothEncodedString, ops);

  assert(ops.size() < 80 && "constmul code has gone haywire\n");
  SDOperand results[80]; // temporary results (of adds/subs of shifts)
  
  // now turn 'ops' into DAG bits
  for(unsigned i=0; i<ops.size(); i++) {
    SDOperand amt = ISelDAG->getConstant(ops[i].firstShift, MVT::i64);
    SDOperand val = (ops[i].firstVal == 0) ? N.getOperand(0) :
      results[ops[i].firstVal-1];
    SDOperand left = ISelDAG->getNode(ISD::SHL, MVT::i64, val, amt);
    amt = ISelDAG->getConstant(ops[i].secondShift, MVT::i64);
    val = (ops[i].secondVal == 0) ? N.getOperand(0) :
      results[ops[i].secondVal-1];
    SDOperand right = ISelDAG->getNode(ISD::SHL, MVT::i64, val, amt);
    if(ops[i].isSub)
      results[i] = ISelDAG->getNode(ISD::SUB, MVT::i64, left, right);
    else
      results[i] = ISelDAG->getNode(ISD::ADD, MVT::i64, left, right);
  }

  // don't forget flippedSign and preliminaryShift!
  SDOperand shiftedresult;
  if(preliminaryShift) {
    SDOperand finalshift = ISelDAG->getConstant(preliminaryShift, MVT::i64);
    shiftedresult = ISelDAG->getNode(ISD::SHL, MVT::i64,
	results[ops.size()-1], finalshift);
  } else { // there was no preliminary divide-by-power-of-2 required
    shiftedresult = results[ops.size()-1];
  }
 
  SDOperand finalresult;
  if(flippedSign) { // if we were multiplying by a negative constant:
    SDOperand zero = ISelDAG->getConstant(0, MVT::i64);
    // subtract the result from 0 to flip its sign
    finalresult = ISelDAG->getNode(ISD::SUB, MVT::i64, zero, shiftedresult);
  } else { // there was no preliminary multiply by -1 required
    finalresult = shiftedresult;
  }
  
  return finalresult; 
}

/// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
/// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(uint64_t Val) {
  if (Val == 0 || (Val & (Val-1))) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    Val >>= 1;
    ++Count;
  }
  return Count;
}

/// ExactLog2sub1 - This function solves for (Val == (1 << (N-1))-1)
/// and returns N.  It returns 666 if Val is not 2^n -1 for some n.
static unsigned ExactLog2sub1(uint64_t Val) {
  unsigned int n;
  for(n=0; n<64; n++) {
    if(Val==(uint64_t)((1LL<<n)-1))
      return n;
  }
  return 666;
}

/// ponderIntegerDivisionBy - When handling integer divides, if the divide
/// is by a constant such that we can efficiently codegen it, this
/// function says what to do. Currently, it returns 0 if the division must
/// become a genuine divide, and 1 if the division can be turned into a
/// right shift.
static unsigned ponderIntegerDivisionBy(SDOperand N, bool isSigned,
                                      unsigned& Imm) {
  if (N.getOpcode() != ISD::Constant) return 0; // if not a divide by
                                                // a constant, give up.

  int64_t v = (int64_t)cast<ConstantSDNode>(N)->getSignExtended();

  if ((Imm = ExactLog2(v))) { // if a division by a power of two, say so
    return 1;
  }

  return 0; // fallthrough
}

static unsigned ponderIntegerAndWith(SDOperand N, unsigned& Imm) {
  if (N.getOpcode() != ISD::Constant) return 0; // if not ANDing with
                                                // a constant, give up.

  int64_t v = (int64_t)cast<ConstantSDNode>(N)->getSignExtended();

  if ((Imm = ExactLog2sub1(v))!=666) { // if ANDing with ((2^n)-1) for some n
    return 1; // say so
  }

  return 0; // fallthrough
}

static unsigned ponderIntegerAdditionWith(SDOperand N, unsigned& Imm) {
  if (N.getOpcode() != ISD::Constant) return 0; // if not adding a
                                                // constant, give up.
  int64_t v = (int64_t)cast<ConstantSDNode>(N)->getSignExtended();

  if (v <= 8191 && v >= -8192) { // if this constants fits in 14 bits, say so
    Imm = v & 0x3FFF; // 14 bits
    return 1;
  }
  return 0; // fallthrough
}

static unsigned ponderIntegerSubtractionFrom(SDOperand N, unsigned& Imm) {
  if (N.getOpcode() != ISD::Constant) return 0; // if not subtracting a
                                                // constant, give up.
  int64_t v = (int64_t)cast<ConstantSDNode>(N)->getSignExtended();

  if (v <= 127 && v >= -128) { // if this constants fits in 8 bits, say so
    Imm = v & 0xFF; // 8 bits
    return 1;
  }
  return 0; // fallthrough
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  MVT::ValueType DestType = N.getValueType();

  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  SDOperand Op0, Op1;

  if (Node->getOpcode() == ISD::CopyFromReg)
    // Just use the specified register as our input.
    return dyn_cast<RegSDNode>(Node)->getReg();

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  if (N.getOpcode() != ISD::CALL && N.getOpcode() != ISD::TAILCALL)
    Reg = Result = (N.getValueType() != MVT::Other) ?
      MakeReg(N.getValueType()) : 1;
  else {
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    if (Node->getNumValues() == 1)
      Reg = Result = 1;  // Void call, just a chain.
    else {
      Result = MakeReg(Node->getValueType(0));
      ExprMap[N.getValue(0)] = Result;
      for (unsigned i = 1, e = N.Val->getNumValues()-1; i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
      ExprMap[SDOperand(Node, Node->getNumValues()-1)] = 1;
    }
  }

  switch (N.getOpcode()) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");

  case ISD::FrameIndex: {
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    BuildMI(BB, IA64::MOV, 1, Result).addFrameIndex(Tmp1);
    return Result;
  }

  case ISD::ConstantPool: {
    Tmp1 = cast<ConstantPoolSDNode>(N)->getIndex();
    IA64Lowering.restoreGP(BB); // FIXME: do i really need this?
    BuildMI(BB, IA64::ADD, 2, Result).addConstantPoolIndex(Tmp1)
      .addReg(IA64::r1);
    return Result;
  }

  case ISD::ConstantFP: {
    Tmp1 = Result;   // Intermediate Register
    if (cast<ConstantFPSDNode>(N)->getValue() < 0.0 ||
        cast<ConstantFPSDNode>(N)->isExactlyValue(-0.0))
      Tmp1 = MakeReg(MVT::f64);

    if (cast<ConstantFPSDNode>(N)->isExactlyValue(+0.0) ||
        cast<ConstantFPSDNode>(N)->isExactlyValue(-0.0))
      BuildMI(BB, IA64::FMOV, 1, Tmp1).addReg(IA64::F0); // load 0.0
    else if (cast<ConstantFPSDNode>(N)->isExactlyValue(+1.0) ||
             cast<ConstantFPSDNode>(N)->isExactlyValue(-1.0))
      BuildMI(BB, IA64::FMOV, 1, Tmp1).addReg(IA64::F1); // load 1.0
    else
      assert(0 && "Unexpected FP constant!");
    if (Tmp1 != Result)
      // we multiply by +1.0, negate (this is FNMA), and then add 0.0
      BuildMI(BB, IA64::FNMA, 3, Result).addReg(Tmp1).addReg(IA64::F1)
        .addReg(IA64::F0);
    return Result;
  }

  case ISD::DYNAMIC_STACKALLOC: {
    // Generate both result values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    // FIXME: We are currently ignoring the requested alignment for handling
    // greater than the stack alignment.  This will need to be revisited at some
    // point.  Align = N.getOperand(2);

    if (!isa<ConstantSDNode>(N.getOperand(2)) ||
        cast<ConstantSDNode>(N.getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }

/*
    Select(N.getOperand(0));
    if (ConstantSDNode* CN = dyn_cast<ConstantSDNode>(N.getOperand(1)))
    {
      if (CN->getValue() < 32000)
      {
        BuildMI(BB, IA64::ADDIMM22, 2, IA64::r12).addReg(IA64::r12)
          .addImm(-CN->getValue());
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        // Subtract size from stack pointer, thereby allocating some space.
        BuildMI(BB, IA64::SUB, 2, IA64::r12).addReg(IA64::r12).addReg(Tmp1);
      }
    } else {
      Tmp1 = SelectExpr(N.getOperand(1));
      // Subtract size from stack pointer, thereby allocating some space.
      BuildMI(BB, IA64::SUB, 2, IA64::r12).addReg(IA64::r12).addReg(Tmp1);
    }
*/
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    // Subtract size from stack pointer, thereby allocating some space.
    BuildMI(BB, IA64::SUB, 2, IA64::r12).addReg(IA64::r12).addReg(Tmp1);
    // Put a pointer to the space into the result register, by copying the
    // stack pointer.
    BuildMI(BB, IA64::MOV, 1, Result).addReg(IA64::r12);
    return Result;
  }

  case ISD::SELECT: {
      Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE

      unsigned bogoResult;

      switch (N.getOperand(1).getValueType()) {
        default: assert(0 &&
        "ISD::SELECT: 'select'ing something other than i1, i64 or f64!\n");
        // for i1, we load the condition into an integer register, then
        // conditionally copy Tmp2 and Tmp3 to Tmp1 in parallel (only one
        // of them will go through, since the integer register will hold
        // either 0 or 1)
        case MVT::i1: {
          bogoResult=MakeReg(MVT::i1);

          // load the condition into an integer register
          unsigned condReg=MakeReg(MVT::i64);
          unsigned dummy=MakeReg(MVT::i64);
          BuildMI(BB, IA64::MOV, 1, dummy).addReg(IA64::r0);
          BuildMI(BB, IA64::TPCADDIMM22, 2, condReg).addReg(dummy)
            .addImm(1).addReg(Tmp1);

          // initialize Result (bool) to false (hence UNC) and if
          // the select condition (condReg) is false (0), copy Tmp3
          BuildMI(BB, IA64::PCMPEQUNC, 3, bogoResult)
            .addReg(condReg).addReg(IA64::r0).addReg(Tmp3);

          // now, if the selection condition is true, write 1 to the
          // result if Tmp2 is 1
          BuildMI(BB, IA64::TPCMPNE, 3, Result).addReg(bogoResult)
            .addReg(condReg).addReg(IA64::r0).addReg(Tmp2);
          break;
        }
        // for i64/f64, we just copy Tmp3 and then conditionally overwrite it
        // with Tmp2 if Tmp1 is true
        case MVT::i64:
          bogoResult=MakeReg(MVT::i64);
          BuildMI(BB, IA64::MOV, 1, bogoResult).addReg(Tmp3);
          BuildMI(BB, IA64::CMOV, 2, Result).addReg(bogoResult).addReg(Tmp2)
            .addReg(Tmp1);
          break;
        case MVT::f64:
          bogoResult=MakeReg(MVT::f64);
          BuildMI(BB, IA64::FMOV, 1, bogoResult).addReg(Tmp3);
          BuildMI(BB, IA64::CFMOV, 2, Result).addReg(bogoResult).addReg(Tmp2)
            .addReg(Tmp1);
          break;
      }
      
      return Result;
  }

  case ISD::Constant: {
    unsigned depositPos=0;
    unsigned depositLen=0;
    switch (N.getValueType()) {
      default: assert(0 && "Cannot use constants of this type!");
      case MVT::i1: { // if a bool, we don't 'load' so much as generate
        // the constant:
        if(cast<ConstantSDNode>(N)->getValue())  // true:
          BuildMI(BB, IA64::CMPEQ, 2, Result).addReg(IA64::r0).addReg(IA64::r0);
        else // false:
          BuildMI(BB, IA64::CMPNE, 2, Result).addReg(IA64::r0).addReg(IA64::r0);
        return Result; // early exit
      }
      case MVT::i64: break;
    }

    int64_t immediate = cast<ConstantSDNode>(N)->getValue();

    if(immediate==0) { // if the constant is just zero,
      BuildMI(BB, IA64::MOV, 1, Result).addReg(IA64::r0); // just copy r0
      return Result; // early exit
    }

    if (immediate <= 8191 && immediate >= -8192) {
      // if this constants fits in 14 bits, we use a mov the assembler will
      // turn into:   "adds rDest=imm,r0"  (and _not_ "andl"...)
      BuildMI(BB, IA64::MOVSIMM14, 1, Result).addSImm(immediate);
      return Result; // early exit
    }

    if (immediate <= 2097151 && immediate >= -2097152) {
      // if this constants fits in 22 bits, we use a mov the assembler will
      // turn into:   "addl rDest=imm,r0"
      BuildMI(BB, IA64::MOVSIMM22, 1, Result).addSImm(immediate);
      return Result; // early exit
    }

    /* otherwise, our immediate is big, so we use movl */
    uint64_t Imm = immediate;
    BuildMI(BB, IA64::MOVLIMM64, 1, Result).addImm64(Imm);
    return Result;
  }

  case ISD::UNDEF: {
    BuildMI(BB, IA64::IDEF, 0, Result);
    return Result;
  }

  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    unsigned Tmp1 = MakeReg(MVT::i64);

    BuildMI(BB, IA64::ADD, 2, Tmp1).addGlobalAddress(GV).addReg(IA64::r1);
    BuildMI(BB, IA64::LD8, 1, Result).addReg(Tmp1);

    return Result;
  }

  case ISD::ExternalSymbol: {
    const char *Sym = cast<ExternalSymbolSDNode>(N)->getSymbol();
// assert(0 && "sorry, but what did you want an ExternalSymbol for again?");
    BuildMI(BB, IA64::MOV, 1, Result).addExternalSymbol(Sym); // XXX
    return Result;
  }

  case ISD::FP_EXTEND: {
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, IA64::FMOV, 1, Result).addReg(Tmp1);
    return Result;
  }

  case ISD::ZERO_EXTEND: {
    Tmp1 = SelectExpr(N.getOperand(0)); // value

    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Cannot zero-extend this type!");
    case MVT::i8:  Opc = IA64::ZXT1; break;
    case MVT::i16: Opc = IA64::ZXT2; break;
    case MVT::i32: Opc = IA64::ZXT4; break;

    // we handle bools differently! :
    case MVT::i1: { // if the predicate reg has 1, we want a '1' in our GR.
      unsigned dummy = MakeReg(MVT::i64);
      // first load zero:
      BuildMI(BB, IA64::MOV, 1, dummy).addReg(IA64::r0);
      // ...then conditionally (PR:Tmp1) add 1:
      BuildMI(BB, IA64::TPCADDIMM22, 2, Result).addReg(dummy)
        .addImm(1).addReg(Tmp1);
      return Result; // XXX early exit!
    }
    }

    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
   }

  case ISD::SIGN_EXTEND: {   // we should only have to handle i1 -> i64 here!!!

assert(0 && "hmm, ISD::SIGN_EXTEND: shouldn't ever be reached. bad luck!\n");

    Tmp1 = SelectExpr(N.getOperand(0)); // value

    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Cannot sign-extend this type!");
    case MVT::i1:  assert(0 && "trying to sign extend a bool? ow.\n");
      Opc = IA64::SXT1; break;
      // FIXME: for now, we treat bools the same as i8s
    case MVT::i8:  Opc = IA64::SXT1; break;
    case MVT::i16: Opc = IA64::SXT2; break;
    case MVT::i32: Opc = IA64::SXT4; break;
    }

    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
   }

  case ISD::TRUNCATE: {
    // we use the funky dep.z (deposit (zero)) instruction to deposit bits
    // of R0 appropriately.
    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Unknown truncate!");
    case MVT::i64: break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    unsigned depositPos, depositLen;

    switch (N.getValueType()) {
    default: assert(0 && "Unknown truncate!");
    case MVT::i1: {
      // if input (normal reg) is 0, 0!=0 -> false (0), if 1, 1!=0 ->true (1):
        BuildMI(BB, IA64::CMPNE, 2, Result).addReg(Tmp1)
          .addReg(IA64::r0);
        return Result; // XXX early exit!
      }
    case MVT::i8:  depositPos=0; depositLen=8;  break;
    case MVT::i16: depositPos=0; depositLen=16; break;
    case MVT::i32: depositPos=0; depositLen=32; break;
    }
    BuildMI(BB, IA64::DEPZ, 1, Result).addReg(Tmp1)
      .addImm(depositPos).addImm(depositLen);
    return Result;
  }

/*
  case ISD::FP_ROUND: {
    assert (DestType == MVT::f32 && N.getOperand(0).getValueType() == MVT::f64 &&
  "error: trying to FP_ROUND something other than f64 -> f32!\n");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, IA64::FADDS, 2, Result).addReg(Tmp1).addReg(IA64::F0);
    // we add 0.0 using a single precision add to do rounding
    return Result;
  }
*/

// FIXME: the following 4 cases need cleaning
  case ISD::SINT_TO_FP: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    unsigned dummy = MakeReg(MVT::f64);
    BuildMI(BB, IA64::SETFSIG, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::FCVTXF, 1, dummy).addReg(Tmp2);
    BuildMI(BB, IA64::FNORMD, 1, Result).addReg(dummy);
    return Result;
  }

  case ISD::UINT_TO_FP: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    unsigned dummy = MakeReg(MVT::f64);
    BuildMI(BB, IA64::SETFSIG, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::FCVTXUF, 1, dummy).addReg(Tmp2);
    BuildMI(BB, IA64::FNORMD, 1, Result).addReg(dummy);
    return Result;
  }

  case ISD::FP_TO_SINT: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    BuildMI(BB, IA64::FCVTFXTRUNC, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(Tmp2);
    return Result;
  }

  case ISD::FP_TO_UINT: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    BuildMI(BB, IA64::FCVTFXUTRUNC, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(Tmp2);
    return Result;
  }

  case ISD::ADD: {
    if(DestType == MVT::f64 && N.getOperand(0).getOpcode() == ISD::MUL &&
       N.getOperand(0).Val->hasOneUse()) { // if we can fold this add
                                           // into an fma, do so:
      // ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::FMA, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
      return Result; // early exit
    }

    if(DestType != MVT::f64 && N.getOperand(0).getOpcode() == ISD::SHL &&
        N.getOperand(0).Val->hasOneUse()) { // if we might be able to fold
                                            // this add into a shladd, try:
      ConstantSDNode *CSD = NULL;
      if((CSD = dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(1))) &&
          (CSD->getValue() >= 1) && (CSD->getValue() <= 4) ) { // we can:

        // ++FusedSHLADD; // Statistic
        Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
        int shl_amt = CSD->getValue();
        Tmp3 = SelectExpr(N.getOperand(1));
        
        BuildMI(BB, IA64::SHLADD, 3, Result)
          .addReg(Tmp1).addImm(shl_amt).addReg(Tmp3);
        return Result; // early exit
      }
    }

    //else, fallthrough:
    Tmp1 = SelectExpr(N.getOperand(0));
    if(DestType != MVT::f64) { // integer addition:
        switch (ponderIntegerAdditionWith(N.getOperand(1), Tmp3)) {
          case 1: // adding a constant that's 14 bits
            BuildMI(BB, IA64::ADDIMM14, 2, Result).addReg(Tmp1).addSImm(Tmp3);
            return Result; // early exit
        } // fallthrough and emit a reg+reg ADD:
        Tmp2 = SelectExpr(N.getOperand(1));
        BuildMI(BB, IA64::ADD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    } else { // this is a floating point addition
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::FADD, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;
  }

  case ISD::MUL: {

    if(DestType != MVT::f64) { // TODO: speed!
      if(N.getOperand(1).getOpcode() != ISD::Constant) { // if not a const mul
	// boring old integer multiply with xma
	Tmp1 = SelectExpr(N.getOperand(0));
	Tmp2 = SelectExpr(N.getOperand(1));

	unsigned TempFR1=MakeReg(MVT::f64);
	unsigned TempFR2=MakeReg(MVT::f64);
	unsigned TempFR3=MakeReg(MVT::f64);
	BuildMI(BB, IA64::SETFSIG, 1, TempFR1).addReg(Tmp1);
	BuildMI(BB, IA64::SETFSIG, 1, TempFR2).addReg(Tmp2);
	BuildMI(BB, IA64::XMAL, 1, TempFR3).addReg(TempFR1).addReg(TempFR2)
	  .addReg(IA64::F0);
	BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(TempFR3);
	return Result; // early exit
      } else { // we are multiplying by an integer constant! yay
	return Reg = SelectExpr(BuildConstmulSequence(N)); // avert your eyes!
      }
    }
    else { // floating point multiply
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::FMPY, 2, Result).addReg(Tmp1).addReg(Tmp2);
      return Result;
    }
  }

  case ISD::SUB: {
    if(DestType == MVT::f64 && N.getOperand(0).getOpcode() == ISD::MUL &&
       N.getOperand(0).Val->hasOneUse()) { // if we can fold this sub
                                           // into an fms, do so:
      // ++FusedFP; // Statistic
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(0).getOperand(1));
      Tmp3 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::FMS, 3, Result).addReg(Tmp1).addReg(Tmp2).addReg(Tmp3);
      return Result; // early exit
    }
    Tmp2 = SelectExpr(N.getOperand(1));
    if(DestType != MVT::f64) { // integer subtraction:
        switch (ponderIntegerSubtractionFrom(N.getOperand(0), Tmp3)) {
          case 1: // subtracting *from* an 8 bit constant:
            BuildMI(BB, IA64::SUBIMM8, 2, Result).addSImm(Tmp3).addReg(Tmp2);
            return Result; // early exit
        } // fallthrough and emit a reg+reg SUB:
        Tmp1 = SelectExpr(N.getOperand(0));
        BuildMI(BB, IA64::SUB, 2, Result).addReg(Tmp1).addReg(Tmp2);
    } else { // this is a floating point subtraction
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, IA64::FSUB, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;
  }

  case ISD::FABS: {
    Tmp1 = SelectExpr(N.getOperand(0));
    assert(DestType == MVT::f64 && "trying to fabs something other than f64?");
    BuildMI(BB, IA64::FABS, 1, Result).addReg(Tmp1);
    return Result;
  }

  case ISD::FNEG: {
    assert(DestType == MVT::f64 && "trying to fneg something other than f64?");

    if (ISD::FABS == N.getOperand(0).getOpcode()) { // && hasOneUse()?
      Tmp1 = SelectExpr(N.getOperand(0).getOperand(0));
      BuildMI(BB, IA64::FNEGABS, 1, Result).addReg(Tmp1); // fold in abs
    } else {
      Tmp1 = SelectExpr(N.getOperand(0));
      BuildMI(BB, IA64::FNEG, 1, Result).addReg(Tmp1); // plain old fneg
    }

    return Result;
  }

  case ISD::AND: {
     switch (N.getValueType()) {
    default: assert(0 && "Cannot AND this type!");
    case MVT::i1: { // if a bool, we emit a pseudocode AND
      unsigned pA = SelectExpr(N.getOperand(0));
      unsigned pB = SelectExpr(N.getOperand(1));

/* our pseudocode for AND is:
 *
(pA) cmp.eq.unc pC,p0 = r0,r0   // pC = pA
     cmp.eq pTemp,p0 = r0,r0    // pTemp = NOT pB
     ;;
(pB) cmp.ne pTemp,p0 = r0,r0
     ;;
(pTemp)cmp.ne pC,p0 = r0,r0    // if (NOT pB) pC = 0

*/
      unsigned pTemp = MakeReg(MVT::i1);

      unsigned bogusTemp1 = MakeReg(MVT::i1);
      unsigned bogusTemp2 = MakeReg(MVT::i1);
      unsigned bogusTemp3 = MakeReg(MVT::i1);
      unsigned bogusTemp4 = MakeReg(MVT::i1);

      BuildMI(BB, IA64::PCMPEQUNC, 3, bogusTemp1)
        .addReg(IA64::r0).addReg(IA64::r0).addReg(pA);
      BuildMI(BB, IA64::CMPEQ, 2, bogusTemp2)
        .addReg(IA64::r0).addReg(IA64::r0);
      BuildMI(BB, IA64::TPCMPNE, 3, pTemp)
        .addReg(bogusTemp2).addReg(IA64::r0).addReg(IA64::r0).addReg(pB);
      BuildMI(BB, IA64::TPCMPNE, 3, Result)
        .addReg(bogusTemp1).addReg(IA64::r0).addReg(IA64::r0).addReg(pTemp);
      break;
    }

    // if not a bool, we just AND away:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64: {
      Tmp1 = SelectExpr(N.getOperand(0));
      switch (ponderIntegerAndWith(N.getOperand(1), Tmp3)) {
        case 1: // ANDing a constant that is 2^n-1 for some n
          switch (Tmp3) {
            case 8:  // if AND 0x00000000000000FF, be quaint and use zxt1
              BuildMI(BB, IA64::ZXT1, 1, Result).addReg(Tmp1);
              break;
            case 16: // if AND 0x000000000000FFFF, be quaint and use zxt2
              BuildMI(BB, IA64::ZXT2, 1, Result).addReg(Tmp1);
              break;
            case 32: // if AND 0x00000000FFFFFFFF, be quaint and use zxt4
              BuildMI(BB, IA64::ZXT4, 1, Result).addReg(Tmp1);
              break;
            default: // otherwise, use dep.z to paste zeros
              BuildMI(BB, IA64::DEPZ, 3, Result).addReg(Tmp1)
                .addImm(0).addImm(Tmp3);
              break;
          }
          return Result; // early exit
      } // fallthrough and emit a simple AND:
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::AND, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    }
    return Result;
  }

  case ISD::OR: {
  switch (N.getValueType()) {
    default: assert(0 && "Cannot OR this type!");
    case MVT::i1: { // if a bool, we emit a pseudocode OR
      unsigned pA = SelectExpr(N.getOperand(0));
      unsigned pB = SelectExpr(N.getOperand(1));

      unsigned pTemp1 = MakeReg(MVT::i1);

/* our pseudocode for OR is:
 *

pC = pA OR pB
-------------

(pA) cmp.eq.unc pC,p0 = r0,r0  // pC = pA
 ;;
(pB) cmp.eq pC,p0 = r0,r0 // if (pB) pC = 1

*/
      BuildMI(BB, IA64::PCMPEQUNC, 3, pTemp1)
        .addReg(IA64::r0).addReg(IA64::r0).addReg(pA);
      BuildMI(BB, IA64::TPCMPEQ, 3, Result)
        .addReg(pTemp1).addReg(IA64::r0).addReg(IA64::r0).addReg(pB);
      break;
    }
    // if not a bool, we just OR away:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64: {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::OR, 2, Result).addReg(Tmp1).addReg(Tmp2);
      break;
    }
    }
    return Result;
  }

  case ISD::XOR: {
     switch (N.getValueType()) {
    default: assert(0 && "Cannot XOR this type!");
    case MVT::i1: { // if a bool, we emit a pseudocode XOR
      unsigned pY = SelectExpr(N.getOperand(0));
      unsigned pZ = SelectExpr(N.getOperand(1));

/* one possible routine for XOR is:

      // Compute px = py ^ pz
        // using sum of products: px = (py & !pz) | (pz & !py)
        // Uses 5 instructions in 3 cycles.
        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(py)    cmp.eq.unc      pt = r0, r0     // pt = py
        ;;
        // cycle 2
(pt)    cmp.ne.and      px = r0, r0     // px = px & !pt (px = pz & !pt)
(pz)    cmp.ne.and      pt = r0, r0     // pt = pt & !pz
        ;;
        } { .mmi
        // cycle 3
(pt)    cmp.eq.or       px = r0, r0     // px = px | pt

*** Another, which we use here, requires one scratch GR. it is:

        mov             rt = 0          // initialize rt off critical path
        ;;

        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(pz)    mov             rt = 1          // rt = pz
        ;;
        // cycle 2
(py)    cmp.ne          px = 1, rt      // if (py) px = !pz

.. these routines kindly provided by Jim Hull
*/
      unsigned rt = MakeReg(MVT::i64);

      // these two temporaries will never actually appear,
      // due to the two-address form of some of the instructions below
      unsigned bogoPR = MakeReg(MVT::i1);  // becomes Result
      unsigned bogoGR = MakeReg(MVT::i64); // becomes rt

      BuildMI(BB, IA64::MOV, 1, bogoGR).addReg(IA64::r0);
      BuildMI(BB, IA64::PCMPEQUNC, 3, bogoPR)
        .addReg(IA64::r0).addReg(IA64::r0).addReg(pZ);
      BuildMI(BB, IA64::TPCADDIMM22, 2, rt)
        .addReg(bogoGR).addImm(1).addReg(pZ);
      BuildMI(BB, IA64::TPCMPIMM8NE, 3, Result)
        .addReg(bogoPR).addImm(1).addReg(rt).addReg(pY);
      break;
    }
    // if not a bool, we just XOR away:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64: {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::XOR, 2, Result).addReg(Tmp1).addReg(Tmp2);
      break;
    }
    }
    return Result;
  }

  case ISD::CTPOP: {
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, IA64::POPCNT, 1, Result).addReg(Tmp1);
    return Result;
  }

  case ISD::SHL: {
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue();
      BuildMI(BB, IA64::SHLI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::SHL, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;
  }

  case ISD::SRL: {
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue();
      BuildMI(BB, IA64::SHRUI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::SHRU, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;
  }

  case ISD::SRA: {
    Tmp1 = SelectExpr(N.getOperand(0));
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      Tmp2 = CN->getValue();
      BuildMI(BB, IA64::SHRSI, 2, Result).addReg(Tmp1).addImm(Tmp2);
    } else {
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::SHRS, 2, Result).addReg(Tmp1).addReg(Tmp2);
    }
    return Result;
  }

  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM: {

    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));

    bool isFP=false;

    if(DestType == MVT::f64) // XXX: we're not gonna be fed MVT::f32, are we?
      isFP=true;

    bool isModulus=false; // is it a division or a modulus?
    bool isSigned=false;

    switch(N.getOpcode()) {
      case ISD::SDIV:  isModulus=false; isSigned=true;  break;
      case ISD::UDIV:  isModulus=false; isSigned=false; break;
      case ISD::SREM:  isModulus=true;  isSigned=true;  break;
      case ISD::UREM:  isModulus=true;  isSigned=false; break;
    }

    if(!isModulus && !isFP) { // if this is an integer divide,
      switch (ponderIntegerDivisionBy(N.getOperand(1), isSigned, Tmp3)) {
        case 1: // division by a constant that's a power of 2
          Tmp1 = SelectExpr(N.getOperand(0));
          if(isSigned) {  // argument could be negative, so emit some code:
            unsigned divAmt=Tmp3;
            unsigned tempGR1=MakeReg(MVT::i64);
            unsigned tempGR2=MakeReg(MVT::i64);
            unsigned tempGR3=MakeReg(MVT::i64);
            BuildMI(BB, IA64::SHRS, 2, tempGR1)
              .addReg(Tmp1).addImm(divAmt-1);
            BuildMI(BB, IA64::EXTRU, 3, tempGR2)
              .addReg(tempGR1).addImm(64-divAmt).addImm(divAmt);
            BuildMI(BB, IA64::ADD, 2, tempGR3)
              .addReg(Tmp1).addReg(tempGR2);
            BuildMI(BB, IA64::SHRS, 2, Result)
              .addReg(tempGR3).addImm(divAmt);
          }
          else // unsigned div-by-power-of-2 becomes a simple shift right:
            BuildMI(BB, IA64::SHRU, 2, Result).addReg(Tmp1).addImm(Tmp3);
          return Result; // early exit
      }
    }

    unsigned TmpPR=MakeReg(MVT::i1);  // we need two scratch
    unsigned TmpPR2=MakeReg(MVT::i1); // predicate registers,
    unsigned TmpF1=MakeReg(MVT::f64); // and one metric truckload of FP regs.
    unsigned TmpF2=MakeReg(MVT::f64); // lucky we have IA64?
    unsigned TmpF3=MakeReg(MVT::f64); // well, the real FIXME is to have
    unsigned TmpF4=MakeReg(MVT::f64); // isTwoAddress forms of these
    unsigned TmpF5=MakeReg(MVT::f64); // FP instructions so we can end up with
    unsigned TmpF6=MakeReg(MVT::f64); // stuff like setf.sig f10=f10 etc.
    unsigned TmpF7=MakeReg(MVT::f64);
    unsigned TmpF8=MakeReg(MVT::f64);
    unsigned TmpF9=MakeReg(MVT::f64);
    unsigned TmpF10=MakeReg(MVT::f64);
    unsigned TmpF11=MakeReg(MVT::f64);
    unsigned TmpF12=MakeReg(MVT::f64);
    unsigned TmpF13=MakeReg(MVT::f64);
    unsigned TmpF14=MakeReg(MVT::f64);
    unsigned TmpF15=MakeReg(MVT::f64);

    // OK, emit some code:

    if(!isFP) {
      // first, load the inputs into FP regs.
      BuildMI(BB, IA64::SETFSIG, 1, TmpF1).addReg(Tmp1);
      BuildMI(BB, IA64::SETFSIG, 1, TmpF2).addReg(Tmp2);

      // next, convert the inputs to FP
      if(isSigned) {
        BuildMI(BB, IA64::FCVTXF, 1, TmpF3).addReg(TmpF1);
        BuildMI(BB, IA64::FCVTXF, 1, TmpF4).addReg(TmpF2);
      } else {
        BuildMI(BB, IA64::FCVTXUFS1, 1, TmpF3).addReg(TmpF1);
        BuildMI(BB, IA64::FCVTXUFS1, 1, TmpF4).addReg(TmpF2);
      }

    } else { // this is an FP divide/remainder, so we 'leak' some temp
             // regs and assign TmpF3=Tmp1, TmpF4=Tmp2
      TmpF3=Tmp1;
      TmpF4=Tmp2;
    }

    // we start by computing an approximate reciprocal (good to 9 bits?)
    // note, this instruction writes _both_ TmpF5 (answer) and TmpPR (predicate)
    BuildMI(BB, IA64::FRCPAS1, 4)
      .addReg(TmpF5, MachineOperand::Def)
      .addReg(TmpPR, MachineOperand::Def)
      .addReg(TmpF3).addReg(TmpF4);

    if(!isModulus) { // if this is a divide, we worry about div-by-zero
      unsigned bogusPR=MakeReg(MVT::i1); // won't appear, due to twoAddress
                                       // TPCMPNE below
      BuildMI(BB, IA64::CMPEQ, 2, bogusPR).addReg(IA64::r0).addReg(IA64::r0);
      BuildMI(BB, IA64::TPCMPNE, 3, TmpPR2).addReg(bogusPR)
        .addReg(IA64::r0).addReg(IA64::r0).addReg(TmpPR);
    }

    // now we apply newton's method, thrice! (FIXME: this is ~72 bits of
    // precision, don't need this much for f32/i32)
    BuildMI(BB, IA64::CFNMAS1, 4, TmpF6)
      .addReg(TmpF4).addReg(TmpF5).addReg(IA64::F1).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4, TmpF7)
      .addReg(TmpF3).addReg(TmpF5).addReg(IA64::F0).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4, TmpF8)
      .addReg(TmpF6).addReg(TmpF6).addReg(IA64::F0).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4, TmpF9)
      .addReg(TmpF6).addReg(TmpF7).addReg(TmpF7).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF10)
      .addReg(TmpF6).addReg(TmpF5).addReg(TmpF5).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF11)
      .addReg(TmpF8).addReg(TmpF9).addReg(TmpF9).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF12)
      .addReg(TmpF8).addReg(TmpF10).addReg(TmpF10).addReg(TmpPR);
    BuildMI(BB, IA64::CFNMAS1, 4,TmpF13)
      .addReg(TmpF4).addReg(TmpF11).addReg(TmpF3).addReg(TmpPR);

       // FIXME: this is unfortunate :(
       // the story is that the dest reg of the fnma above and the fma below
       // (and therefore possibly the src of the fcvt.fx[u] as well) cannot
       // be the same register, or this code breaks if the first argument is
       // zero. (e.g. without this hack, 0%8 yields -64, not 0.)
    BuildMI(BB, IA64::CFMAS1,  4,TmpF14)
      .addReg(TmpF13).addReg(TmpF12).addReg(TmpF11).addReg(TmpPR);

    if(isModulus) { // XXX: fragile! fixes _only_ mod, *breaks* div! !
      BuildMI(BB, IA64::IUSE, 1).addReg(TmpF13); // hack :(
    }

    if(!isFP) {
      // round to an integer
      if(isSigned)
        BuildMI(BB, IA64::FCVTFXTRUNCS1, 1, TmpF15).addReg(TmpF14);
      else
        BuildMI(BB, IA64::FCVTFXUTRUNCS1, 1, TmpF15).addReg(TmpF14);
    } else {
      BuildMI(BB, IA64::FMOV, 1, TmpF15).addReg(TmpF14);
     // EXERCISE: can you see why TmpF15=TmpF14 does not work here, and
     // we really do need the above FMOV? ;)
    }

    if(!isModulus) {
      if(isFP) { // extra worrying about div-by-zero
      unsigned bogoResult=MakeReg(MVT::f64);

      // we do a 'conditional fmov' (of the correct result, depending
      // on how the frcpa predicate turned out)
      BuildMI(BB, IA64::PFMOV, 2, bogoResult)
        .addReg(TmpF12).addReg(TmpPR2);
      BuildMI(BB, IA64::CFMOV, 2, Result)
        .addReg(bogoResult).addReg(TmpF15).addReg(TmpPR);
      }
      else {
        BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(TmpF15);
      }
    } else { // this is a modulus
      if(!isFP) {
        // answer = q * (-b) + a
        unsigned ModulusResult = MakeReg(MVT::f64);
        unsigned TmpF = MakeReg(MVT::f64);
        unsigned TmpI = MakeReg(MVT::i64);
        
        BuildMI(BB, IA64::SUB, 2, TmpI).addReg(IA64::r0).addReg(Tmp2);
        BuildMI(BB, IA64::SETFSIG, 1, TmpF).addReg(TmpI);
        BuildMI(BB, IA64::XMAL, 3, ModulusResult)
          .addReg(TmpF15).addReg(TmpF).addReg(TmpF1);
        BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(ModulusResult);
      } else { // FP modulus! The horror... the horror....
        assert(0 && "sorry, no FP modulus just yet!\n!\n");
      }
    }

    return Result;
  }

  case ISD::SIGN_EXTEND_INREG: {
    Tmp1 = SelectExpr(N.getOperand(0));
    MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
    switch(MVN->getExtraValueType())
    {
    default:
      Node->dump();
      assert(0 && "don't know how to sign extend this type");
      break;
    case MVT::i8: Opc = IA64::SXT1; break;
    case MVT::i16: Opc = IA64::SXT2; break;
    case MVT::i32: Opc = IA64::SXT4; break;
    }
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
  }

  case ISD::SETCC: {
    Tmp1 = SelectExpr(N.getOperand(0));

    if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
      if (MVT::isInteger(SetCC->getOperand(0).getValueType())) {

        if(ConstantSDNode *CSDN =
             dyn_cast<ConstantSDNode>(N.getOperand(1))) {
        // if we are comparing against a constant zero
        if(CSDN->getValue()==0)
          Tmp2 = IA64::r0; // then we can just compare against r0
        else
          Tmp2 = SelectExpr(N.getOperand(1));
        } else // not comparing against a constant
          Tmp2 = SelectExpr(N.getOperand(1));
        
        switch (SetCC->getCondition()) {
        default: assert(0 && "Unknown integer comparison!");
        case ISD::SETEQ:
          BuildMI(BB, IA64::CMPEQ, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETGT:
          BuildMI(BB, IA64::CMPGT, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETGE:
          BuildMI(BB, IA64::CMPGE, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETLT:
          BuildMI(BB, IA64::CMPLT, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETLE:
          BuildMI(BB, IA64::CMPLE, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETNE:
          BuildMI(BB, IA64::CMPNE, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETULT:
          BuildMI(BB, IA64::CMPLTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETUGT:
          BuildMI(BB, IA64::CMPGTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETULE:
          BuildMI(BB, IA64::CMPLEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETUGE:
          BuildMI(BB, IA64::CMPGEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        }
      }
      else { // if not integer, should be FP. FIXME: what about bools? ;)
        assert(SetCC->getOperand(0).getValueType() != MVT::f32 &&
            "error: SETCC should have had incoming f32 promoted to f64!\n");

        if(ConstantFPSDNode *CFPSDN =
             dyn_cast<ConstantFPSDNode>(N.getOperand(1))) {

          // if we are comparing against a constant +0.0 or +1.0
          if(CFPSDN->isExactlyValue(+0.0))
            Tmp2 = IA64::F0; // then we can just compare against f0
          else if(CFPSDN->isExactlyValue(+1.0))
            Tmp2 = IA64::F1; // or f1
          else
            Tmp2 = SelectExpr(N.getOperand(1));
        } else // not comparing against a constant
          Tmp2 = SelectExpr(N.getOperand(1));

        switch (SetCC->getCondition()) {
        default: assert(0 && "Unknown FP comparison!");
        case ISD::SETEQ:
          BuildMI(BB, IA64::FCMPEQ, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETGT:
          BuildMI(BB, IA64::FCMPGT, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETGE:
          BuildMI(BB, IA64::FCMPGE, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETLT:
          BuildMI(BB, IA64::FCMPLT, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETLE:
          BuildMI(BB, IA64::FCMPLE, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETNE:
          BuildMI(BB, IA64::FCMPNE, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETULT:
          BuildMI(BB, IA64::FCMPLTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETUGT:
          BuildMI(BB, IA64::FCMPGTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETULE:
          BuildMI(BB, IA64::FCMPLEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        case ISD::SETUGE:
          BuildMI(BB, IA64::FCMPGEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
          break;
        }
      }
    }
    else
      assert(0 && "this setcc not implemented yet");

    return Result;
  }

  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD: {
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    bool isBool=false;

    if(opcode == ISD::LOAD) { // this is a LOAD
      switch (Node->getValueType(0)) {
        default: assert(0 && "Cannot load this type!");
        case MVT::i1:  Opc = IA64::LD1; isBool=true; break;
              // FIXME: for now, we treat bool loads the same as i8 loads */
        case MVT::i8:  Opc = IA64::LD1; break;
        case MVT::i16: Opc = IA64::LD2; break;
        case MVT::i32: Opc = IA64::LD4; break;
        case MVT::i64: Opc = IA64::LD8; break;
                
        case MVT::f32: Opc = IA64::LDF4; break;
        case MVT::f64: Opc = IA64::LDF8; break;
      }
    } else { // this is an EXTLOAD or ZEXTLOAD
      MVT::ValueType TypeBeingLoaded = cast<MVTSDNode>(Node)->getExtraValueType();
      switch (TypeBeingLoaded) {
        default: assert(0 && "Cannot extload/zextload this type!");
        // FIXME: bools?
        case MVT::i8: Opc = IA64::LD1; break;
        case MVT::i16: Opc = IA64::LD2; break;
        case MVT::i32: Opc = IA64::LD4; break;
        case MVT::f32: Opc = IA64::LDF4; break;
      }
    }

    SDOperand Chain = N.getOperand(0);
    SDOperand Address = N.getOperand(1);

    if(Address.getOpcode() == ISD::GlobalAddress) {
      Select(Chain);
      unsigned dummy = MakeReg(MVT::i64);
      unsigned dummy2 = MakeReg(MVT::i64);
      BuildMI(BB, IA64::ADD, 2, dummy)
        .addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal())
        .addReg(IA64::r1);
      BuildMI(BB, IA64::LD8, 1, dummy2).addReg(dummy);
      if(!isBool)
        BuildMI(BB, Opc, 1, Result).addReg(dummy2);
      else { // emit a little pseudocode to load a bool (stored in one byte)
             // into a predicate register
        assert(Opc==IA64::LD1 && "problem loading a bool");
        unsigned dummy3 = MakeReg(MVT::i64);
        BuildMI(BB, Opc, 1, dummy3).addReg(dummy2);
        // we compare to 0. true? 0. false? 1.
        BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy3).addReg(IA64::r0);
      }
    } else if(ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
      Select(Chain);
      IA64Lowering.restoreGP(BB);
      unsigned dummy = MakeReg(MVT::i64);
      BuildMI(BB, IA64::ADD, 2, dummy).addConstantPoolIndex(CP->getIndex())
        .addReg(IA64::r1); // CPI+GP
      if(!isBool)
        BuildMI(BB, Opc, 1, Result).addReg(dummy);
      else { // emit a little pseudocode to load a bool (stored in one byte)
             // into a predicate register
        assert(Opc==IA64::LD1 && "problem loading a bool");
        unsigned dummy3 = MakeReg(MVT::i64);
        BuildMI(BB, Opc, 1, dummy3).addReg(dummy);
        // we compare to 0. true? 0. false? 1.
        BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy3).addReg(IA64::r0);
      }
    } else if(Address.getOpcode() == ISD::FrameIndex) {
      Select(Chain);  // FIXME ? what about bools?
      unsigned dummy = MakeReg(MVT::i64);
      BuildMI(BB, IA64::MOV, 1, dummy)
        .addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex());
      if(!isBool)
        BuildMI(BB, Opc, 1, Result).addReg(dummy);
      else { // emit a little pseudocode to load a bool (stored in one byte)
             // into a predicate register
        assert(Opc==IA64::LD1 && "problem loading a bool");
        unsigned dummy3 = MakeReg(MVT::i64);
        BuildMI(BB, Opc, 1, dummy3).addReg(dummy);
        // we compare to 0. true? 0. false? 1.
        BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy3).addReg(IA64::r0);
      }
    } else { // none of the above...
      Select(Chain);
      Tmp2 = SelectExpr(Address);
      if(!isBool)
        BuildMI(BB, Opc, 1, Result).addReg(Tmp2);
      else { // emit a little pseudocode to load a bool (stored in one byte)
             // into a predicate register
        assert(Opc==IA64::LD1 && "problem loading a bool");
        unsigned dummy = MakeReg(MVT::i64);
        BuildMI(BB, Opc, 1, dummy).addReg(Tmp2);
        // we compare to 0. true? 0. false? 1.
        BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy).addReg(IA64::r0);
      }        
    }

    return Result;
  }

  case ISD::CopyFromReg: {
    if (Result == 1)
        Result = ExprMap[N.getValue(0)] =
          MakeReg(N.getValue(0).getValueType());

      SDOperand Chain   = N.getOperand(0);

      Select(Chain);
      unsigned r = dyn_cast<RegSDNode>(Node)->getReg();

      if(N.getValueType() == MVT::i1) // if a bool, we use pseudocode
        BuildMI(BB, IA64::PCMPEQUNC, 3, Result)
          .addReg(IA64::r0).addReg(IA64::r0).addReg(r);
                            // (r) Result =cmp.eq.unc(r0,r0)
      else
        BuildMI(BB, IA64::MOV, 1, Result).addReg(r); // otherwise MOV
      return Result;
  }

  case ISD::TAILCALL:
  case ISD::CALL: {
      Select(N.getOperand(0));

      // The chain for this call is now lowered.
      ExprMap.insert(std::make_pair(N.getValue(Node->getNumValues()-1), 1));

      //grab the arguments
      std::vector<unsigned> argvregs;

      for(int i = 2, e = Node->getNumOperands(); i < e; ++i)
        argvregs.push_back(SelectExpr(N.getOperand(i)));

      // see section 8.5.8 of "Itanium Software Conventions and
      // Runtime Architecture Guide to see some examples of what's going
      // on here. (in short: int args get mapped 1:1 'slot-wise' to out0->out7,
      // while FP args get mapped to F8->F15 as needed)

      unsigned used_FPArgs=0; // how many FP Args have been used so far?

      // in reg args
      for(int i = 0, e = std::min(8, (int)argvregs.size()); i < e; ++i)
      {
        unsigned intArgs[] = {IA64::out0, IA64::out1, IA64::out2, IA64::out3,
                              IA64::out4, IA64::out5, IA64::out6, IA64::out7 };
        unsigned FPArgs[] = {IA64::F8, IA64::F9, IA64::F10, IA64::F11,
                             IA64::F12, IA64::F13, IA64::F14, IA64::F15 };

        switch(N.getOperand(i+2).getValueType())
        {
          default:  // XXX do we need to support MVT::i1 here?
            Node->dump();
            N.getOperand(i).Val->dump();
            std::cerr << "Type for " << i << " is: " <<
              N.getOperand(i+2).getValueType() << std::endl;
            assert(0 && "Unknown value type for call");
          case MVT::i64:
            BuildMI(BB, IA64::MOV, 1, intArgs[i]).addReg(argvregs[i]);
            break;
          case MVT::f64:
            BuildMI(BB, IA64::FMOV, 1, FPArgs[used_FPArgs++])
              .addReg(argvregs[i]);
            // FIXME: we don't need to do this _all_ the time:
            BuildMI(BB, IA64::GETFD, 1, intArgs[i]).addReg(argvregs[i]);
            break;
          }
      }

      //in mem args
      for (int i = 8, e = argvregs.size(); i < e; ++i)
      {
        unsigned tempAddr = MakeReg(MVT::i64);
        
        switch(N.getOperand(i+2).getValueType()) {
        default:
          Node->dump();
          N.getOperand(i).Val->dump();
          std::cerr << "Type for " << i << " is: " <<
            N.getOperand(i+2).getValueType() << "\n";
          assert(0 && "Unknown value type for call");
        case MVT::i1: // FIXME?
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
          BuildMI(BB, IA64::ADDIMM22, 2, tempAddr)
            .addReg(IA64::r12).addImm(16 + (i - 8) * 8); // r12 is SP
          BuildMI(BB, IA64::ST8, 2).addReg(tempAddr).addReg(argvregs[i]);
          break;
        case MVT::f32:
        case MVT::f64:
          BuildMI(BB, IA64::ADDIMM22, 2, tempAddr)
            .addReg(IA64::r12).addImm(16 + (i - 8) * 8); // r12 is SP
          BuildMI(BB, IA64::STF8, 2).addReg(tempAddr).addReg(argvregs[i]);
          break;
        }
      }

    // build the right kind of call. if we can branch directly, do so:
    if (GlobalAddressSDNode *GASD =
               dyn_cast<GlobalAddressSDNode>(N.getOperand(1)))
      {
        BuildMI(BB, IA64::BRCALL, 1).addGlobalAddress(GASD->getGlobal(),true);
        IA64Lowering.restoreGP_SP_RP(BB);
      } else
    if (ExternalSymbolSDNode *ESSDN =
             dyn_cast<ExternalSymbolSDNode>(N.getOperand(1)))
      { // FIXME : currently need this case for correctness, to avoid
        // "non-pic code with imm relocation against dynamic symbol" errors
        BuildMI(BB, IA64::BRCALL, 1)
          .addExternalSymbol(ESSDN->getSymbol(), true);
        IA64Lowering.restoreGP_SP_RP(BB);
      }
    else { // otherwise we need to get the function descriptor
           // load the branch target (function)'s entry point and
	   // GP, then branch
      Tmp1 = SelectExpr(N.getOperand(1));

      unsigned targetEntryPoint=MakeReg(MVT::i64);
      unsigned targetGPAddr=MakeReg(MVT::i64);
      unsigned currentGP=MakeReg(MVT::i64);

      // b6 is a scratch branch register, we load the target entry point
      // from the base of the function descriptor
      BuildMI(BB, IA64::LD8, 1, targetEntryPoint).addReg(Tmp1);
      BuildMI(BB, IA64::MOV, 1, IA64::B6).addReg(targetEntryPoint);

      // save the current GP:
      BuildMI(BB, IA64::MOV, 1, currentGP).addReg(IA64::r1);

      /* TODO: we need to make sure doing this never, ever loads a
       * bogus value into r1 (GP). */
      // load the target GP (which is at mem[functiondescriptor+8])
      BuildMI(BB, IA64::ADDIMM22, 2, targetGPAddr)
        .addReg(Tmp1).addImm(8); // FIXME: addimm22? why not postincrement ld
      BuildMI(BB, IA64::LD8, 1, IA64::r1).addReg(targetGPAddr);

      // and then jump: (well, call)
      BuildMI(BB, IA64::BRCALL, 1).addReg(IA64::B6);
      // and finally restore the old GP
      BuildMI(BB, IA64::MOV, 1, IA64::r1).addReg(currentGP);
      IA64Lowering.restoreSP_RP(BB);
    }

    switch (Node->getValueType(0)) {
    default: assert(0 && "Unknown value type for call result!");
    case MVT::Other: return 1;
    case MVT::i1:
      BuildMI(BB, IA64::CMPNE, 2, Result)
        .addReg(IA64::r8).addReg(IA64::r0);
      break;
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      BuildMI(BB, IA64::MOV, 1, Result).addReg(IA64::r8);
      break;
    case MVT::f64:
      BuildMI(BB, IA64::FMOV, 1, Result).addReg(IA64::F8);
      break;
    }
    return Result+N.ResNo;
  }

  } // <- uhhh XXX
  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;
  unsigned opcode = N.getOpcode();

  if (!LoweredTokens.insert(N).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (Node->getOpcode()) {
  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");

  case ISD::EntryToken: return;  // Noop

  case ISD::TokenFactor: {
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));
    return;
  }

  case ISD::CopyToReg: {
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    Tmp2 = cast<RegSDNode>(N)->getReg();

    if (Tmp1 != Tmp2) {
      if(N.getValueType() == MVT::i1) // if a bool, we use pseudocode
        BuildMI(BB, IA64::PCMPEQUNC, 3, Tmp2)
          .addReg(IA64::r0).addReg(IA64::r0).addReg(Tmp1);
                                   // (Tmp1) Tmp2 = cmp.eq.unc(r0,r0)
      else
        BuildMI(BB, IA64::MOV, 1, Tmp2).addReg(Tmp1);
                      // XXX is this the right way 'round? ;)
    }
    return;
  }

  case ISD::RET: {

  /* what the heck is going on here:

<_sabre_> ret with two operands is obvious: chain and value
<camel_> yep
<_sabre_> ret with 3 values happens when 'expansion' occurs
<_sabre_> e.g. i64 gets split into 2x i32
<camel_> oh right
<_sabre_> you don't have this case on ia64
<camel_> yep
<_sabre_> so the two returned values go into EAX/EDX on ia32
<camel_> ahhh *memories*
<_sabre_> :)
<camel_> ok, thanks :)
<_sabre_> so yeah, everything that has a side effect takes a 'token chain'
<_sabre_> this is the first operand always
<_sabre_> these operand often define chains, they are the last operand
<_sabre_> they are printed as 'ch' if you do DAG.dump()
  */

    switch (N.getNumOperands()) {
    default:
      assert(0 && "Unknown return instruction!");
    case 2:
        Select(N.getOperand(0));
        Tmp1 = SelectExpr(N.getOperand(1));
      switch (N.getOperand(1).getValueType()) {
      default: assert(0 && "All other types should have been promoted!!");
               // FIXME: do I need to add support for bools here?
               // (return '0' or '1' r8, basically...)
               //
               // FIXME: need to round floats - 80 bits is bad, the tester
               // told me so
      case MVT::i64:
        // we mark r8 as live on exit up above in LowerArguments()
        BuildMI(BB, IA64::MOV, 1, IA64::r8).addReg(Tmp1);
        break;
      case MVT::f64:
        // we mark F8 as live on exit up above in LowerArguments()
        BuildMI(BB, IA64::FMOV, 1, IA64::F8).addReg(Tmp1);
      }
      break;
    case 1:
      Select(N.getOperand(0));
      break;
    }
    // before returning, restore the ar.pfs register (set by the 'alloc' up top)
    BuildMI(BB, IA64::MOV, 1).addReg(IA64::AR_PFS).addReg(IA64Lowering.VirtGPR);
    BuildMI(BB, IA64::RET, 0); // and then just emit a 'ret' instruction
    return;
  }

  case ISD::BR: {
    Select(N.getOperand(0));
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();
    BuildMI(BB, IA64::BRLCOND_NOTCALL, 1).addReg(IA64::p0).addMBB(Dest);
    // XXX HACK! we do _not_ need long branches all the time
    return;
  }

  case ISD::ImplicitDef: {
    Select(N.getOperand(0));
    BuildMI(BB, IA64::IDEF, 0, cast<RegSDNode>(N)->getReg());
    return;
  }

  case ISD::BRCOND: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();

    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    BuildMI(BB, IA64::BRLCOND_NOTCALL, 1).addReg(Tmp1).addMBB(Dest);
    // XXX HACK! we do _not_ need long branches all the time
    return;
  }

  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::LOAD:
  case ISD::TAILCALL:
  case ISD::CALL:
  case ISD::CopyFromReg:
  case ISD::DYNAMIC_STACKALLOC:
    SelectExpr(N);
    return;

  case ISD::TRUNCSTORE:
  case ISD::STORE: {
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1)); // value

      bool isBool=false;

      if(opcode == ISD::STORE) {
        switch (N.getOperand(1).getValueType()) {
          default: assert(0 && "Cannot store this type!");
          case MVT::i1:  Opc = IA64::ST1; isBool=true; break;
              // FIXME?: for now, we treat bool loads the same as i8 stores */
          case MVT::i8:  Opc = IA64::ST1; break;
          case MVT::i16: Opc = IA64::ST2; break;
          case MVT::i32: Opc = IA64::ST4; break;
          case MVT::i64: Opc = IA64::ST8; break;
                        
          case MVT::f32: Opc = IA64::STF4; break;
          case MVT::f64: Opc = IA64::STF8; break;
        }
      } else { // truncstore
        switch(cast<MVTSDNode>(Node)->getExtraValueType()) {
          default: assert(0 && "unknown type in truncstore");
          case MVT::i1: Opc = IA64::ST1; isBool=true; break;
                        //FIXME: DAG does not promote this load?
          case MVT::i8: Opc = IA64::ST1; break;
          case MVT::i16: Opc = IA64::ST2; break;
          case MVT::i32: Opc = IA64::ST4; break;
          case MVT::f32: Opc = IA64::STF4; break;
        }
      }

      if(N.getOperand(2).getOpcode() == ISD::GlobalAddress) {
        unsigned dummy = MakeReg(MVT::i64);
        unsigned dummy2 = MakeReg(MVT::i64);
        BuildMI(BB, IA64::ADD, 2, dummy)
          .addGlobalAddress(cast<GlobalAddressSDNode>
              (N.getOperand(2))->getGlobal()).addReg(IA64::r1);
        BuildMI(BB, IA64::LD8, 1, dummy2).addReg(dummy);

        if(!isBool)
          BuildMI(BB, Opc, 2).addReg(dummy2).addReg(Tmp1);
        else { // we are storing a bool, so emit a little pseudocode
               // to store a predicate register as one byte
          assert(Opc==IA64::ST1);
          unsigned dummy3 = MakeReg(MVT::i64);
          unsigned dummy4 = MakeReg(MVT::i64);
          BuildMI(BB, IA64::MOV, 1, dummy3).addReg(IA64::r0);
          BuildMI(BB, IA64::TPCADDIMM22, 2, dummy4)
            .addReg(dummy3).addImm(1).addReg(Tmp1); // if(Tmp1) dummy=0+1;
          BuildMI(BB, Opc, 2).addReg(dummy2).addReg(dummy4);
        }
      } else if(N.getOperand(2).getOpcode() == ISD::FrameIndex) {

        // FIXME? (what about bools?)
        
        unsigned dummy = MakeReg(MVT::i64);
        BuildMI(BB, IA64::MOV, 1, dummy)
          .addFrameIndex(cast<FrameIndexSDNode>(N.getOperand(2))->getIndex());
        BuildMI(BB, Opc, 2).addReg(dummy).addReg(Tmp1);
      } else { // otherwise
        Tmp2 = SelectExpr(N.getOperand(2)); //address
        if(!isBool)
          BuildMI(BB, Opc, 2).addReg(Tmp2).addReg(Tmp1);
        else { // we are storing a bool, so emit a little pseudocode
               // to store a predicate register as one byte
          assert(Opc==IA64::ST1);
          unsigned dummy3 = MakeReg(MVT::i64);
          unsigned dummy4 = MakeReg(MVT::i64);
          BuildMI(BB, IA64::MOV, 1, dummy3).addReg(IA64::r0);
          BuildMI(BB, IA64::TPCADDIMM22, 2, dummy4)
            .addReg(dummy3).addImm(1).addReg(Tmp1); // if(Tmp1) dummy=0+1;
          BuildMI(BB, Opc, 2).addReg(Tmp2).addReg(dummy4);
        }
      }
    return;
  }

  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END: {
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();

    Opc = N.getOpcode() == ISD::CALLSEQ_START ? IA64::ADJUSTCALLSTACKDOWN :
                                                IA64::ADJUSTCALLSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  }

    return;
  }
  assert(0 && "GAME OVER. INSERT COIN?");
}


/// createIA64PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createIA64PatternInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}


