//===- LowerPacked.cpp -  Implementation of LowerPacked Transform ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Brad Jones and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering Packed datatypes into more primitive
// Packed datatypes, and finally to scalar operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Argument.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <map>
#include <functional>
using namespace llvm;

namespace {

/// This pass converts packed operators to an
/// equivalent operations on smaller packed data, to possibly
/// scalar operations.  Currently it supports lowering
/// to scalar operations.
///
/// @brief Transforms packed instructions to simpler instructions.
///
class VISIBILITY_HIDDEN LowerPacked 
  : public FunctionPass, public InstVisitor<LowerPacked> {
public:
    static char ID; // Pass identification, replacement for typeid
    LowerPacked() : FunctionPass((intptr_t)&ID) {}

   /// @brief Lowers packed operations to scalar operations.
   /// @param F The fuction to process
   virtual bool runOnFunction(Function &F);

   /// @brief Lowers packed load instructions.
   /// @param LI the load instruction to convert
   void visitLoadInst(LoadInst& LI);

   /// @brief Lowers packed store instructions.
   /// @param SI the store instruction to convert
   void visitStoreInst(StoreInst& SI);

   /// @brief Lowers packed binary operations.
   /// @param BO the binary operator to convert
   void visitBinaryOperator(BinaryOperator& BO);

   /// @brief Lowers packed icmp operations.
   /// @param CI the icmp operator to convert
   void visitICmpInst(ICmpInst& IC);

   /// @brief Lowers packed select instructions.
   /// @param SELI the select operator to convert
   void visitSelectInst(SelectInst& SELI);

   /// @brief Lowers packed extractelement instructions.
   /// @param EI the extractelement operator to convert
   void visitExtractElementInst(ExtractElementInst& EE);

   /// @brief Lowers packed insertelement instructions.
   /// @param EI the insertelement operator to convert
   void visitInsertElementInst(InsertElementInst& IE);

   /// This function asserts if the instruction is a VectorType but
   /// is handled by another function.
   ///
   /// @brief Asserts if VectorType instruction is not handled elsewhere.
   /// @param I the unhandled instruction
   void visitInstruction(Instruction &I) {
     if (isa<VectorType>(I.getType()))
       cerr << "Unhandled Instruction with Packed ReturnType: " << I << '\n';
   }
private:
   /// @brief Retrieves lowered values for a packed value.
   /// @param val the packed value
   /// @return the lowered values
   std::vector<Value*>& getValues(Value* val);

   /// @brief Sets lowered values for a packed value.
   /// @param val the packed value
   /// @param values the corresponding lowered values
   void setValues(Value* val,const std::vector<Value*>& values);

   // Data Members
   /// @brief whether we changed the function or not
   bool Changed;

   /// @brief a map from old packed values to new smaller packed values
   std::map<Value*,std::vector<Value*> > packedToScalarMap;

   /// Instructions in the source program to get rid of
   /// after we do a pass (the old packed instructions)
   std::vector<Instruction*> instrsToRemove;
};

char LowerPacked::ID = 0;
RegisterPass<LowerPacked>
X("lower-packed",
  "lowers packed operations to operations on smaller packed datatypes");

} // end namespace

FunctionPass *llvm::createLowerPackedPass() { return new LowerPacked(); }


// This function sets lowered values for a corresponding
// packed value.  Note, in the case of a forward reference
// getValues(Value*) will have already been called for
// the packed parameter.  This function will then replace
// all references in the in the function of the "dummy"
// value the previous getValues(Value*) call
// returned with actual references.
void LowerPacked::setValues(Value* value,const std::vector<Value*>& values)
{
   std::map<Value*,std::vector<Value*> >::iterator it =
         packedToScalarMap.lower_bound(value);
   if (it == packedToScalarMap.end() || it->first != value) {
       // there was not a forward reference to this element
       packedToScalarMap.insert(it,std::make_pair(value,values));
   }
   else {
      // replace forward declarations with actual definitions
      assert(it->second.size() == values.size() &&
             "Error forward refences and actual definition differ in size");
      for (unsigned i = 0, e = values.size(); i != e; ++i) {
           // replace and get rid of old forward references
           it->second[i]->replaceAllUsesWith(values[i]);
           delete it->second[i];
           it->second[i] = values[i];
      }
   }
}

// This function will examine the packed value parameter
// and if it is a packed constant or a forward reference
// properly create the lowered values needed.  Otherwise
// it will simply retreive values from a
// setValues(Value*,const std::vector<Value*>&)
// call.  Failing both of these cases, it will abort
// the program.
std::vector<Value*>& LowerPacked::getValues(Value* value)
{
   assert(isa<VectorType>(value->getType()) &&
          "Value must be VectorType");

   // reject further processing if this one has
   // already been handled
   std::map<Value*,std::vector<Value*> >::iterator it =
      packedToScalarMap.lower_bound(value);
   if (it != packedToScalarMap.end() && it->first == value) {
       return it->second;
   }

   if (ConstantVector* CP = dyn_cast<ConstantVector>(value)) {
       // non-zero constant case
       std::vector<Value*> results;
       results.reserve(CP->getNumOperands());
       for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i) {
          results.push_back(CP->getOperand(i));
       }
       return packedToScalarMap.insert(it,
                                       std::make_pair(value,results))->second;
   }
   else if (ConstantAggregateZero* CAZ =
            dyn_cast<ConstantAggregateZero>(value)) {
       // zero constant
       const VectorType* PKT = cast<VectorType>(CAZ->getType());
       std::vector<Value*> results;
       results.reserve(PKT->getNumElements());

       Constant* C = Constant::getNullValue(PKT->getElementType());
       for (unsigned i = 0, e = PKT->getNumElements(); i != e; ++i) {
            results.push_back(C);
       }
       return packedToScalarMap.insert(it,
                                       std::make_pair(value,results))->second;
   }
   else if (isa<Instruction>(value)) {
       // foward reference
       const VectorType* PKT = cast<VectorType>(value->getType());
       std::vector<Value*> results;
       results.reserve(PKT->getNumElements());

      for (unsigned i = 0, e = PKT->getNumElements(); i != e; ++i) {
           results.push_back(new Argument(PKT->getElementType()));
      }
      return packedToScalarMap.insert(it,
                                      std::make_pair(value,results))->second;
   }
   else {
       // we don't know what it is, and we are trying to retrieve
       // a value for it
       assert(false && "Unhandled VectorType value");
       abort();
   }
}

void LowerPacked::visitLoadInst(LoadInst& LI)
{
   // Make sure what we are dealing with is a vector type
   if (const VectorType* PKT = dyn_cast<VectorType>(LI.getType())) {
       // Initialization, Idx is needed for getelementptr needed later
       std::vector<Value*> Idx(2);
       Idx[0] = ConstantInt::get(Type::Int32Ty,0);

       ArrayType* AT = ArrayType::get(PKT->getContainedType(0),
                                      PKT->getNumElements());
       PointerType* APT = PointerType::get(AT);

       // Cast the pointer to vector type to an equivalent array
       Value* array = new BitCastInst(LI.getPointerOperand(), APT, 
                                      LI.getName() + ".a", &LI);

       // Convert this load into num elements number of loads
       std::vector<Value*> values;
       values.reserve(PKT->getNumElements());

       for (unsigned i = 0, e = PKT->getNumElements(); i != e; ++i) {
            // Calculate the second index we will need
            Idx[1] = ConstantInt::get(Type::Int32Ty,i);

            // Get the pointer
            Value* val = new GetElementPtrInst(array,
                                               &Idx[0], Idx.size(),
                                               LI.getName() +
                                               ".ge." + utostr(i),
                                               &LI);

            // generate the new load and save the result in packedToScalar map
            values.push_back(new LoadInst(val, LI.getName()+"."+utostr(i),
                             LI.isVolatile(), &LI));
       }

       setValues(&LI,values);
       Changed = true;
       instrsToRemove.push_back(&LI);
   }
}

void LowerPacked::visitBinaryOperator(BinaryOperator& BO)
{
   // Make sure both operands are VectorTypes
   if (isa<VectorType>(BO.getOperand(0)->getType())) {
       std::vector<Value*>& op0Vals = getValues(BO.getOperand(0));
       std::vector<Value*>& op1Vals = getValues(BO.getOperand(1));
       std::vector<Value*> result;
       assert((op0Vals.size() == op1Vals.size()) &&
              "The two packed operand to scalar maps must be equal in size.");

       result.reserve(op0Vals.size());

       // generate the new binary op and save the result
       for (unsigned i = 0; i != op0Vals.size(); ++i) {
            result.push_back(BinaryOperator::create(BO.getOpcode(),
                                                    op0Vals[i],
                                                    op1Vals[i],
                                                    BO.getName() +
                                                    "." + utostr(i),
                                                    &BO));
       }

       setValues(&BO,result);
       Changed = true;
       instrsToRemove.push_back(&BO);
   }
}

void LowerPacked::visitICmpInst(ICmpInst& IC)
{
   // Make sure both operands are VectorTypes
   if (isa<VectorType>(IC.getOperand(0)->getType())) {
       std::vector<Value*>& op0Vals = getValues(IC.getOperand(0));
       std::vector<Value*>& op1Vals = getValues(IC.getOperand(1));
       std::vector<Value*> result;
       assert((op0Vals.size() == op1Vals.size()) &&
              "The two packed operand to scalar maps must be equal in size.");

       result.reserve(op0Vals.size());

       // generate the new binary op and save the result
       for (unsigned i = 0; i != op0Vals.size(); ++i) {
            result.push_back(CmpInst::create(IC.getOpcode(),
                                             IC.getPredicate(),
                                             op0Vals[i],
                                             op1Vals[i],
                                             IC.getName() +
                                             "." + utostr(i),
                                             &IC));
       }

       setValues(&IC,result);
       Changed = true;
       instrsToRemove.push_back(&IC);
   }
}

void LowerPacked::visitStoreInst(StoreInst& SI)
{
   if (const VectorType* PKT =
       dyn_cast<VectorType>(SI.getOperand(0)->getType())) {
       // We will need this for getelementptr
       std::vector<Value*> Idx(2);
       Idx[0] = ConstantInt::get(Type::Int32Ty,0);

       ArrayType* AT = ArrayType::get(PKT->getContainedType(0),
                                      PKT->getNumElements());
       PointerType* APT = PointerType::get(AT);

       // Cast the pointer to packed to an array of equivalent type
       Value* array = new BitCastInst(SI.getPointerOperand(), APT, 
                                      "store.ge.a.", &SI);

       std::vector<Value*>& values = getValues(SI.getOperand(0));

       assert((values.size() == PKT->getNumElements()) &&
              "Scalar must have the same number of elements as Vector Type");

       for (unsigned i = 0, e = PKT->getNumElements(); i != e; ++i) {
            // Generate the indices for getelementptr
            Idx[1] = ConstantInt::get(Type::Int32Ty,i);
            Value* val = new GetElementPtrInst(array,
                                               &Idx[0], Idx.size(),
                                               "store.ge." +
                                               utostr(i) + ".",
                                               &SI);
            new StoreInst(values[i], val, SI.isVolatile(),&SI);
       }

       Changed = true;
       instrsToRemove.push_back(&SI);
   }
}

void LowerPacked::visitSelectInst(SelectInst& SELI)
{
   // Make sure both operands are VectorTypes
   if (isa<VectorType>(SELI.getType())) {
       std::vector<Value*>& op0Vals = getValues(SELI.getTrueValue());
       std::vector<Value*>& op1Vals = getValues(SELI.getFalseValue());
       std::vector<Value*> result;

      assert((op0Vals.size() == op1Vals.size()) &&
             "The two packed operand to scalar maps must be equal in size.");

      for (unsigned i = 0; i != op0Vals.size(); ++i) {
           result.push_back(new SelectInst(SELI.getCondition(),
                                           op0Vals[i],
                                           op1Vals[i],
                                           SELI.getName()+ "." + utostr(i),
                                           &SELI));
      }

      setValues(&SELI,result);
      Changed = true;
      instrsToRemove.push_back(&SELI);
   }
}

void LowerPacked::visitExtractElementInst(ExtractElementInst& EI)
{
  std::vector<Value*>& op0Vals = getValues(EI.getOperand(0));
  const VectorType *PTy = cast<VectorType>(EI.getOperand(0)->getType());
  Value *op1 = EI.getOperand(1);

  if (ConstantInt *C = dyn_cast<ConstantInt>(op1)) {
    EI.replaceAllUsesWith(op0Vals[C->getZExtValue()]);
  } else {
    AllocaInst *alloca = 
      new AllocaInst(PTy->getElementType(),
                     ConstantInt::get(Type::Int32Ty, PTy->getNumElements()),
                     EI.getName() + ".alloca", 
                     EI.getParent()->getParent()->getEntryBlock().begin());
    for (unsigned i = 0; i < PTy->getNumElements(); ++i) {
      GetElementPtrInst *GEP = 
        new GetElementPtrInst(alloca, ConstantInt::get(Type::Int32Ty, i),
                              "store.ge", &EI);
      new StoreInst(op0Vals[i], GEP, &EI);
    }
    GetElementPtrInst *GEP = 
      new GetElementPtrInst(alloca, op1, EI.getName() + ".ge", &EI);
    LoadInst *load = new LoadInst(GEP, EI.getName() + ".load", &EI);
    EI.replaceAllUsesWith(load);
  }

  Changed = true;
  instrsToRemove.push_back(&EI);
}

void LowerPacked::visitInsertElementInst(InsertElementInst& IE)
{
  std::vector<Value*>& Vals = getValues(IE.getOperand(0));
  Value *Elt = IE.getOperand(1);
  Value *Idx = IE.getOperand(2);
  std::vector<Value*> result;
  result.reserve(Vals.size());

  if (ConstantInt *C = dyn_cast<ConstantInt>(Idx)) {
    unsigned idxVal = C->getZExtValue();
    for (unsigned i = 0; i != Vals.size(); ++i) {
      result.push_back(i == idxVal ? Elt : Vals[i]);
    }
  } else {
    for (unsigned i = 0; i != Vals.size(); ++i) {
      ICmpInst *icmp =
        new ICmpInst(ICmpInst::ICMP_EQ, Idx, 
                     ConstantInt::get(Type::Int32Ty, i),
                     "icmp", &IE);
      SelectInst *select =
        new SelectInst(icmp, Elt, Vals[i], "select", &IE);
      result.push_back(select);
    }
  }

  setValues(&IE, result);
  Changed = true;
  instrsToRemove.push_back(&IE);
}

bool LowerPacked::runOnFunction(Function& F)
{
   // initialize
   Changed = false;

   // Does three passes:
   // Pass 1) Converts Packed Operations to
   //         new Packed Operations on smaller
   //         datatypes
   visit(F);

   // Pass 2) Drop all references
   std::for_each(instrsToRemove.begin(),
                 instrsToRemove.end(),
                 std::mem_fun(&Instruction::dropAllReferences));

   // Pass 3) Delete the Instructions to remove aka packed instructions
   for (std::vector<Instruction*>::iterator i = instrsToRemove.begin(),
                                            e = instrsToRemove.end();
        i != e; ++i) {
        (*i)->getParent()->getInstList().erase(*i);
   }

   // clean-up
   packedToScalarMap.clear();
   instrsToRemove.clear();

   return Changed;
}

