//===-- lib/Transforms/Scalar/LowerConstantExprs.cpp ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was written by Vladimir Prus and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerConstantExpression pass, which converts all
// constant expressions into instructions. This is primarily usefull for
// code generators which don't yet want or don't have a need to handle
// constant expressions themself.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Support/InstIterator.h"
#include <vector>
#include <iostream>

using namespace llvm;
using namespace std;

namespace {

    class ConstantExpressionsLower : public FunctionPass {
    private: // FunctionPass overrides

        bool runOnFunction(Function& f);

    private: // internal methods

        /// For all operands of 'insn' which are constant expressions, generates
        /// an appropriate instruction and replaces the use of constant
        /// expression with the use of the generated instruction.
        bool runOnInstruction(Instruction& insn);

        /// Given an constant expression 'c' which occures in 'instruction',
        /// at position 'pos',
        /// generates instruction to compute 'c' and replaces the use of 'c'
        /// with the use of that instruction. This handles only top-level
        /// expression in 'c', any subexpressions are not handled.
        Instruction* convert(const ConstantExpr& c, Instruction* where);
    };

    RegisterOpt<ConstantExpressionsLower> X(
        "lowerconstantexprs", "Lower constant expressions");
}

bool ConstantExpressionsLower::runOnFunction(Function& f)
{
    bool modified = false;
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    {
        modified |= runOnInstruction(*i);
    }
    return modified;
}

bool ConstantExpressionsLower::runOnInstruction(Instruction& instruction)
{
    bool modified = false;
    for (unsigned pos = 0; pos < instruction.getNumOperands(); ++pos)
    {
        if (ConstantExpr* ce
            = dyn_cast<ConstantExpr>(instruction.getOperand(pos))) {

            // Decide where to insert the new instruction
            Instruction* where = &instruction;

            // For PHI nodes we can't insert new instruction before phi,
            // since phi should always come at the beginning of the
            // basic block.
            // So, we need to insert it in the predecessor, right before
            // the terminating instruction.
            if (PHINode* p = dyn_cast<PHINode>(&instruction)) {
                BasicBlock* predecessor = 0;
                for(unsigned i = 0; i < p->getNumIncomingValues(); ++i)
                    if (p->getIncomingValue(i) == ce) {
                        predecessor = p->getIncomingBlock(i);
                        break;
                    }
                assert(predecessor && "could not find predecessor");
                where = predecessor->getTerminator();
            }
            Instruction* n = convert(*ce, where);

            // Note: we can't call replaceAllUsesWith, since
            // that might replace uses in another functions,
            // where the instruction(s) we've generated are not
            // available.

            // Moreover, we can't replace all the users in the same
            // function, because we can't be sure the definition
            // made in this block will be available in other
            // places where the constant is used.
            instruction.setOperand(pos, n);

            // The new instruction might have constant expressions in
            // it. Extract them too.
            runOnInstruction(*n);
            modified = true;
        }
    }
    return modified;
}

Instruction*
ConstantExpressionsLower::convert(const ConstantExpr& c, Instruction* where)
{
    Instruction* result = 0;

    if (c.getOpcode() >= Instruction::BinaryOpsBegin &&
        c.getOpcode() < Instruction::BinaryOpsEnd)
    {
        result = BinaryOperator::create(
            static_cast<Instruction::BinaryOps>(c.getOpcode()),
            c.getOperand(0), c.getOperand(1), "", where);
    }
    else
    {
        switch(c.getOpcode()) {
        case Instruction::GetElementPtr:
        {
            vector<Value*> idx;
            for (unsigned i = 1; i < c.getNumOperands(); ++i)
                idx.push_back(c.getOperand(i));
            result = new GetElementPtrInst(c.getOperand(0),
                                           idx, "", where);
            break;
        }

        case Instruction::Cast:
            result = new CastInst(c.getOperand(0), c.getType(), "",
                                  where);
            break;


        case Instruction::Shl:
        case Instruction::Shr:
            result = new ShiftInst(
                static_cast<Instruction::OtherOps>(c.getOpcode()),
                c.getOperand(0), c.getOperand(1), "", where);
            break;

        case Instruction::Select:
            result = new SelectInst(c.getOperand(0), c.getOperand(1),
                                    c.getOperand(2), "", where);
            break;

        default:
            std::cerr << "Offending expr: " << c << "\n";
            assert(0 && "Constant expression not yet handled!\n");
        }
    }
    return result;
}



namespace llvm {
    FunctionPass* createLowerConstantExpressionsPass()
    {
        return new ConstantExpressionsLower;
    }
}
