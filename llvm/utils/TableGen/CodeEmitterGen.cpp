//===- CodeEmitterGen.cpp - Code Emitter Generator ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// CodeEmitterGen uses the descriptions of instructions and their fields to
// construct an automated code emitter: a function that, given a MachineInstr,
// returns the (currently, 32-bit unsigned) value of the instruction.
//
//===----------------------------------------------------------------------===//

#include "CodeEmitterGen.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

void CodeEmitterGen::emitInstrOpBits(std::ostream &o,
                                     const std::vector<RecordVal> &Vals,
                                     std::map<std::string, unsigned> &OpOrder,
                                     std::map<std::string, bool> &OpContinuous)
{
  for (unsigned f = 0, e = Vals.size(); f != e; ++f) {
    if (Vals[f].getPrefix()) {
      BitsInit *FieldInitializer = (BitsInit*)Vals[f].getValue();

      // Scan through the field looking for bit initializers of the current
      // variable...
      for (int i = FieldInitializer->getNumBits()-1; i >= 0; --i) {
        Init *I = FieldInitializer->getBit(i);
        if (BitInit *BI = dynamic_cast<BitInit*>(I)) {
          DEBUG(o << "      // bit init: f: " << f << ", i: " << i << "\n");
        } else if (UnsetInit *UI = dynamic_cast<UnsetInit*>(I)) {
          DEBUG(o << "      // unset init: f: " << f << ", i: " << i << "\n");
        } else if (VarBitInit *VBI = dynamic_cast<VarBitInit*>(I)) {
          TypedInit *TI = VBI->getVariable();
          if (VarInit *VI = dynamic_cast<VarInit*>(TI)) {
            // If the bits of the field are laid out consecutively in the
            // instruction, then instead of separately ORing in bits, just
            // mask and shift the entire field for efficiency.
            if (OpContinuous[VI->getName()]) {
              // already taken care of in the loop above, thus there is no
              // need to individually OR in the bits

              // for debugging, output the regular version anyway, commented
              DEBUG(o << "      // Value |= getValueBit(op"
                      << OpOrder[VI->getName()] << ", " << VBI->getBitNum()
                      << ")" << " << " << i << ";\n");
            } else {
              o << "      Value |= getValueBit(op" << OpOrder[VI->getName()]
                << ", " << VBI->getBitNum()
                << ")" << " << " << i << ";\n";
            }
          } else if (FieldInit *FI = dynamic_cast<FieldInit*>(TI)) {
            // FIXME: implement this!
            std::cerr << "Error: FieldInit not implemented!\n";
            abort();
          } else {
            std::cerr << "Error: unimplemented case in "
                      << "CodeEmitterGen::emitInstrOpBits()\n";
            abort();
          }
        }
      }
    }
  }
}


void CodeEmitterGen::run(std::ostream &o) {
  CodeGenTarget Target;
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  EmitSourceFileHeader("Machine Code Emitter", o);
  std::string Namespace = Insts[0]->getValueAsString("Namespace") + "::";

  // Emit function declaration
  o << "unsigned " << Target.getName() << "CodeEmitter::"
    << "getBinaryCodeForInstr(MachineInstr &MI) {\n"
    << "  unsigned Value = 0;\n"
    << "  switch (MI.getOpcode()) {\n";

  // Emit a case statement for each opcode
  for (std::vector<Record*>::iterator I = Insts.begin(), E = Insts.end();
       I != E; ++I) {
    Record *R = *I;
    if (R->getName() == "PHI" || R->getName() == "INLINEASM") continue;
    
    o << "    case " << Namespace << R->getName() << ": {\n";

    BitsInit *BI = R->getValueAsBitsInit("Inst");

    // For little-endian instruction bit encodings, reverse the bit order
    if (Target.isLittleEndianEncoding()) {
      unsigned numBits = BI->getNumBits();
      BitsInit *NewBI = new BitsInit(numBits);
      for (unsigned bit = 0, end = numBits / 2; bit != end; ++bit) {
        unsigned bitSwapIdx = numBits - bit - 1;
        Init *OrigBit = BI->getBit(bit);
        Init *BitSwap = BI->getBit(bitSwapIdx);
        NewBI->setBit(bit, BitSwap);
        NewBI->setBit(bitSwapIdx, OrigBit);
      }
      if (numBits % 2) {
        unsigned middle = (numBits + 1) / 2;
        NewBI->setBit(middle, BI->getBit(middle));
      }
      BI = NewBI;
      
      // Update the bits in reversed order so that emitInstrOpBits will get the
      // correct endianness.
      R->getValue("Inst")->setValue(NewBI);
    }

    unsigned Value = 0;
    const std::vector<RecordVal> &Vals = R->getValues();

    DEBUG(o << "      // prefilling: ");
    // Start by filling in fixed values...
    for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i) {
      if (BitInit *B = dynamic_cast<BitInit*>(BI->getBit(e-i-1))) {
        Value |= B->getValue() << (e-i-1);
        DEBUG(o << B->getValue());
      } else {
        DEBUG(o << "0");
      }
    }
    DEBUG(o << "\n");

    DEBUG(o << "      // " << *R->getValue("Inst") << "\n");
    o << "      Value = " << Value << "U;\n\n";

    // Loop over all of the fields in the instruction, determining which are the
    // operands to the instruction.
    unsigned op = 0;
    std::map<std::string, unsigned> OpOrder;
    std::map<std::string, bool> OpContinuous;
    for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
      if (!Vals[i].getPrefix() && !Vals[i].getValue()->isComplete()) {
        // Is the operand continuous? If so, we can just mask and OR it in
        // instead of doing it bit-by-bit, saving a lot in runtime cost.
        BitsInit *InstInit = BI;
        int beginBitInVar = -1, endBitInVar = -1;
        int beginBitInInst = -1, endBitInInst = -1;
        bool continuous = true;

        for (int bit = InstInit->getNumBits()-1; bit >= 0; --bit) {
          if (VarBitInit *VBI =
              dynamic_cast<VarBitInit*>(InstInit->getBit(bit))) {
            TypedInit *TI = VBI->getVariable();
            if (VarInit *VI = dynamic_cast<VarInit*>(TI)) {
              // only process the current variable
              if (VI->getName() != Vals[i].getName())
                continue;

              if (beginBitInVar == -1)
                beginBitInVar = VBI->getBitNum();

              if (endBitInVar == -1)
                endBitInVar = VBI->getBitNum();
              else {
                if (endBitInVar == (int)VBI->getBitNum() + 1)
                  endBitInVar = VBI->getBitNum();
                else {
                  continuous = false;
                  break;
                }
              }

              if (beginBitInInst == -1)
                beginBitInInst = bit;
              if (endBitInInst == -1)
                endBitInInst = bit;
              else {
                if (endBitInInst == bit + 1)
                  endBitInInst = bit;
                else {
                  continuous = false;
                  break;
                }
              }

              // maintain same distance between bits in field and bits in
              // instruction. if the relative distances stay the same
              // throughout,
              if (beginBitInVar - (int)VBI->getBitNum() !=
                  beginBitInInst - bit) {
                continuous = false;
                break;
              }
            }
          }
        }

        // If we have found no bit in "Inst" which comes from this field, then
        // this is not an operand!!
        if (beginBitInInst != -1) {
          o << "      // op" << op << ": " << Vals[i].getName() << "\n"
            << "      int op" << op
            <<" = getMachineOpValue(MI, MI.getOperand("<<op<<"));\n";
          //<< "   MachineOperand &op" << op <<" = MI.getOperand("<<op<<");\n";
          OpOrder[Vals[i].getName()] = op++;

          DEBUG(o << "      // Var: begin = " << beginBitInVar
                  << ", end = " << endBitInVar
                  << "; Inst: begin = " << beginBitInInst
                  << ", end = " << endBitInInst << "\n");

          if (continuous) {
            DEBUG(o << "      // continuous: op" << OpOrder[Vals[i].getName()]
                    << "\n");

            // Mask off the right bits
            // Low mask (ie. shift, if necessary)
            assert(endBitInVar >= 0 && "Negative shift amount in masking!");
            if (endBitInVar != 0) {
              o << "      op" << OpOrder[Vals[i].getName()]
                << " >>= " << endBitInVar << ";\n";
              beginBitInVar -= endBitInVar;
              endBitInVar = 0;
            }

            // High mask
            o << "      op" << OpOrder[Vals[i].getName()]
              << " &= (1<<" << beginBitInVar+1 << ") - 1;\n";

            // Shift the value to the correct place (according to place in inst)
            assert(endBitInInst >= 0 && "Negative shift amount!");
            if (endBitInInst != 0)
              o << "      op" << OpOrder[Vals[i].getName()]
              << " <<= " << endBitInInst << ";\n";

            // Just OR in the result
            o << "      Value |= op" << OpOrder[Vals[i].getName()] << ";\n";
          }

          // otherwise, will be taken care of in the loop below using this
          // value:
          OpContinuous[Vals[i].getName()] = continuous;
        }
      }
    }

    emitInstrOpBits(o, Vals, OpOrder, OpContinuous);

    o << "      break;\n"
      << "    }\n";
  }

  // Default case: unhandled opcode
  o << "  default:\n"
    << "    std::cerr << \"Not supported instr: \" << MI << \"\\n\";\n"
    << "    abort();\n"
    << "  }\n"
    << "  return Value;\n"
    << "}\n\n";
}
