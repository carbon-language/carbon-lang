//===- Target.td - Target Independent TableGen interface ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces which should be
// implemented by each target which is using a TableGen based code generator.
//
//===----------------------------------------------------------------------===//

// Include all information about LLVM intrinsics.
include "llvm/Intrinsics.td"

//===----------------------------------------------------------------------===//
// Register file description - These classes are used to fill in the target
// description classes.

class RegisterClass; // Forward def

// Register - You should define one instance of this class for each register
// in the target machine.  String n will become the "name" of the register.
class Register<string n> {
  string Namespace = "";
  string Name = n;

  // SpillSize - If this value is set to a non-zero value, it is the size in
  // bits of the spill slot required to hold this register.  If this value is
  // set to zero, the information is inferred from any register classes the
  // register belongs to.
  int SpillSize = 0;

  // SpillAlignment - This value is used to specify the alignment required for
  // spilling the register.  Like SpillSize, this should only be explicitly
  // specified if the register is not in a register class.
  int SpillAlignment = 0;

  // Aliases - A list of registers that this register overlaps with.  A read or
  // modification of this register can potentially read or modifie the aliased
  // registers.
  //
  list<Register> Aliases = [];
  
  // DwarfNumber - Number used internally by gcc/gdb to identify the register.
  // These values can be determined by locating the <target>.h file in the
  // directory llvmgcc/gcc/config/<target>/ and looking for REGISTER_NAMES.  The
  // order of these names correspond to the enumeration used by gcc.  A value of
  // -1 indicates that the gcc number is undefined.
  int DwarfNumber = -1;
}

// RegisterGroup - This can be used to define instances of Register which
// need to specify aliases.
// List "aliases" specifies which registers are aliased to this one.  This
// allows the code generator to be careful not to put two values with 
// overlapping live ranges into registers which alias.
class RegisterGroup<string n, list<Register> aliases> : Register<n> {
  let Aliases = aliases;
}

// RegisterClass - Now that all of the registers are defined, and aliases
// between registers are defined, specify which registers belong to which
// register classes.  This also defines the default allocation order of
// registers by register allocators.
//
class RegisterClass<string namespace, list<ValueType> regTypes, int alignment,
                    list<Register> regList> {
  string Namespace = namespace;

  // RegType - Specify the list ValueType of the registers in this register
  // class.  Note that all registers in a register class must have the same
  // ValueTypes.  This is a list because some targets permit storing different 
  // types in same register, for example vector values with 128-bit total size,
  // but different count/size of items, like SSE on x86.
  //
  list<ValueType> RegTypes = regTypes;

  // Size - Specify the spill size in bits of the registers.  A default value of
  // zero lets tablgen pick an appropriate size.
  int Size = 0;

  // Alignment - Specify the alignment required of the registers when they are
  // stored or loaded to memory.
  //
  int Alignment = alignment;

  // MemberList - Specify which registers are in this class.  If the
  // allocation_order_* method are not specified, this also defines the order of
  // allocation used by the register allocator.
  //
  list<Register> MemberList = regList;

  // MethodProtos/MethodBodies - These members can be used to insert arbitrary
  // code into a generated register class.   The normal usage of this is to 
  // overload virtual methods.
  code MethodProtos = [{}];
  code MethodBodies = [{}];
}


//===----------------------------------------------------------------------===//
// DwarfRegNum - This class provides a mapping of the llvm register enumeration
// to the register numbering used by gcc and gdb.  These values are used by a
// debug information writer (ex. DwarfWriter) to describe where values may be
// located during execution.
class DwarfRegNum<int N> {
  // DwarfNumber - Number used internally by gcc/gdb to identify the register.
  // These values can be determined by locating the <target>.h file in the
  // directory llvmgcc/gcc/config/<target>/ and looking for REGISTER_NAMES.  The
  // order of these names correspond to the enumeration used by gcc.  A value of
  // -1 indicates that the gcc number is undefined.
  int DwarfNumber = N;
}

//===----------------------------------------------------------------------===//
// Pull in the common support for scheduling
//
include "TargetSchedule.td"

class Predicate; // Forward def

//===----------------------------------------------------------------------===//
// Instruction set description - These classes correspond to the C++ classes in
// the Target/TargetInstrInfo.h file.
//
class Instruction {
  string Name = "";         // The opcode string for this instruction
  string Namespace = "";

  dag OperandList;          // An dag containing the MI operand list.
  string AsmString = "";    // The .s format to print the instruction with.

  // Pattern - Set to the DAG pattern for this instruction, if we know of one,
  // otherwise, uninitialized.
  list<dag> Pattern;

  // The follow state will eventually be inferred automatically from the
  // instruction pattern.

  list<Register> Uses = []; // Default to using no non-operand registers
  list<Register> Defs = []; // Default to modifying no non-operand registers

  // Predicates - List of predicates which will be turned into isel matching
  // code.
  list<Predicate> Predicates = [];

  // Code size.
  int CodeSize = 0;

  // Added complexity passed onto matching pattern.
  int AddedComplexity  = 0;

  // These bits capture information about the high-level semantics of the
  // instruction.
  bit isReturn     = 0;     // Is this instruction a return instruction?
  bit isBranch     = 0;     // Is this instruction a branch instruction?
  bit isBarrier    = 0;     // Can control flow fall through this instruction?
  bit isCall       = 0;     // Is this instruction a call instruction?
  bit isLoad       = 0;     // Is this instruction a load instruction?
  bit isStore      = 0;     // Is this instruction a store instruction?
  bit isTwoAddress = 0;     // Is this a two address instruction?
  bit isConvertibleToThreeAddress = 0;  // Can this 2-addr instruction promote?
  bit isCommutable = 0;     // Is this 3 operand instruction commutable?
  bit isTerminator = 0;     // Is this part of the terminator for a basic block?
  bit hasDelaySlot = 0;     // Does this instruction have an delay slot?
  bit usesCustomDAGSchedInserter = 0; // Pseudo instr needing special help.
  bit hasCtrlDep   = 0;     // Does this instruction r/w ctrl-flow chains?
  bit noResults    = 0;     // Does this instruction produce no results?
  
  InstrItinClass Itinerary = NoItinerary;// Execution steps used for scheduling.
}

/// Predicates - These are extra conditionals which are turned into instruction
/// selector matching code. Currently each predicate is just a string.
class Predicate<string cond> {
  string CondString = cond;
}

class Requires<list<Predicate> preds> {
  list<Predicate> Predicates = preds;
}

/// ops definition - This is just a simple marker used to identify the operands
/// list for an instruction.  This should be used like this:
///     (ops R32:$dst, R32:$src) or something similar.
def ops;

/// variable_ops definition - Mark this instruction as taking a variable number
/// of operands.
def variable_ops;

/// ptr_rc definition - Mark this operand as being a pointer value whose
/// register class is resolved dynamically via a callback to TargetInstrInfo.
/// FIXME: We should probably change this to a class which contain a list of
/// flags. But currently we have but one flag.
def ptr_rc;

/// Operand Types - These provide the built-in operand types that may be used
/// by a target.  Targets can optionally provide their own operand types as
/// needed, though this should not be needed for RISC targets.
class Operand<ValueType ty> {
  ValueType Type = ty;
  string PrintMethod = "printOperand";
  int NumMIOperands = 1;
  dag MIOperandInfo = (ops);
}

def i1imm  : Operand<i1>;
def i8imm  : Operand<i8>;
def i16imm : Operand<i16>;
def i32imm : Operand<i32>;
def i64imm : Operand<i64>;

// InstrInfo - This class should only be instantiated once to provide parameters
// which are global to the the target machine.
//
class InstrInfo {
  // If the target wants to associate some target-specific information with each
  // instruction, it should provide these two lists to indicate how to assemble
  // the target specific information into the 32 bits available.
  //
  list<string> TSFlagsFields = [];
  list<int>    TSFlagsShifts = [];

  // Target can specify its instructions in either big or little-endian formats.
  // For instance, while both Sparc and PowerPC are big-endian platforms, the
  // Sparc manual specifies its instructions in the format [31..0] (big), while
  // PowerPC specifies them using the format [0..31] (little).
  bit isLittleEndianEncoding = 0;
}

// Standard Instructions.
def PHI : Instruction {
  let OperandList = (ops variable_ops);
  let AsmString = "PHINODE";
  let Namespace = "TargetInstrInfo";
}
def INLINEASM : Instruction {
  let OperandList = (ops variable_ops);
  let AsmString = "";
  let Namespace = "TargetInstrInfo";
}

//===----------------------------------------------------------------------===//
// AsmWriter - This class can be implemented by targets that need to customize
// the format of the .s file writer.
//
// Subtargets can have multiple different asmwriters (e.g. AT&T vs Intel syntax
// on X86 for example).
//
class AsmWriter {
  // AsmWriterClassName - This specifies the suffix to use for the asmwriter
  // class.  Generated AsmWriter classes are always prefixed with the target
  // name.
  string AsmWriterClassName  = "AsmPrinter";

  // InstFormatName - AsmWriters can specify the name of the format string to
  // print instructions with.
  string InstFormatName = "AsmString";

  // Variant - AsmWriters can be of multiple different variants.  Variants are
  // used to support targets that need to emit assembly code in ways that are
  // mostly the same for different targets, but have minor differences in
  // syntax.  If the asmstring contains {|} characters in them, this integer
  // will specify which alternative to use.  For example "{x|y|z}" with Variant
  // == 1, will expand to "y".
  int Variant = 0;
}
def DefaultAsmWriter : AsmWriter;


//===----------------------------------------------------------------------===//
// Target - This class contains the "global" target information
//
class Target {
  // InstructionSet - Instruction set description for this target.
  InstrInfo InstructionSet;

  // AssemblyWriters - The AsmWriter instances available for this target.
  list<AsmWriter> AssemblyWriters = [DefaultAsmWriter];
}

//===----------------------------------------------------------------------===//
// SubtargetFeature - A characteristic of the chip set.
//
class SubtargetFeature<string n, string a,  string v, string d> {
  // Name - Feature name.  Used by command line (-mattr=) to determine the
  // appropriate target chip.
  //
  string Name = n;
  
  // Attribute - Attribute to be set by feature.
  //
  string Attribute = a;
  
  // Value - Value the attribute to be set to by feature.
  //
  string Value = v;
  
  // Desc - Feature description.  Used by command line (-mattr=) to display help
  // information.
  //
  string Desc = d;
}

//===----------------------------------------------------------------------===//
// Processor chip sets - These values represent each of the chip sets supported
// by the scheduler.  Each Processor definition requires corresponding
// instruction itineraries.
//
class Processor<string n, ProcessorItineraries pi, list<SubtargetFeature> f> {
  // Name - Chip set name.  Used by command line (-mcpu=) to determine the
  // appropriate target chip.
  //
  string Name = n;
  
  // ProcItin - The scheduling information for the target processor.
  //
  ProcessorItineraries ProcItin = pi;
  
  // Features - list of 
  list<SubtargetFeature> Features = f;
}

//===----------------------------------------------------------------------===//
// Pull in the common support for DAG isel generation
//
include "TargetSelectionDAG.td"
