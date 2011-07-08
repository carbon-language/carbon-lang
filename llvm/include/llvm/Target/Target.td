//===- Target.td - Target Independent TableGen interface ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

// SubRegIndex - Use instances of SubRegIndex to identify subregisters.
class SubRegIndex {
  string Namespace = "";
}

// RegAltNameIndex - The alternate name set to use for register operands of
// this register class when printing.
class RegAltNameIndex {
  string Namespace = "";
}
def NoRegAltName : RegAltNameIndex;

// Register - You should define one instance of this class for each register
// in the target machine.  String n will become the "name" of the register.
class Register<string n, list<string> altNames = []> {
  string Namespace = "";
  string AsmName = n;
  list<string> AltNames = altNames;

  // Aliases - A list of registers that this register overlaps with.  A read or
  // modification of this register can potentially read or modify the aliased
  // registers.
  list<Register> Aliases = [];

  // SubRegs - A list of registers that are parts of this register. Note these
  // are "immediate" sub-registers and the registers within the list do not
  // themselves overlap. e.g. For X86, EAX's SubRegs list contains only [AX],
  // not [AX, AH, AL].
  list<Register> SubRegs = [];

  // SubRegIndices - For each register in SubRegs, specify the SubRegIndex used
  // to address it. Sub-sub-register indices are automatically inherited from
  // SubRegs.
  list<SubRegIndex> SubRegIndices = [];

  // RegAltNameIndices - The alternate name indices which are valid for this
  // register.
  list<RegAltNameIndex> RegAltNameIndices = [];

  // CompositeIndices - Specify subreg indices that don't correspond directly to
  // a register in SubRegs and are not inherited. The following formats are
  // supported:
  //
  // (a)     Identity  - Reg:a == Reg
  // (a b)   Alias     - Reg:a == Reg:b
  // (a b,c) Composite - Reg:a == (Reg:b):c
  //
  // This can be used to disambiguate a sub-sub-register that exists in more
  // than one subregister and other weird stuff.
  list<dag> CompositeIndices = [];

  // DwarfNumbers - Numbers used internally by gcc/gdb to identify the register.
  // These values can be determined by locating the <target>.h file in the
  // directory llvmgcc/gcc/config/<target>/ and looking for REGISTER_NAMES.  The
  // order of these names correspond to the enumeration used by gcc.  A value of
  // -1 indicates that the gcc number is undefined and -2 that register number
  // is invalid for this mode/flavour.
  list<int> DwarfNumbers = [];

  // CostPerUse - Additional cost of instructions using this register compared
  // to other registers in its class. The register allocator will try to
  // minimize the number of instructions using a register with a CostPerUse.
  // This is used by the x86-64 and ARM Thumb targets where some registers 
  // require larger instruction encodings.
  int CostPerUse = 0;
}

// RegisterWithSubRegs - This can be used to define instances of Register which
// need to specify sub-registers.
// List "subregs" specifies which registers are sub-registers to this one. This
// is used to populate the SubRegs and AliasSet fields of TargetRegisterDesc.
// This allows the code generator to be careful not to put two values with
// overlapping live ranges into registers which alias.
class RegisterWithSubRegs<string n, list<Register> subregs> : Register<n> {
  let SubRegs = subregs;
}

// RegisterClass - Now that all of the registers are defined, and aliases
// between registers are defined, specify which registers belong to which
// register classes.  This also defines the default allocation order of
// registers by register allocators.
//
class RegisterClass<string namespace, list<ValueType> regTypes, int alignment,
                    dag regList, RegAltNameIndex idx = NoRegAltName> {
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

  // CopyCost - This value is used to specify the cost of copying a value
  // between two registers in this register class. The default value is one
  // meaning it takes a single instruction to perform the copying. A negative
  // value means copying is extremely expensive or impossible.
  int CopyCost = 1;

  // MemberList - Specify which registers are in this class.  If the
  // allocation_order_* method are not specified, this also defines the order of
  // allocation used by the register allocator.
  //
  dag MemberList = regList;

  // AltNameIndex - The alternate register name to use when printing operands
  // of this register class. Every register in the register class must have
  // a valid alternate name for the given index.
  RegAltNameIndex altNameIndex = idx;

  // SubRegClasses - Specify the register class of subregisters as a list of
  // dags: (RegClass SubRegIndex, SubRegindex, ...)
  list<dag> SubRegClasses = [];

  // isAllocatable - Specify that the register class can be used for virtual
  // registers and register allocation.  Some register classes are only used to
  // model instruction operand constraints, and should have isAllocatable = 0.
  bit isAllocatable = 1;

  // AltOrders - List of alternative allocation orders. The default order is
  // MemberList itself, and that is good enough for most targets since the
  // register allocators automatically remove reserved registers and move
  // callee-saved registers to the end.
  list<dag> AltOrders = [];

  // AltOrderSelect - The body of a function that selects the allocation order
  // to use in a given machine function. The code will be inserted in a
  // function like this:
  //
  //   static inline unsigned f(const MachineFunction &MF) { ... }
  //
  // The function should return 0 to select the default order defined by
  // MemberList, 1 to select the first AltOrders entry and so on.
  code AltOrderSelect = [{}];
}

// The memberList in a RegisterClass is a dag of set operations. TableGen
// evaluates these set operations and expand them into register lists. These
// are the most common operation, see test/TableGen/SetTheory.td for more
// examples of what is possible:
//
// (add R0, R1, R2) - Set Union. Each argument can be an individual register, a
// register class, or a sub-expression. This is also the way to simply list
// registers.
//
// (sub GPR, SP) - Set difference. Subtract the last arguments from the first.
//
// (and GPR, CSR) - Set intersection. All registers from the first set that are
// also in the second set.
//
// (sequence "R%u", 0, 15) -> [R0, R1, ..., R15]. Generate a sequence of
// numbered registers.
//
// (shl GPR, 4) - Remove the first N elements.
//
// (trunc GPR, 4) - Truncate after the first N elements.
//
// (rotl GPR, 1) - Rotate N places to the left.
//
// (rotr GPR, 1) - Rotate N places to the right.
//
// (decimate GPR, 2) - Pick every N'th element, starting with the first.
//
// All of these operators work on ordered sets, not lists. That means
// duplicates are removed from sub-expressions.

// Set operators. The rest is defined in TargetSelectionDAG.td.
def sequence;
def decimate;

// RegisterTuples - Automatically generate super-registers by forming tuples of
// sub-registers. This is useful for modeling register sequence constraints
// with pseudo-registers that are larger than the architectural registers.
//
// The sub-register lists are zipped together:
//
//   def EvenOdd : RegisterTuples<[sube, subo], [(add R0, R2), (add R1, R3)]>;
//
// Generates the same registers as:
//
//   let SubRegIndices = [sube, subo] in {
//     def R0_R1 : RegisterWithSubRegs<"", [R0, R1]>;
//     def R2_R3 : RegisterWithSubRegs<"", [R2, R3]>;
//   }
//
// The generated pseudo-registers inherit super-classes and fields from their
// first sub-register. Most fields from the Register class are inferred, and
// the AsmName and Dwarf numbers are cleared.
//
// RegisterTuples instances can be used in other set operations to form
// register classes and so on. This is the only way of using the generated
// registers.
class RegisterTuples<list<SubRegIndex> Indices, list<dag> Regs> {
  // SubRegs - N lists of registers to be zipped up. Super-registers are
  // synthesized from the first element of each SubRegs list, the second
  // element and so on.
  list<dag> SubRegs = Regs;

  // SubRegIndices - N SubRegIndex instances. This provides the names of the
  // sub-registers in the synthesized super-registers.
  list<SubRegIndex> SubRegIndices = Indices;

  // Compose sub-register indices like in a normal Register.
  list<dag> CompositeIndices = [];
}


//===----------------------------------------------------------------------===//
// DwarfRegNum - This class provides a mapping of the llvm register enumeration
// to the register numbering used by gcc and gdb.  These values are used by a
// debug information writer to describe where values may be located during
// execution.
class DwarfRegNum<list<int> Numbers> {
  // DwarfNumbers - Numbers used internally by gcc/gdb to identify the register.
  // These values can be determined by locating the <target>.h file in the
  // directory llvmgcc/gcc/config/<target>/ and looking for REGISTER_NAMES.  The
  // order of these names correspond to the enumeration used by gcc.  A value of
  // -1 indicates that the gcc number is undefined and -2 that register number
  // is invalid for this mode/flavour.
  list<int> DwarfNumbers = Numbers;
}

// DwarfRegAlias - This class declares that a given register uses the same dwarf
// numbers as another one. This is useful for making it clear that the two
// registers do have the same number. It also lets us build a mapping
// from dwarf register number to llvm register.
class DwarfRegAlias<Register reg> {
      Register DwarfAlias = reg;
}

//===----------------------------------------------------------------------===//
// Pull in the common support for scheduling
//
include "llvm/Target/TargetSchedule.td"

class Predicate; // Forward def

//===----------------------------------------------------------------------===//
// Instruction set description - These classes correspond to the C++ classes in
// the Target/TargetInstrInfo.h file.
//
class Instruction {
  string Namespace = "";

  dag OutOperandList;       // An dag containing the MI def operand list.
  dag InOperandList;        // An dag containing the MI use operand list.
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
  bit isIndirectBranch = 0; // Is this instruction an indirect branch?
  bit isCompare    = 0;     // Is this instruction a comparison instruction?
  bit isMoveImm    = 0;     // Is this instruction a move immediate instruction?
  bit isBitcast    = 0;     // Is this instruction a bitcast instruction?
  bit isBarrier    = 0;     // Can control flow fall through this instruction?
  bit isCall       = 0;     // Is this instruction a call instruction?
  bit canFoldAsLoad = 0;    // Can this be folded as a simple memory operand?
  bit mayLoad      = 0;     // Is it possible for this inst to read memory?
  bit mayStore     = 0;     // Is it possible for this inst to write memory?
  bit isConvertibleToThreeAddress = 0;  // Can this 2-addr instruction promote?
  bit isCommutable = 0;     // Is this 3 operand instruction commutable?
  bit isTerminator = 0;     // Is this part of the terminator for a basic block?
  bit isReMaterializable = 0; // Is this instruction re-materializable?
  bit isPredicable = 0;     // Is this instruction predicable?
  bit hasDelaySlot = 0;     // Does this instruction have an delay slot?
  bit usesCustomInserter = 0; // Pseudo instr needing special help.
  bit hasCtrlDep   = 0;     // Does this instruction r/w ctrl-flow chains?
  bit isNotDuplicable = 0;  // Is it unsafe to duplicate this instruction?
  bit isAsCheapAsAMove = 0; // As cheap (or cheaper) than a move instruction.
  bit hasExtraSrcRegAllocReq = 0; // Sources have special regalloc requirement?
  bit hasExtraDefRegAllocReq = 0; // Defs have special regalloc requirement?
  bit isPseudo     = 0;     // Is this instruction a pseudo-instruction?
                            // If so, won't have encoding information for
                            // the [MC]CodeEmitter stuff.

  // Side effect flags - When set, the flags have these meanings:
  //
  //  hasSideEffects - The instruction has side effects that are not
  //    captured by any operands of the instruction or other flags.
  //
  //  neverHasSideEffects - Set on an instruction with no pattern if it has no
  //    side effects.
  bit hasSideEffects = 0;
  bit neverHasSideEffects = 0;

  // Is this instruction a "real" instruction (with a distinct machine
  // encoding), or is it a pseudo instruction used for codegen modeling
  // purposes.
  // FIXME: For now this is distinct from isPseudo, above, as code-gen-only
  // instructions can (and often do) still have encoding information
  // associated with them. Once we've migrated all of them over to true
  // pseudo-instructions that are lowered to real instructions prior to
  // the printer/emitter, we can remove this attribute and just use isPseudo.
  bit isCodeGenOnly = 0;

  // Is this instruction a pseudo instruction for use by the assembler parser.
  bit isAsmParserOnly = 0;

  InstrItinClass Itinerary = NoItinerary;// Execution steps used for scheduling.

  string Constraints = "";  // OperandConstraint, e.g. $src = $dst.

  /// DisableEncoding - List of operand names (e.g. "$op1,$op2") that should not
  /// be encoded into the output machineinstr.
  string DisableEncoding = "";

  string PostEncoderMethod = "";
  string DecoderMethod = "";

  /// Target-specific flags. This becomes the TSFlags field in TargetInstrDesc.
  bits<64> TSFlags = 0;

  ///@name Assembler Parser Support
  ///@{

  string AsmMatchConverter = "";

  ///@}
}

/// PseudoInstExpansion - Expansion information for a pseudo-instruction.
/// Which instruction it expands to and how the operands map from the
/// pseudo.
class PseudoInstExpansion<dag Result> {
  dag ResultInst = Result;     // The instruction to generate.
  bit isPseudo = 1;
}

/// Predicates - These are extra conditionals which are turned into instruction
/// selector matching code. Currently each predicate is just a string.
class Predicate<string cond> {
  string CondString = cond;

  /// AssemblerMatcherPredicate - If this feature can be used by the assembler
  /// matcher, this is true.  Targets should set this by inheriting their
  /// feature from the AssemblerPredicate class in addition to Predicate.
  bit AssemblerMatcherPredicate = 0;

  /// AssemblerCondString - Name of the subtarget feature being tested used
  /// as alternative condition string used for assembler matcher.
  /// e.g. "ModeThumb" is translated to "(Bits & ModeThumb) != 0".
  ///      "!ModeThumb" is translated to "(Bits & ModeThumb) == 0".
  /// It can also list multiple features separated by ",".
  /// e.g. "ModeThumb,FeatureThumb2" is translated to
  ///      "(Bits & ModeThumb) != 0 && (Bits & FeatureThumb2) != 0".
  string AssemblerCondString = "";
}

/// NoHonorSignDependentRounding - This predicate is true if support for
/// sign-dependent-rounding is not enabled.
def NoHonorSignDependentRounding
 : Predicate<"!HonorSignDependentRoundingFPMath()">;

class Requires<list<Predicate> preds> {
  list<Predicate> Predicates = preds;
}

/// ops definition - This is just a simple marker used to identify the operand
/// list for an instruction. outs and ins are identical both syntactically and
/// semanticallyr; they are used to define def operands and use operands to
/// improve readibility. This should be used like this:
///     (outs R32:$dst), (ins R32:$src1, R32:$src2) or something similar.
def ops;
def outs;
def ins;

/// variable_ops definition - Mark this instruction as taking a variable number
/// of operands.
def variable_ops;


/// PointerLikeRegClass - Values that are designed to have pointer width are
/// derived from this.  TableGen treats the register class as having a symbolic
/// type that it doesn't know, and resolves the actual regclass to use by using
/// the TargetRegisterInfo::getPointerRegClass() hook at codegen time.
class PointerLikeRegClass<int Kind> {
  int RegClassKind = Kind;
}


/// ptr_rc definition - Mark this operand as being a pointer value whose
/// register class is resolved dynamically via a callback to TargetInstrInfo.
/// FIXME: We should probably change this to a class which contain a list of
/// flags. But currently we have but one flag.
def ptr_rc : PointerLikeRegClass<0>;

/// unknown definition - Mark this operand as being of unknown type, causing
/// it to be resolved by inference in the context it is used.
def unknown;

/// AsmOperandClass - Representation for the kinds of operands which the target
/// specific parser can create and the assembly matcher may need to distinguish.
///
/// Operand classes are used to define the order in which instructions are
/// matched, to ensure that the instruction which gets matched for any
/// particular list of operands is deterministic.
///
/// The target specific parser must be able to classify a parsed operand into a
/// unique class which does not partially overlap with any other classes. It can
/// match a subset of some other class, in which case the super class field
/// should be defined.
class AsmOperandClass {
  /// The name to use for this class, which should be usable as an enum value.
  string Name = ?;

  /// The super classes of this operand.
  list<AsmOperandClass> SuperClasses = [];

  /// The name of the method on the target specific operand to call to test
  /// whether the operand is an instance of this class. If not set, this will
  /// default to "isFoo", where Foo is the AsmOperandClass name. The method
  /// signature should be:
  ///   bool isFoo() const;
  string PredicateMethod = ?;

  /// The name of the method on the target specific operand to call to add the
  /// target specific operand to an MCInst. If not set, this will default to
  /// "addFooOperands", where Foo is the AsmOperandClass name. The method
  /// signature should be:
  ///   void addFooOperands(MCInst &Inst, unsigned N) const;
  string RenderMethod = ?;

  /// The name of the method on the target specific operand to call to custom
  /// handle the operand parsing. This is useful when the operands do not relate
  /// to immediates or registers and are very instruction specific (as flags to
  /// set in a processor register, coprocessor number, ...).
  string ParserMethod = ?;
}

def ImmAsmOperand : AsmOperandClass {
  let Name = "Imm";
}

/// Operand Types - These provide the built-in operand types that may be used
/// by a target.  Targets can optionally provide their own operand types as
/// needed, though this should not be needed for RISC targets.
class Operand<ValueType ty> {
  ValueType Type = ty;
  string PrintMethod = "printOperand";
  string EncoderMethod = "";
  string DecoderMethod = "";
  string AsmOperandLowerMethod = ?;
  dag MIOperandInfo = (ops);

  // ParserMatchClass - The "match class" that operands of this type fit
  // in. Match classes are used to define the order in which instructions are
  // match, to ensure that which instructions gets matched is deterministic.
  //
  // The target specific parser must be able to classify an parsed operand into
  // a unique class, which does not partially overlap with any other classes. It
  // can match a subset of some other class, in which case the AsmOperandClass
  // should declare the other operand as one of its super classes.
  AsmOperandClass ParserMatchClass = ImmAsmOperand;
}

class RegisterOperand<RegisterClass regclass, string pm = "printOperand"> {
  // RegClass - The register class of the operand.
  RegisterClass RegClass = regclass;
  // PrintMethod - The target method to call to print register operands of
  // this type. The method normally will just use an alt-name index to look
  // up the name to print. Default to the generic printOperand().
  string PrintMethod = pm;
  // ParserMatchClass - The "match class" that operands of this type fit
  // in. Match classes are used to define the order in which instructions are
  // match, to ensure that which instructions gets matched is deterministic.
  //
  // The target specific parser must be able to classify an parsed operand into
  // a unique class, which does not partially overlap with any other classes. It
  // can match a subset of some other class, in which case the AsmOperandClass
  // should declare the other operand as one of its super classes.
  AsmOperandClass ParserMatchClass;
}

def i1imm  : Operand<i1>;
def i8imm  : Operand<i8>;
def i16imm : Operand<i16>;
def i32imm : Operand<i32>;
def i64imm : Operand<i64>;

def f32imm : Operand<f32>;
def f64imm : Operand<f64>;

/// zero_reg definition - Special node to stand for the zero register.
///
def zero_reg;

/// PredicateOperand - This can be used to define a predicate operand for an
/// instruction.  OpTypes specifies the MIOperandInfo for the operand, and
/// AlwaysVal specifies the value of this predicate when set to "always
/// execute".
class PredicateOperand<ValueType ty, dag OpTypes, dag AlwaysVal>
  : Operand<ty> {
  let MIOperandInfo = OpTypes;
  dag DefaultOps = AlwaysVal;
}

/// OptionalDefOperand - This is used to define a optional definition operand
/// for an instruction. DefaultOps is the register the operand represents if
/// none is supplied, e.g. zero_reg.
class OptionalDefOperand<ValueType ty, dag OpTypes, dag defaultops>
  : Operand<ty> {
  let MIOperandInfo = OpTypes;
  dag DefaultOps = defaultops;
}


// InstrInfo - This class should only be instantiated once to provide parameters
// which are global to the target machine.
//
class InstrInfo {
  // Target can specify its instructions in either big or little-endian formats.
  // For instance, while both Sparc and PowerPC are big-endian platforms, the
  // Sparc manual specifies its instructions in the format [31..0] (big), while
  // PowerPC specifies them using the format [0..31] (little).
  bit isLittleEndianEncoding = 0;
}

// Standard Pseudo Instructions.
// This list must match TargetOpcodes.h and CodeGenTarget.cpp.
// Only these instructions are allowed in the TargetOpcode namespace.
let isCodeGenOnly = 1, Namespace = "TargetOpcode" in {
def PHI : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins variable_ops);
  let AsmString = "PHINODE";
}
def INLINEASM : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins variable_ops);
  let AsmString = "";
  let neverHasSideEffects = 1;  // Note side effect is encoded in an operand.
}
def PROLOG_LABEL : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$id);
  let AsmString = "";
  let hasCtrlDep = 1;
  let isNotDuplicable = 1;
}
def EH_LABEL : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$id);
  let AsmString = "";
  let hasCtrlDep = 1;
  let isNotDuplicable = 1;
}
def GC_LABEL : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$id);
  let AsmString = "";
  let hasCtrlDep = 1;
  let isNotDuplicable = 1;
}
def KILL : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins variable_ops);
  let AsmString = "";
  let neverHasSideEffects = 1;
}
def EXTRACT_SUBREG : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$supersrc, i32imm:$subidx);
  let AsmString = "";
  let neverHasSideEffects = 1;
}
def INSERT_SUBREG : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$supersrc, unknown:$subsrc, i32imm:$subidx);
  let AsmString = "";
  let neverHasSideEffects = 1;
  let Constraints = "$supersrc = $dst";
}
def IMPLICIT_DEF : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins);
  let AsmString = "";
  let neverHasSideEffects = 1;
  let isReMaterializable = 1;
  let isAsCheapAsAMove = 1;
}
def SUBREG_TO_REG : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$implsrc, unknown:$subsrc, i32imm:$subidx);
  let AsmString = "";
  let neverHasSideEffects = 1;
}
def COPY_TO_REGCLASS : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src, i32imm:$regclass);
  let AsmString = "";
  let neverHasSideEffects = 1;
  let isAsCheapAsAMove = 1;
}
def DBG_VALUE : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins variable_ops);
  let AsmString = "DBG_VALUE";
  let neverHasSideEffects = 1;
}
def REG_SEQUENCE : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins variable_ops);
  let AsmString = "";
  let neverHasSideEffects = 1;
  let isAsCheapAsAMove = 1;
}
def COPY : Instruction {
  let OutOperandList = (outs unknown:$dst);
  let InOperandList = (ins unknown:$src);
  let AsmString = "";
  let neverHasSideEffects = 1;
  let isAsCheapAsAMove = 1;
}
}

//===----------------------------------------------------------------------===//
// AsmParser - This class can be implemented by targets that wish to implement
// .s file parsing.
//
// Subtargets can have multiple different assembly parsers (e.g. AT&T vs Intel
// syntax on X86 for example).
//
class AsmParser {
  // AsmParserClassName - This specifies the suffix to use for the asmparser
  // class.  Generated AsmParser classes are always prefixed with the target
  // name.
  string AsmParserClassName  = "AsmParser";

  // AsmParserInstCleanup - If non-empty, this is the name of a custom member
  // function of the AsmParser class to call on every matched instruction.
  // This can be used to perform target specific instruction post-processing.
  string AsmParserInstCleanup  = "";

  // Variant - AsmParsers can be of multiple different variants.  Variants are
  // used to support targets that need to parser multiple formats for the
  // assembly language.
  int Variant = 0;

  // CommentDelimiter - If given, the delimiter string used to recognize
  // comments which are hard coded in the .td assembler strings for individual
  // instructions.
  string CommentDelimiter = "";

  // RegisterPrefix - If given, the token prefix which indicates a register
  // token. This is used by the matcher to automatically recognize hard coded
  // register tokens as constrained registers, instead of tokens, for the
  // purposes of matching.
  string RegisterPrefix = "";
}
def DefaultAsmParser : AsmParser;

/// AssemblerPredicate - This is a Predicate that can be used when the assembler
/// matches instructions and aliases.
class AssemblerPredicate<string cond> {
  bit AssemblerMatcherPredicate = 1;
  string AssemblerCondString = cond;
}



/// MnemonicAlias - This class allows targets to define assembler mnemonic
/// aliases.  This should be used when all forms of one mnemonic are accepted
/// with a different mnemonic.  For example, X86 allows:
///   sal %al, 1    -> shl %al, 1
///   sal %ax, %cl  -> shl %ax, %cl
///   sal %eax, %cl -> shl %eax, %cl
/// etc.  Though "sal" is accepted with many forms, all of them are directly
/// translated to a shl, so it can be handled with (in the case of X86, it
/// actually has one for each suffix as well):
///   def : MnemonicAlias<"sal", "shl">;
///
/// Mnemonic aliases are mapped before any other translation in the match phase,
/// and do allow Requires predicates, e.g.:
///
///  def : MnemonicAlias<"pushf", "pushfq">, Requires<[In64BitMode]>;
///  def : MnemonicAlias<"pushf", "pushfl">, Requires<[In32BitMode]>;
///
class MnemonicAlias<string From, string To> {
  string FromMnemonic = From;
  string ToMnemonic = To;

  // Predicates - Predicates that must be true for this remapping to happen.
  list<Predicate> Predicates = [];
}

/// InstAlias - This defines an alternate assembly syntax that is allowed to
/// match an instruction that has a different (more canonical) assembly
/// representation.
class InstAlias<string Asm, dag Result, bit Emit = 0b1> {
  string AsmString = Asm;      // The .s format to match the instruction with.
  dag ResultInst = Result;     // The MCInst to generate.
  bit EmitAlias = Emit;        // Emit the alias instead of what's aliased.

  // Predicates - Predicates that must be true for this to match.
  list<Predicate> Predicates = [];
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

  // Variant - AsmWriters can be of multiple different variants.  Variants are
  // used to support targets that need to emit assembly code in ways that are
  // mostly the same for different targets, but have minor differences in
  // syntax.  If the asmstring contains {|} characters in them, this integer
  // will specify which alternative to use.  For example "{x|y|z}" with Variant
  // == 1, will expand to "y".
  int Variant = 0;


  // FirstOperandColumn/OperandSpacing - If the assembler syntax uses a columnar
  // layout, the asmwriter can actually generate output in this columns (in
  // verbose-asm mode).  These two values indicate the width of the first column
  // (the "opcode" area) and the width to reserve for subsequent operands.  When
  // verbose asm mode is enabled, operands will be indented to respect this.
  int FirstOperandColumn = -1;

  // OperandSpacing - Space between operand columns.
  int OperandSpacing = -1;

  // isMCAsmWriter - Is this assembly writer for an MC emitter? This controls
  // generation of the printInstruction() method. For MC printers, it takes
  // an MCInstr* operand, otherwise it takes a MachineInstr*.
  bit isMCAsmWriter = 0;
}
def DefaultAsmWriter : AsmWriter;


//===----------------------------------------------------------------------===//
// Target - This class contains the "global" target information
//
class Target {
  // InstructionSet - Instruction set description for this target.
  InstrInfo InstructionSet;

  // AssemblyParsers - The AsmParser instances available for this target.
  list<AsmParser> AssemblyParsers = [DefaultAsmParser];

  // AssemblyWriters - The AsmWriter instances available for this target.
  list<AsmWriter> AssemblyWriters = [DefaultAsmWriter];
}

//===----------------------------------------------------------------------===//
// SubtargetFeature - A characteristic of the chip set.
//
class SubtargetFeature<string n, string a,  string v, string d,
                       list<SubtargetFeature> i = []> {
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

  // Implies - Features that this feature implies are present. If one of those
  // features isn't set, then this one shouldn't be set either.
  //
  list<SubtargetFeature> Implies = i;
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
// Pull in the common support for calling conventions.
//
include "llvm/Target/TargetCallingConv.td"

//===----------------------------------------------------------------------===//
// Pull in the common support for DAG isel generation.
//
include "llvm/Target/TargetSelectionDAG.td"
