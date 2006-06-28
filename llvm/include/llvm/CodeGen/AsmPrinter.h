//===-- llvm/CodeGen/AsmPrinter.h - AsmPrinter Framework --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class is intended to be used as a base class for target-specific
// asmwriters.  This class primarily takes care of printing global constants,
// which are printed in a very similar way across all targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMPRINTER_H
#define LLVM_CODEGEN_ASMPRINTER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class Constant;
  class ConstantArray;
  class Mangler;
  class GlobalVariable;

  class AsmPrinter : public MachineFunctionPass {
    /// FunctionNumber - This provides a unique ID for each function emitted in
    /// this translation unit.  It is autoincremented by SetupMachineFunction,
    /// and can be accessed with getFunctionNumber() and 
    /// IncrementFunctionNumber().
    ///
    unsigned FunctionNumber;

  public:
    /// Output stream on which we're printing assembly code.
    ///
    std::ostream &O;

    /// Target machine description.
    ///
    TargetMachine &TM;

    /// Name-mangler for global names.
    ///
    Mangler *Mang;

    /// Cache of mangled name for current function. This is recalculated at the
    /// beginning of each call to runOnMachineFunction().
    ///
    std::string CurrentFnName;

    //===------------------------------------------------------------------===//
    // Properties to be set by the derived class ctor, used to configure the
    // asmwriter.

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;     // Defaults to "#"

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;    // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .o file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;   // Defaults to "."
    
    /// GlobalVarAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable (that isn't a function)
    ///
    const char *GlobalVarAddrPrefix;       // Defaults to ""
    const char *GlobalVarAddrSuffix;       // Defaults to ""

    /// FunctionAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable that points to a function.
    /// For example, this is used by the IA64 backend to materialize
    /// function descriptors, by decorating the ".data8" object with the
    /// \literal @fptr( ) \endliteral
    /// link-relocation operator.
    ///
    const char *FunctionAddrPrefix;       // Defaults to ""
    const char *FunctionAddrSuffix;       // Defaults to ""

    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assmebly statement.
    const char *InlineAsmStart;           // Defaults to "#APP\n"
    const char *InlineAsmEnd;             // Defaults to "#NO_APP\n"
    
    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;   // Defaults to "\t.zero\t"
    const char *ZeroDirectiveSuffix;  // Defaults to ""

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;  // Defaults to "\t.ascii\t"
    
    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;  // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;   // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;  // Defaults to "\t.short\t"
    const char *Data32bitsDirective;  // Defaults to "\t.long\t"
    const char *Data64bitsDirective;  // Defaults to "\t.quad\t"

    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;       // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;          // Defaults to true
    
    //===--- Section Switching Directives ---------------------------------===//
    
    /// CurrentSection - The current section we are emitting to.  This is
    /// controlled and used by the SwitchSection method.
    std::string CurrentSection;
    
    /// SwitchToSectionDirective - This is the directive used when we want to
    /// emit a global to an arbitrary section.  The section name is emited after
    /// this.
    const char *SwitchToSectionDirective;  // Defaults to "\t.section\t"
    
    /// TextSectionStartSuffix - This is printed after each start of section
    /// directive for text sections.
    const char *TextSectionStartSuffix;        // Defaults to "".

    /// DataSectionStartSuffix - This is printed after each start of section
    /// directive for data sections.
    const char *DataSectionStartSuffix;        // Defaults to "".
    
    /// SectionEndDirectiveSuffix - If non-null, the asm printer will close each
    /// section with the section name and this suffix printed.
    const char *SectionEndDirectiveSuffix; // Defaults to null.
    
    /// ConstantPoolSection - This is the section that we SwitchToSection right
    /// before emitting the constant pool for a function.
    const char *ConstantPoolSection;     // Defaults to "\t.section .rodata\n"

    /// JumpTableSection - This is the section that we SwitchToSection right
    /// before emitting the jump tables for a function.
    const char *JumpTableSection;     // Defaults to "\t.section .rodata\n"
    
    /// StaticCtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static constructor list.
    /// Defaults to "\t.section .ctors,\"aw\",@progbits".
    const char *StaticCtorsSection;

    /// StaticDtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static destructor list.
    /// Defaults to "\t.section .dtors,\"aw\",@progbits".
    const char *StaticDtorsSection;
    
    //===--- Global Variable Emission Directives --------------------------===//
    
    /// LCOMMDirective - This is the name of a directive (if supported) that can
    /// be used to efficiently declare a local (internal) block of zero
    /// initialized data in the .bss/.data section.  The syntax expected is:
    /// \literal <LCOMMDirective> SYMBOLNAME LENGTHINBYTES, ALIGNMENT
    /// \endliteral
    const char *LCOMMDirective;          // Defaults to null.
    
    const char *COMMDirective;           // Defaults to "\t.comm\t".
    
    /// COMMDirectiveTakesAlignment - True if COMMDirective take a third
    /// argument that specifies the alignment of the declaration.
    bool COMMDirectiveTakesAlignment;    // Defaults to true.
    
    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;     // Defaults to true.
  
  protected:
    AsmPrinter(std::ostream &o, TargetMachine &TM);
    
  public:
    /// SwitchToTextSection - Switch to the specified section of the executable
    /// if we are not already in it!  If GV is non-null and if the global has an
    /// explicitly requested section, we switch to the section indicated for the
    /// global instead of NewSection.
    ///
    /// If the new section is an empty string, this method forgets what the
    /// current section is, but does not emit a .section directive.
    ///
    /// This method is used when about to emit executable code.
    ///
    void SwitchToTextSection(const char *NewSection, const GlobalValue *GV);

    /// SwitchToDataSection - Switch to the specified section of the executable
    /// if we are not already in it!  If GV is non-null and if the global has an
    /// explicitly requested section, we switch to the section indicated for the
    /// global instead of NewSection.
    ///
    /// If the new section is an empty string, this method forgets what the
    /// current section is, but does not emit a .section directive.
    ///
    /// This method is used when about to emit data.  For most assemblers, this
    /// is the same as the SwitchToTextSection method, but not all assemblers
    /// are the same.
    ///
    void SwitchToDataSection(const char *NewSection, const GlobalValue *GV);
    
    /// getPreferredAlignmentLog - Return the preferred alignment of the
    /// specified global, returned in log form.  This includes an explicitly
    /// requested alignment (if the global has one).
    unsigned getPreferredAlignmentLog(const GlobalVariable *GV) const;
  protected:
    /// doInitialization - Set up the AsmPrinter when we are working on a new
    /// module.  If your pass overrides this, it must make sure to explicitly
    /// call this implementation.
    bool doInitialization(Module &M);

    /// doFinalization - Shut down the asmprinter.  If you override this in your
    /// pass, you must make sure to call it explicitly.
    bool doFinalization(Module &M);

    /// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
    /// instruction, using the specified assembler variant.  Targets should
    /// override this to format as appropriate.  This method can return true if
    /// the operand is erroneous.
    virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode);
    
    /// PrintAsmMemoryOperand - Print the specified operand of MI, an INLINEASM
    /// instruction, using the specified assembler variant as an address.
    /// Targets should override this to format as appropriate.  This method can
    /// return true if the operand is erroneous.
    virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant, 
                                       const char *ExtraCode);
    
    /// SetupMachineFunction - This should be called when a new MachineFunction
    /// is being processed from runOnMachineFunction.
    void SetupMachineFunction(MachineFunction &MF);
    
    /// getFunctionNumber - Return a unique ID for the current function.
    ///
    unsigned getFunctionNumber() const { return FunctionNumber; }
    
    /// IncrementFunctionNumber - Increase Function Number.  AsmPrinters should
    /// not normally call this, as the counter is automatically bumped by
    /// SetupMachineFunction.
    void IncrementFunctionNumber() { FunctionNumber++; }
    
    /// EmitConstantPool - Print to the current output stream assembly
    /// representations of the constants in the constant pool MCP. This is
    /// used to print out constants which have been "spilled to memory" by
    /// the code generator.
    ///
    virtual void EmitConstantPool(MachineConstantPool *MCP);

    /// EmitJumpTableInfo - Print assembly representations of the jump tables 
    /// used by the current function to the current output stream.  
    ///
    void EmitJumpTableInfo(MachineJumpTableInfo *MJTI);
    
    /// EmitSpecialLLVMGlobal - Check to see if the specified global is a
    /// special global used by LLVM.  If so, emit it and return true, otherwise
    /// do nothing and return false.
    bool EmitSpecialLLVMGlobal(const GlobalVariable *GV);

    /// EmitAlignment - Emit an alignment directive to the specified power of
    /// two boundary.  For example, if you pass in 3 here, you will get an 8
    /// byte alignment.  If a global value is specified, and if that global has
    /// an explicit alignment requested, it will override the alignment request.
    void EmitAlignment(unsigned NumBits, const GlobalValue *GV = 0) const;

    /// EmitZeros - Emit a block of zeros.
    ///
    void EmitZeros(uint64_t NumZeros) const;

    /// EmitString - Emit a zero-byte-terminated string constant.
    ///
    virtual void EmitString(const ConstantArray *CVA) const;

    /// EmitConstantValueOnly - Print out the specified constant, without a
    /// storage class.  Only constants of first-class type are allowed here.
    void EmitConstantValueOnly(const Constant *CV);

    /// EmitGlobalConstant - Print a general LLVM constant to the .s file.
    ///
    void EmitGlobalConstant(const Constant* CV);
    
    /// printInlineAsm - This method formats and prints the specified machine
    /// instruction that is an inline asm.
    void printInlineAsm(const MachineInstr *MI) const;
    
    /// printBasicBlockLabel - This method prints the label for the specified
    /// MachineBasicBlock
    virtual void printBasicBlockLabel(const MachineBasicBlock *MBB,
                                      bool printColon = false,
                                      bool printComment = true) const;
    
  private:
    void EmitXXStructorList(Constant *List);

  };
}

#endif
