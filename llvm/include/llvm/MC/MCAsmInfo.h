//===-- llvm/MC/MCAsmInfo.h - Asm info --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the basis for target specific
// asm writers.  This class primarily takes care of global printing constants,
// which are used in very similar ways across all targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ASM_INFO_H
#define LLVM_TARGET_ASM_INFO_H

#include <cassert>

namespace llvm {
  /// MCAsmInfo - This class is intended to be used as a base class for asm
  /// properties and features specific to the target.
  namespace ExceptionHandling { enum ExceptionsType { None, Dwarf, SjLj }; }

  class MCAsmInfo {
  protected:
    //===------------------------------------------------------------------===//
    // Properties to be set by the target writer, used to configure asm printer.
    //

    /// ZeroFillDirective - Directive for emitting a global to the ZeroFill
    /// section on this target.  Null if this target doesn't support zerofill.
    const char *ZeroFillDirective;           // Default is null.

    /// NonexecutableStackDirective - Directive for declaring to the
    /// linker and beyond that the emitted code does not require stack
    /// memory to be executable.
    const char *NonexecutableStackDirective; // Default is null.

    /// NeedsSet - True if target asm treats expressions in data directives
    /// as linktime-relocatable.  For assembly-time computation, we need to
    /// use a .set.  Thus:
    /// .set w, x-y
    /// .long w
    /// is computed at assembly time, while
    /// .long x-y
    /// is relocated if the relative locations of x and y change at linktime.
    /// We want both these things in different places.
    bool NeedsSet;                           // Defaults to false.
    
    /// MaxInstLength - This is the maximum possible length of an instruction,
    /// which is needed to compute the size of an inline asm.
    unsigned MaxInstLength;                  // Defaults to 4.
    
    /// PCSymbol - The symbol used to represent the current PC.  Used in PC
    /// relative expressions.
    const char *PCSymbol;                    // Defaults to "$".

    /// SeparatorChar - This character, if specified, is used to separate
    /// instructions from each other when on the same line.  This is used to
    /// measure inline asm instructions.
    char SeparatorChar;                      // Defaults to ';'

    /// CommentColumn - This indicates the comment num (zero-based) at
    /// which asm comments should be printed.
    unsigned CommentColumn;                  // Defaults to 60

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;               // Defaults to "#"

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;                // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .s file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;         // Defaults to "."
    
    /// LinkerPrivateGlobalPrefix - This prefix is used for symbols that should
    /// be passed through the assembler but be removed by the linker.  This
    /// is "l" on Darwin, currently used for some ObjC metadata.
    const char *LinkerPrivateGlobalPrefix;   // Defaults to ""
    
    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assembly statement.
    const char *InlineAsmStart;              // Defaults to "#APP\n"
    const char *InlineAsmEnd;                // Defaults to "#NO_APP\n"

    /// AssemblerDialect - Which dialect of an assembler variant to use.
    unsigned AssemblerDialect;               // Defaults to 0

    /// AllowQuotesInName - This is true if the assembler allows for complex
    /// symbol names to be surrounded in quotes.  This defaults to false.
    bool AllowQuotesInName;

    /// AllowNameToStartWithDigit - This is true if the assembler allows symbol
    /// names to start with a digit (e.g., "0x0021").  This defaults to false.
    bool AllowNameToStartWithDigit;
    
    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;               // Defaults to "\t.zero\t"
    const char *ZeroDirectiveSuffix;         // Defaults to ""

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;              // Defaults to "\t.ascii\t"
    
    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;              // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;          // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;         // Defaults to "\t.short\t"
    const char *Data32bitsDirective;         // Defaults to "\t.long\t"
    const char *Data64bitsDirective;         // Defaults to "\t.quad\t"

    /// getDataASDirective - Return the directive that should be used to emit
    /// data of the specified size to the specified numeric address space.
    virtual const char *getDataASDirective(unsigned Size, unsigned AS) const {
      assert(AS != 0 && "Don't know the directives for default addr space");
      return 0;
    }

    /// SunStyleELFSectionSwitchSyntax - This is true if this target uses "Sun
    /// Style" syntax for section switching ("#alloc,#write" etc) instead of the
    /// normal ELF syntax (,"a,w") in .section directives.
    bool SunStyleELFSectionSwitchSyntax;     // Defaults to false.

    /// UsesELFSectionDirectiveForBSS - This is true if this target uses ELF
    /// '.section' directive before the '.bss' one. It's used for PPC/Linux 
    /// which doesn't support the '.bss' directive only.
    bool UsesELFSectionDirectiveForBSS;      // Defaults to false.
    
    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;              // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;                 // Defaults to true

    /// TextAlignFillValue - If non-zero, this is used to fill the executable
    /// space created as the result of a alignment directive.
    unsigned TextAlignFillValue;             // Defaults to 0

    //===--- Section Switching Directives ---------------------------------===//
    
    /// JumpTableDirective - if non-null, the directive to emit before jump
    /// table entries.  FIXME: REMOVE THIS.
    const char *JumpTableDirective;          // Defaults to NULL.
    const char *PICJumpTableDirective;       // Defaults to NULL.


    //===--- Global Variable Emission Directives --------------------------===//
    
    /// GlobalDirective - This is the directive used to declare a global entity.
    ///
    const char *GlobalDirective;             // Defaults to NULL.

    /// ExternDirective - This is the directive used to declare external 
    /// globals.
    ///
    const char *ExternDirective;             // Defaults to NULL.
    
    /// SetDirective - This is the name of a directive that can be used to tell
    /// the assembler to set the value of a variable to some expression.
    const char *SetDirective;                // Defaults to null.
    
    /// LCOMMDirective - This is the name of a directive (if supported) that can
    /// be used to efficiently declare a local (internal) block of zero
    /// initialized data in the .bss/.data section.  The syntax expected is:
    /// @verbatim <LCOMMDirective> SYMBOLNAME LENGTHINBYTES, ALIGNMENT
    /// @endverbatim
    const char *LCOMMDirective;              // Defaults to null.
    
    const char *COMMDirective;               // Defaults to "\t.comm\t".

    /// COMMDirectiveTakesAlignment - True if COMMDirective take a third
    /// argument that specifies the alignment of the declaration.
    bool COMMDirectiveTakesAlignment;        // Defaults to true.
    
    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;         // Defaults to true.

    /// HasSingleParameterDotFile - True if the target has a single parameter
    /// .file directive, this is true for ELF targets.
    bool HasSingleParameterDotFile;          // Defaults to true.

    /// UsedDirective - This directive, if non-null, is used to declare a global
    /// as being used somehow that the assembler can't see.  This prevents dead
    /// code elimination on some targets.
    const char *UsedDirective;               // Defaults to NULL.

    /// WeakRefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak undefined symbol.
    const char *WeakRefDirective;            // Defaults to NULL.
    
    /// WeakDefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak defined symbol.
    const char *WeakDefDirective;            // Defaults to NULL.
    
    /// HiddenDirective - This directive, if non-null, is used to declare a
    /// global or function as having hidden visibility.
    const char *HiddenDirective;             // Defaults to "\t.hidden\t".

    /// ProtectedDirective - This directive, if non-null, is used to declare a
    /// global or function as having protected visibility.
    const char *ProtectedDirective;          // Defaults to "\t.protected\t".

    //===--- Dwarf Emission Directives -----------------------------------===//

    /// AbsoluteDebugSectionOffsets - True if we should emit abolute section
    /// offsets for debug information.
    bool AbsoluteDebugSectionOffsets;        // Defaults to false.

    /// AbsoluteEHSectionOffsets - True if we should emit abolute section
    /// offsets for EH information. Defaults to false.
    bool AbsoluteEHSectionOffsets;

    /// HasLEB128 - True if target asm supports leb128 directives.
    bool HasLEB128;                          // Defaults to false.

    /// hasDotLocAndDotFile - True if target asm supports .loc and .file
    /// directives for emitting debugging information.
    bool HasDotLocAndDotFile;                // Defaults to false.

    /// SupportsDebugInformation - True if target supports emission of debugging
    /// information.
    bool SupportsDebugInformation;           // Defaults to false.

    /// SupportsExceptionHandling - True if target supports exception handling.
    ExceptionHandling::ExceptionsType ExceptionsType; // Defaults to None

    /// RequiresFrameSection - true if the Dwarf2 output needs a frame section
    bool DwarfRequiresFrameSection;          // Defaults to true.

    /// DwarfUsesInlineInfoSection - True if DwarfDebugInlineSection is used to
    /// encode inline subroutine information.
    bool DwarfUsesInlineInfoSection;         // Defaults to false.

    /// Is_EHSymbolPrivate - If set, the "_foo.eh" is made private so that it
    /// doesn't show up in the symbol table of the object file.
    bool Is_EHSymbolPrivate;                 // Defaults to true.

    /// GlobalEHDirective - This is the directive used to make exception frame
    /// tables globally visible.
    const char *GlobalEHDirective;           // Defaults to NULL.

    /// SupportsWeakEmptyEHFrame - True if target assembler and linker will
    /// handle a weak_definition of constant 0 for an omitted EH frame.
    bool SupportsWeakOmittedEHFrame;         // Defaults to true.

    /// DwarfSectionOffsetDirective - Special section offset directive.
    const char* DwarfSectionOffsetDirective; // Defaults to NULL
    
    //===--- CBE Asm Translation Table -----------------------------------===//

    const char *const *AsmTransCBE;          // Defaults to empty

  public:
    explicit MCAsmInfo();
    virtual ~MCAsmInfo();

    /// getSLEB128Size - Compute the number of bytes required for a signed
    /// leb128 value.
    static unsigned getSLEB128Size(int Value);

    /// getULEB128Size - Compute the number of bytes required for an unsigned
    /// leb128 value.
    static unsigned getULEB128Size(unsigned Value);

    // Data directive accessors.
    //
    const char *getData8bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data8bitsDirective : getDataASDirective(8, AS);
    }
    const char *getData16bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data16bitsDirective : getDataASDirective(16, AS);
    }
    const char *getData32bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data32bitsDirective : getDataASDirective(32, AS);
    }
    const char *getData64bitsDirective(unsigned AS = 0) const {
      return AS == 0 ? Data64bitsDirective : getDataASDirective(64, AS);
    }

    
    bool usesSunStyleELFSectionSwitchSyntax() const {
      return SunStyleELFSectionSwitchSyntax;
    }
    
    bool usesELFSectionDirectiveForBSS() const {
      return UsesELFSectionDirectiveForBSS;
    }

    // Accessors.
    //
    const char *getZeroFillDirective() const {
      return ZeroFillDirective;
    }
    bool hasZeroFillDirective() const { return ZeroFillDirective != 0; }
    
    const char *getNonexecutableStackDirective() const {
      return NonexecutableStackDirective;
    }
    bool needsSet() const {
      return NeedsSet;
    }
    unsigned getMaxInstLength() const {
      return MaxInstLength;
    }
    const char *getPCSymbol() const {
      return PCSymbol;
    }
    char getSeparatorChar() const {
      return SeparatorChar;
    }
    unsigned getCommentColumn() const {
      return CommentColumn;
    }
    const char *getCommentString() const {
      return CommentString;
    }
    const char *getGlobalPrefix() const {
      return GlobalPrefix;
    }
    const char *getPrivateGlobalPrefix() const {
      return PrivateGlobalPrefix;
    }
    const char *getLinkerPrivateGlobalPrefix() const {
      return LinkerPrivateGlobalPrefix;
    }
    const char *getInlineAsmStart() const {
      return InlineAsmStart;
    }
    const char *getInlineAsmEnd() const {
      return InlineAsmEnd;
    }
    unsigned getAssemblerDialect() const {
      return AssemblerDialect;
    }
    bool doesAllowQuotesInName() const {
      return AllowQuotesInName;
    }
    bool doesAllowNameToStartWithDigit() const {
      return AllowNameToStartWithDigit;
    }
    const char *getZeroDirective() const {
      return ZeroDirective;
    }
    const char *getZeroDirectiveSuffix() const {
      return ZeroDirectiveSuffix;
    }
    const char *getAsciiDirective() const {
      return AsciiDirective;
    }
    const char *getAscizDirective() const {
      return AscizDirective;
    }
    const char *getJumpTableDirective(bool isPIC) const {
      return isPIC ? PICJumpTableDirective : JumpTableDirective;
    }
    const char *getAlignDirective() const {
      return AlignDirective;
    }
    bool getAlignmentIsInBytes() const {
      return AlignmentIsInBytes;
    }
    unsigned getTextAlignFillValue() const {
      return TextAlignFillValue;
    }
    const char *getGlobalDirective() const {
      return GlobalDirective;
    }
    const char *getExternDirective() const {
      return ExternDirective;
    }
    const char *getSetDirective() const {
      return SetDirective;
    }
    const char *getLCOMMDirective() const {
      return LCOMMDirective;
    }
    const char *getCOMMDirective() const {
      return COMMDirective;
    }
    bool getCOMMDirectiveTakesAlignment() const {
      return COMMDirectiveTakesAlignment;
    }
    bool hasDotTypeDotSizeDirective() const {
      return HasDotTypeDotSizeDirective;
    }
    bool hasSingleParameterDotFile() const {
      return HasSingleParameterDotFile;
    }
    const char *getUsedDirective() const {
      return UsedDirective;
    }
    const char *getWeakRefDirective() const {
      return WeakRefDirective;
    }
    const char *getWeakDefDirective() const {
      return WeakDefDirective;
    }
    const char *getHiddenDirective() const {
      return HiddenDirective;
    }
    const char *getProtectedDirective() const {
      return ProtectedDirective;
    }
    bool isAbsoluteDebugSectionOffsets() const {
      return AbsoluteDebugSectionOffsets;
    }
    bool isAbsoluteEHSectionOffsets() const {
      return AbsoluteEHSectionOffsets;
    }
    bool hasLEB128() const {
      return HasLEB128;
    }
    bool hasDotLocAndDotFile() const {
      return HasDotLocAndDotFile;
    }
    bool doesSupportDebugInformation() const {
      return SupportsDebugInformation;
    }
    bool doesSupportExceptionHandling() const {
      return ExceptionsType != ExceptionHandling::None;
    }
    ExceptionHandling::ExceptionsType getExceptionHandlingType() const {
      return ExceptionsType;
    }
    bool doesDwarfRequireFrameSection() const {
      return DwarfRequiresFrameSection;
    }
    bool doesDwarfUsesInlineInfoSection() const {
      return DwarfUsesInlineInfoSection;
    }
    bool is_EHSymbolPrivate() const {
      return Is_EHSymbolPrivate;
    }
    const char *getGlobalEHDirective() const {
      return GlobalEHDirective;
    }
    bool getSupportsWeakOmittedEHFrame() const {
      return SupportsWeakOmittedEHFrame;
    }
    const char *getDwarfSectionOffsetDirective() const {
      return DwarfSectionOffsetDirective;
    }
    const char *const *getAsmCBE() const {
      return AsmTransCBE;
    }
  };
}

#endif
