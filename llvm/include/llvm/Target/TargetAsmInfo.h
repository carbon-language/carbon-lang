//===-- llvm/Target/TargetAsmInfo.h - Asm info ------------------*- C++ -*-===//
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

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
  template <typename T> class SmallVectorImpl;
  class TargetMachine;
  class GlobalValue;
  class Mangler;
  
  // DWARF encoding query type
  namespace DwarfEncoding {
    enum Target {
      Data       = 0,
      CodeLabels = 1,
      Functions  = 2
    };
  }

  /// SectionKind - This is a simple POD value that classifies the properties of
  /// a section.  A global variable is classified into the deepest possible
  /// classification, and then the target maps them onto their sections based on
  /// what capabilities they have.
  ///
  /// The comments below describe these as if they were an inheritance hierarchy
  /// in order to explain the predicates below.
  class SectionKind {
  public:
    enum Kind {
      /// Metadata - Debug info sections or other metadata.
      Metadata,
      
      /// Text - Text section, used for functions and other executable code.
      Text,
      
      /// ReadOnly - Data that is never written to at program runtime by the
      /// program or the dynamic linker.  Things in the top-level readonly
      /// SectionKind are not mergeable.
      ReadOnly,

          /// MergeableCString - This is a special section for nul-terminated
          /// strings.  The linker can unique the C strings, knowing their
          /// semantics.  Because it uniques based on the nul terminators, the
          /// compiler can't put strings in this section that have embeded nuls
          /// in them.
          MergeableCString,
      
          /// MergeableConst - These are sections for merging fixed-length
          /// constants together.  For example, this can be used to unique
          /// constant pool entries etc.
          MergeableConst,
      
              /// MergeableConst4 - This is a section used by 4-byte constants,
              /// for example, floats.
              MergeableConst4,
      
              /// MergeableConst8 - This is a section used by 8-byte constants,
              /// for example, doubles.
              MergeableConst8,

              /// MergeableConst16 - This is a section used by 16-byte constants,
              /// for example, vectors.
              MergeableConst16,
      
      /// Writeable - This is the base of all segments that need to be written
      /// to during program runtime.
      
         /// ThreadLocal - This is the base of all TLS segments.  All TLS
         /// objects must be writeable, otherwise there is no reason for them to
         /// be thread local!
      
             /// ThreadBSS - Zero-initialized TLS data objects.
             ThreadBSS,
      
             /// ThreadData - Initialized TLS data objects.
             ThreadData,
      
         /// GlobalWriteableData - Writeable data that is global (not thread
         /// local).
      
             /// BSS - Zero initialized writeable data.
             BSS,

             /// DataRel - This is the most general form of data that is written
             /// to by the program, it can have random relocations to arbitrary
             /// globals.
             DataRel,

                 /// DataRelLocal - This is writeable data that has a non-zero
                 /// initializer and has relocations in it, but all of the
                 /// relocations are known to be within the final linked image
                 /// the global is linked into.
                 DataRelLocal,

                     /// DataNoRel - This is writeable data that has a non-zero
                     /// initializer, but whose initializer is known to have no
                     /// relocations.
                     DataNoRel,

             /// ReadOnlyWithRel - These are global variables that are never
             /// written to by the program, but that have relocations, so they
             /// must be stuck in a writeable section so that the dynamic linker
             /// can write to them.  If it chooses to, the dynamic linker can
             /// mark the pages these globals end up on as read-only after it is
             /// done with its relocation phase.
             ReadOnlyWithRel,
      
                 /// ReadOnlyWithRelLocal - This is data that is readonly by the
                 /// program, but must be writeable so that the dynamic linker
                 /// can perform relocations in it.  This is used when we know
                 /// that all the relocations are to globals in this final
                 /// linked image.
                 ReadOnlyWithRelLocal
      
    };
    
  private:
    Kind K : 6;
    
    /// Weak - This is true if the referenced symbol is weak (i.e. linkonce,
    /// weak, weak_odr, etc).  This is orthogonal from the categorization.
    bool Weak : 1;
    
    /// ExplicitSection - This is true if the global had a section explicitly
    /// specified on it.
    bool ExplicitSection : 1;
  public:
    
    // FIXME: REMOVE.
    Kind getKind() const { return K; }
    
    bool isWeak() const { return Weak; }
    bool hasExplicitSection() const { return ExplicitSection; }
    
    
    bool isMetadata() const { return K == Metadata; }
    bool isText() const { return K == Text; }
    
    bool isReadOnly() const {
      return K == ReadOnly || K == MergeableCString || isMergeableConst();
    }

    bool isMergeableCString() const { return K == MergeableCString; }
    bool isMergeableConst() const {
      return K == MergeableConst || K == MergeableConst4 ||
             K == MergeableConst8 || K == MergeableConst16;
    }
    
    bool isMergeableConst4() const { return K == MergeableConst4; }
    bool isMergeableConst8() const { return K == MergeableConst8; }
    bool isMergeableConst16() const { return K == MergeableConst16; }
    
    bool isWriteable() const {
      return isThreadLocal() || isGlobalWriteableData();
    }
    
    bool isThreadLocal() const {
      return K == ThreadData || K == ThreadBSS;
    }
    
    bool isThreadBSS() const { return K == ThreadBSS; } 
    bool isThreadData() const { return K == ThreadData; } 

    bool isGlobalWriteableData() const {
      return isBSS() || isDataRel() || isReadOnlyWithRel();
    }
    
    bool isBSS() const { return K == BSS; }
    
    bool isDataRel() const {
      return K == DataRel || K == DataRelLocal || K == DataNoRel;
    }
    
    bool isDataRelLocal() const {
      return K == DataRelLocal || K == DataNoRel;
    }

    bool isDataNoRel() const { return K == DataNoRel; }
    
    bool isReadOnlyWithRel() const {
      return K == ReadOnlyWithRel || K == ReadOnlyWithRelLocal;
    }

    bool isReadOnlyWithRelLocal() const {
      return K == ReadOnlyWithRelLocal;
    }
    
    static SectionKind get(Kind K, bool isWeak = false,
                           bool hasExplicitSection = false) {
      SectionKind Res;
      Res.K = K;
      Res.Weak = isWeak;
      Res.ExplicitSection = hasExplicitSection;
      return Res;
    }
  };

  class Section {
    friend class TargetAsmInfo;
    friend class StringMapEntry<Section>;
    friend class StringMap<Section>;

    std::string Name;
    SectionKind Kind;
    explicit Section() { }

  public:
    const std::string &getName() const { return Name; }
    SectionKind getKind() const { return Kind; }
  };

  /// TargetAsmInfo - This class is intended to be used as a base class for asm
  /// properties and features specific to the target.
  class TargetAsmInfo {
  private:
    mutable StringMap<Section> Sections;
  protected:
    /// TM - The current TargetMachine.
    const TargetMachine &TM;

    //===------------------------------------------------------------------===//
    // Properties to be set by the target writer, used to configure asm printer.
    //

    /// TextSection - Section directive for standard text.
    ///
    const Section *TextSection;           // Defaults to ".text".

    /// DataSection - Section directive for standard data.
    ///
    const Section *DataSection;           // Defaults to ".data".

    /// BSSSection - Section directive for uninitialized data.  Null if this
    /// target doesn't support a BSS section.
    ///
    const char *BSSSection;               // Default to ".bss".
    const Section *BSSSection_;

    /// ReadOnlySection - This is the directive that is emitted to switch to a
    /// read-only section for constant data (e.g. data declared const,
    /// jump tables).
    const Section *ReadOnlySection;       // Defaults to NULL

    /// TLSDataSection - Section directive for Thread Local data.
    ///
    const Section *TLSDataSection;        // Defaults to ".tdata".

    /// TLSBSSSection - Section directive for Thread Local uninitialized data.
    /// Null if this target doesn't support a BSS section.
    ///
    const Section *TLSBSSSection;         // Defaults to ".tbss".

    /// ZeroFillDirective - Directive for emitting a global to the ZeroFill
    /// section on this target.  Null if this target doesn't support zerofill.
    const char *ZeroFillDirective;        // Default is null.

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
    bool NeedsSet;                        // Defaults to false.
    
    /// MaxInstLength - This is the maximum possible length of an instruction,
    /// which is needed to compute the size of an inline asm.
    unsigned MaxInstLength;               // Defaults to 4.
    
    /// PCSymbol - The symbol used to represent the current PC.  Used in PC
    /// relative expressions.
    const char *PCSymbol;                 // Defaults to "$".

    /// SeparatorChar - This character, if specified, is used to separate
    /// instructions from each other when on the same line.  This is used to
    /// measure inline asm instructions.
    char SeparatorChar;                   // Defaults to ';'

    /// CommentColumn - This indicates the comment num (zero-based) at
    /// which asm comments should be printed.
    unsigned CommentColumn;               // Defaults to 60

    /// CommentString - This indicates the comment character used by the
    /// assembler.
    const char *CommentString;            // Defaults to "#"

    /// FirstOperandColumn - The output column where the first operand
    /// should be printed
    unsigned FirstOperandColumn;          // Defaults to 0 (ignored)

    /// MaxOperandLength - The maximum length of any printed asm
    /// operand
    unsigned MaxOperandLength;            // Defaults to 0 (ignored)

    /// GlobalPrefix - If this is set to a non-empty string, it is prepended
    /// onto all global symbols.  This is often used for "_" or ".".
    const char *GlobalPrefix;             // Defaults to ""

    /// PrivateGlobalPrefix - This prefix is used for globals like constant
    /// pool entries that are completely private to the .s file and should not
    /// have names in the .o file.  This is often "." or "L".
    const char *PrivateGlobalPrefix;      // Defaults to "."
    
    /// LinkerPrivateGlobalPrefix - This prefix is used for symbols that should
    /// be passed through the assembler but be removed by the linker.  This
    /// is "l" on Darwin, currently used for some ObjC metadata.
    const char *LinkerPrivateGlobalPrefix;      // Defaults to ""
    
    /// JumpTableSpecialLabelPrefix - If not null, a extra (dead) label is
    /// emitted before jump tables with the specified prefix.
    const char *JumpTableSpecialLabelPrefix;  // Default to null.
    
    /// GlobalVarAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable (that isn't a function)
    ///
    const char *GlobalVarAddrPrefix;      // Defaults to ""
    const char *GlobalVarAddrSuffix;      // Defaults to ""

    /// FunctionAddrPrefix/Suffix - If these are nonempty, these strings
    /// will enclose any GlobalVariable that points to a function.
    ///
    const char *FunctionAddrPrefix;       // Defaults to ""
    const char *FunctionAddrSuffix;       // Defaults to ""

    /// PersonalityPrefix/Suffix - If these are nonempty, these strings will
    /// enclose any personality function in the common frame section.
    /// 
    const char *PersonalityPrefix;        // Defaults to ""
    const char *PersonalitySuffix;        // Defaults to ""

    /// NeedsIndirectEncoding - If set, we need to set the indirect encoding bit
    /// for EH in Dwarf.
    /// 
    bool NeedsIndirectEncoding;           // Defaults to false

    /// InlineAsmStart/End - If these are nonempty, they contain a directive to
    /// emit before and after an inline assembly statement.
    const char *InlineAsmStart;           // Defaults to "#APP\n"
    const char *InlineAsmEnd;             // Defaults to "#NO_APP\n"

    /// AssemblerDialect - Which dialect of an assembler variant to use.
    unsigned AssemblerDialect;            // Defaults to 0

    /// AllowQuotesInName - This is true if the assembler allows for complex
    /// symbol names to be surrounded in quotes.  This defaults to false.
    bool AllowQuotesInName;
    
    //===--- Data Emission Directives -------------------------------------===//

    /// ZeroDirective - this should be set to the directive used to get some
    /// number of zero bytes emitted to the current section.  Common cases are
    /// "\t.zero\t" and "\t.space\t".  If this is set to null, the
    /// Data*bitsDirective's will be used to emit zero bytes.
    const char *ZeroDirective;            // Defaults to "\t.zero\t"
    const char *ZeroDirectiveSuffix;      // Defaults to ""

    /// AsciiDirective - This directive allows emission of an ascii string with
    /// the standard C escape characters embedded into it.
    const char *AsciiDirective;           // Defaults to "\t.ascii\t"
    
    /// AscizDirective - If not null, this allows for special handling of
    /// zero terminated strings on this target.  This is commonly supported as
    /// ".asciz".  If a target doesn't support this, it can be set to null.
    const char *AscizDirective;           // Defaults to "\t.asciz\t"

    /// DataDirectives - These directives are used to output some unit of
    /// integer data to the current section.  If a data directive is set to
    /// null, smaller data directives will be used to emit the large sizes.
    const char *Data8bitsDirective;       // Defaults to "\t.byte\t"
    const char *Data16bitsDirective;      // Defaults to "\t.short\t"
    const char *Data32bitsDirective;      // Defaults to "\t.long\t"
    const char *Data64bitsDirective;      // Defaults to "\t.quad\t"

    /// getDataASDirective - Return the directive that should be used to emit
    /// data of the specified size to the specified numeric address space.
    virtual const char *getDataASDirective(unsigned Size, unsigned AS) const {
      assert(AS != 0 && "Don't know the directives for default addr space");
      return NULL;
    }

    //===--- Alignment Information ----------------------------------------===//

    /// AlignDirective - The directive used to emit round up to an alignment
    /// boundary.
    ///
    const char *AlignDirective;           // Defaults to "\t.align\t"

    /// AlignmentIsInBytes - If this is true (the default) then the asmprinter
    /// emits ".align N" directives, where N is the number of bytes to align to.
    /// Otherwise, it emits ".align log2(N)", e.g. 3 to align to an 8 byte
    /// boundary.
    bool AlignmentIsInBytes;              // Defaults to true

    /// TextAlignFillValue - If non-zero, this is used to fill the executable
    /// space created as the result of a alignment directive.
    unsigned TextAlignFillValue;

    //===--- Section Switching Directives ---------------------------------===//
    
    /// SwitchToSectionDirective - This is the directive used when we want to
    /// emit a global to an arbitrary section.  The section name is emited after
    /// this.
    const char *SwitchToSectionDirective; // Defaults to "\t.section\t"
    
    /// TextSectionStartSuffix - This is printed after each start of section
    /// directive for text sections.
    const char *TextSectionStartSuffix;   // Defaults to "".

    /// DataSectionStartSuffix - This is printed after each start of section
    /// directive for data sections.
    const char *DataSectionStartSuffix;   // Defaults to "".
    
    /// SectionEndDirectiveSuffix - If non-null, the asm printer will close each
    /// section with the section name and this suffix printed.
    const char *SectionEndDirectiveSuffix;// Defaults to null.
    
    /// ConstantPoolSection - This is the section that we SwitchToSection right
    /// before emitting the constant pool for a function.
    const char *ConstantPoolSection;      // Defaults to "\t.section .rodata"

    /// JumpTableDataSection - This is the section that we SwitchToSection right
    /// before emitting the jump tables for a function when the relocation model
    /// is not PIC.
    const char *JumpTableDataSection;     // Defaults to "\t.section .rodata"
    
    /// JumpTableDirective - if non-null, the directive to emit before a jump
    /// table.
    const char *JumpTableDirective;

    /// CStringSection - If not null, this allows for special handling of
    /// cstring constants (null terminated string that does not contain any
    /// other null bytes) on this target. This is commonly supported as
    /// ".cstring".
    const char *CStringSection;           // Defaults to NULL
    const Section *CStringSection_;

    /// StaticCtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static constructor list.
    /// Defaults to "\t.section .ctors,\"aw\",@progbits".
    const char *StaticCtorsSection;

    /// StaticDtorsSection - This is the directive that is emitted to switch to
    /// a section to emit the static destructor list.
    /// Defaults to "\t.section .dtors,\"aw\",@progbits".
    const char *StaticDtorsSection;

    //===--- Global Variable Emission Directives --------------------------===//
    
    /// GlobalDirective - This is the directive used to declare a global entity.
    ///
    const char *GlobalDirective;          // Defaults to NULL.

    /// ExternDirective - This is the directive used to declare external 
    /// globals.
    ///
    const char *ExternDirective;          // Defaults to NULL.
    
    /// SetDirective - This is the name of a directive that can be used to tell
    /// the assembler to set the value of a variable to some expression.
    const char *SetDirective;             // Defaults to null.
    
    /// LCOMMDirective - This is the name of a directive (if supported) that can
    /// be used to efficiently declare a local (internal) block of zero
    /// initialized data in the .bss/.data section.  The syntax expected is:
    /// @verbatim <LCOMMDirective> SYMBOLNAME LENGTHINBYTES, ALIGNMENT
    /// @endverbatim
    const char *LCOMMDirective;           // Defaults to null.
    
    const char *COMMDirective;            // Defaults to "\t.comm\t".

    /// COMMDirectiveTakesAlignment - True if COMMDirective take a third
    /// argument that specifies the alignment of the declaration.
    bool COMMDirectiveTakesAlignment;     // Defaults to true.
    
    /// HasDotTypeDotSizeDirective - True if the target has .type and .size
    /// directives, this is true for most ELF targets.
    bool HasDotTypeDotSizeDirective;      // Defaults to true.

    /// HasSingleParameterDotFile - True if the target has a single parameter
    /// .file directive, this is true for ELF targets.
    bool HasSingleParameterDotFile;      // Defaults to true.

    /// UsedDirective - This directive, if non-null, is used to declare a global
    /// as being used somehow that the assembler can't see.  This prevents dead
    /// code elimination on some targets.
    const char *UsedDirective;            // Defaults to null.

    /// WeakRefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak undefined symbol.
    const char *WeakRefDirective;         // Defaults to null.
    
    /// WeakDefDirective - This directive, if non-null, is used to declare a
    /// global as being a weak defined symbol.
    const char *WeakDefDirective;         // Defaults to null.
    
    /// HiddenDirective - This directive, if non-null, is used to declare a
    /// global or function as having hidden visibility.
    const char *HiddenDirective;          // Defaults to "\t.hidden\t".

    /// ProtectedDirective - This directive, if non-null, is used to declare a
    /// global or function as having protected visibility.
    const char *ProtectedDirective;       // Defaults to "\t.protected\t".

    //===--- Dwarf Emission Directives -----------------------------------===//

    /// AbsoluteDebugSectionOffsets - True if we should emit abolute section
    /// offsets for debug information. Defaults to false.
    bool AbsoluteDebugSectionOffsets;

    /// AbsoluteEHSectionOffsets - True if we should emit abolute section
    /// offsets for EH information. Defaults to false.
    bool AbsoluteEHSectionOffsets;

    /// HasLEB128 - True if target asm supports leb128 directives.
    ///
    bool HasLEB128; // Defaults to false.

    /// hasDotLocAndDotFile - True if target asm supports .loc and .file
    /// directives for emitting debugging information.
    ///
    bool HasDotLocAndDotFile; // Defaults to false.

    /// SupportsDebugInformation - True if target supports emission of debugging
    /// information.
    bool SupportsDebugInformation;

    /// SupportsExceptionHandling - True if target supports
    /// exception handling.
    ///
    bool SupportsExceptionHandling; // Defaults to false.

    /// RequiresFrameSection - true if the Dwarf2 output needs a frame section
    ///
    bool DwarfRequiresFrameSection; // Defaults to true.

    /// DwarfUsesInlineInfoSection - True if DwarfDebugInlineSection is used to
    /// encode inline subroutine information.
    bool DwarfUsesInlineInfoSection; // Defaults to false.

    /// Is_EHSymbolPrivate - If set, the "_foo.eh" is made private so that it
    /// doesn't show up in the symbol table of the object file.
    bool Is_EHSymbolPrivate;                // Defaults to true.

    /// GlobalEHDirective - This is the directive used to make exception frame
    /// tables globally visible.
    ///
    const char *GlobalEHDirective;          // Defaults to NULL.

    /// SupportsWeakEmptyEHFrame - True if target assembler and linker will
    /// handle a weak_definition of constant 0 for an omitted EH frame.
    bool SupportsWeakOmittedEHFrame;  // Defaults to true.

    /// DwarfSectionOffsetDirective - Special section offset directive.
    const char* DwarfSectionOffsetDirective; // Defaults to NULL
    
    /// DwarfAbbrevSection - Section directive for Dwarf abbrev.
    ///
    const char *DwarfAbbrevSection; // Defaults to ".debug_abbrev".

    /// DwarfInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfInfoSection; // Defaults to ".debug_info".

    /// DwarfLineSection - Section directive for Dwarf info.
    ///
    const char *DwarfLineSection; // Defaults to ".debug_line".
    
    /// DwarfFrameSection - Section directive for Dwarf info.
    ///
    const char *DwarfFrameSection; // Defaults to ".debug_frame".
    
    /// DwarfPubNamesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubNamesSection; // Defaults to ".debug_pubnames".
    
    /// DwarfPubTypesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubTypesSection; // Defaults to ".debug_pubtypes".

    /// DwarfDebugInlineSection - Section directive for inline info.
    ///
    const char *DwarfDebugInlineSection; // Defaults to ".debug_inlined"

    /// DwarfStrSection - Section directive for Dwarf info.
    ///
    const char *DwarfStrSection; // Defaults to ".debug_str".

    /// DwarfLocSection - Section directive for Dwarf info.
    ///
    const char *DwarfLocSection; // Defaults to ".debug_loc".

    /// DwarfARangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfARangesSection; // Defaults to ".debug_aranges".

    /// DwarfRangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfRangesSection; // Defaults to ".debug_ranges".

    /// DwarfMacroInfoSection - Section directive for DWARF macro info.
    ///
    const char *DwarfMacroInfoSection; // Defaults to ".debug_macinfo".
    
    /// DwarfEHFrameSection - Section directive for Exception frames.
    ///
    const char *DwarfEHFrameSection; // Defaults to ".eh_frame".
    
    /// DwarfExceptionSection - Section directive for Exception table.
    ///
    const char *DwarfExceptionSection; // Defaults to ".gcc_except_table".

    //===--- CBE Asm Translation Table -----------------------------------===//

    const char *const *AsmTransCBE; // Defaults to empty

  public:
    explicit TargetAsmInfo(const TargetMachine &TM);
    virtual ~TargetAsmInfo();

    const Section *getOrCreateSection(const char *Name,
                                      bool isDirective,
                                      SectionKind::Kind K) const;

    /// Measure the specified inline asm to determine an approximation of its
    /// length.
    virtual unsigned getInlineAsmLength(const char *Str) const;

    /// emitUsedDirectiveFor - This hook allows targets to selectively decide
    /// not to emit the UsedDirective for some symbols in llvm.used.
// FIXME: REMOVE this (rdar://7071300)
    virtual bool emitUsedDirectiveFor(const GlobalValue *GV,
                                      Mangler *Mang) const {
      return (GV!=0);
    }

    /// PreferredEHDataFormat - This hook allows the target to select data
    /// format used for encoding pointers in exception handling data. Reason is
    /// 0 for data, 1 for code labels, 2 for function pointers. Global is true
    /// if the symbol can be relocated.
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;

    
    /// getSectionForMergeableConstant - Given a Mergeable constant with the
    /// specified size and relocation information, return a section that it
    /// should be placed in.
    virtual const Section *getSectionForMergeableConstant(SectionKind Kind)const;

    
    /// getKindForNamedSection - If this target wants to be able to override
    /// section flags based on the name of the section specified for a global
    /// variable, it can implement this.  This is used on ELF systems so that
    /// ".tbss" gets the TLS bit set etc.
    virtual SectionKind::Kind getKindForNamedSection(const char *Section,
                                                     SectionKind::Kind K) const{
      return K;
    }
    
    /// SectionForGlobal - This method computes the appropriate section to emit
    /// the specified global variable or function definition.  This should not
    /// be passed external (or available externally) globals.
    // FIXME: MOVE TO ASMPRINTER.
    const Section* SectionForGlobal(const GlobalValue *GV) const;
    
    /// getSpecialCasedSectionGlobals - Allow the target to completely override
    /// section assignment of a global.
    /// FIXME: ELIMINATE this by making PIC16 implement ADDRESS with
    /// getFlagsForNamedSection.
    virtual const Section *
    getSpecialCasedSectionGlobals(const GlobalValue *GV,
                                  SectionKind Kind) const {
      return 0;
    }
    
    /// getSectionFlagsAsString - Turn the flags in the specified SectionKind
    /// into a string that can be printed to the assembly file after the
    /// ".section foo" part of a section directive.
    virtual void getSectionFlagsAsString(SectionKind Kind,
                                         SmallVectorImpl<char> &Str) const {
    }

// FIXME: Eliminate this.
    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV,
                                                  SectionKind Kind) const;

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


    // Accessors.
    //
    const Section *getTextSection() const {
      return TextSection;
    }
    const Section *getDataSection() const {
      return DataSection;
    }
    const char *getBSSSection() const {
      return BSSSection;
    }
    const Section *getBSSSection_() const {
      return BSSSection_;
    }
    const Section *getReadOnlySection() const {
      return ReadOnlySection;
    }
    const Section *getTLSDataSection() const {
      return TLSDataSection;
    }
    const Section *getTLSBSSSection() const {
      return TLSBSSSection;
    }
    const char *getZeroFillDirective() const {
      return ZeroFillDirective;
    }
    const char *getNonexecutableStackDirective() const {
      return NonexecutableStackDirective;
    }
    bool needsSet() const {
      return NeedsSet;
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
    unsigned getOperandColumn(int operand) const {
      return FirstOperandColumn + (MaxOperandLength+1)*(operand-1);
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
    const char *getJumpTableSpecialLabelPrefix() const {
      return JumpTableSpecialLabelPrefix;
    }
    const char *getGlobalVarAddrPrefix() const {
      return GlobalVarAddrPrefix;
    }
    const char *getGlobalVarAddrSuffix() const {
      return GlobalVarAddrSuffix;
    }
    const char *getFunctionAddrPrefix() const {
      return FunctionAddrPrefix;
    }
    const char *getFunctionAddrSuffix() const {
      return FunctionAddrSuffix;
    }
    const char *getPersonalityPrefix() const {
      return PersonalityPrefix;
    }
    const char *getPersonalitySuffix() const {
      return PersonalitySuffix;
    }
    bool getNeedsIndirectEncoding() const {
      return NeedsIndirectEncoding;
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
    const char *getJumpTableDirective() const {
      return JumpTableDirective;
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
    const char *getSwitchToSectionDirective() const {
      return SwitchToSectionDirective;
    }
    const char *getTextSectionStartSuffix() const {
      return TextSectionStartSuffix;
    }
    const char *getDataSectionStartSuffix() const {
      return DataSectionStartSuffix;
    }
    const char *getSectionEndDirectiveSuffix() const {
      return SectionEndDirectiveSuffix;
    }
    const char *getConstantPoolSection() const {
      return ConstantPoolSection;
    }
    const char *getJumpTableDataSection() const {
      return JumpTableDataSection;
    }
    const char *getCStringSection() const {
      return CStringSection;
    }
    const Section *getCStringSection_() const {
      return CStringSection_;
    }
    const char *getStaticCtorsSection() const {
      return StaticCtorsSection;
    }
    const char *getStaticDtorsSection() const {
      return StaticDtorsSection;
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
      return SupportsExceptionHandling;
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
    const char *getDwarfAbbrevSection() const {
      return DwarfAbbrevSection;
    }
    const char *getDwarfInfoSection() const {
      return DwarfInfoSection;
    }
    const char *getDwarfLineSection() const {
      return DwarfLineSection;
    }
    const char *getDwarfFrameSection() const {
      return DwarfFrameSection;
    }
    const char *getDwarfPubNamesSection() const {
      return DwarfPubNamesSection;
    }
    const char *getDwarfPubTypesSection() const {
      return DwarfPubTypesSection;
    }
    const char *getDwarfDebugInlineSection() const {
      return DwarfDebugInlineSection;
    }
    const char *getDwarfStrSection() const {
      return DwarfStrSection;
    }
    const char *getDwarfLocSection() const {
      return DwarfLocSection;
    }
    const char *getDwarfARangesSection() const {
      return DwarfARangesSection;
    }
    const char *getDwarfRangesSection() const {
      return DwarfRangesSection;
    }
    const char *getDwarfMacroInfoSection() const {
      return DwarfMacroInfoSection;
    }
    const char *getDwarfEHFrameSection() const {
      return DwarfEHFrameSection;
    }
    const char *getDwarfExceptionSection() const {
      return DwarfExceptionSection;
    }
    const char *const *getAsmCBE() const {
      return AsmTransCBE;
    }
  };
}

#endif
