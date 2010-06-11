//===-- ArchSpec.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ArchSpec_h_
#define liblldb_ArchSpec_h_

#if defined(__cplusplus)

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class ArchSpec ArchSpec.h "lldb/Core/ArchSpec.h"
/// @brief An architecture specification class.
///
/// A class designed to be created from a cpu type and subtype, or a
/// string representation.  Keeping all of the conversions of strings
/// to architecture enumeration values confined to this class allows
/// new architecture support to be added easily.
//----------------------------------------------------------------------
class ArchSpec
{
public:
    // Generic CPU types that each m_type needs to know how to convert 
    // their m_cpu and m_sub to.
    typedef enum CPU
    {
        eCPU_Unknown,
        eCPU_arm,
        eCPU_i386,
        eCPU_x86_64,
        eCPU_ppc,
        eCPU_ppc64,
        eCPU_sparc
    };

    //------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Default constructor that initializes the object with invalid
    /// cpu type and subtype values.
    //------------------------------------------------------------------
    ArchSpec ();

    //------------------------------------------------------------------
    /// Constructor with cpu type and subtype.
    ///
    /// Constructor that initializes the object with supplied cpu and
    /// subtypes.
    //------------------------------------------------------------------
    ArchSpec (lldb::ArchitectureType arch_type, uint32_t cpu, uint32_t sub);

    //------------------------------------------------------------------
    /// Construct with architecture name.
    ///
    /// Constructor that initializes the object with supplied
    /// architecture name. There are also predefined values in
    /// Defines.h:
    /// @li \c LLDB_ARCH_DEFAULT
    ///     The arch the current system defaults to when a program is
    ///     launched without any extra attributes or settings.
    ///
    /// @li \c LLDB_ARCH_DEFAULT_32BIT
    ///     The 32 bit arch the current system defaults to (if any)
    ///
    /// @li \c LLDB_ARCH_DEFAULT_32BIT
    ///     The 64 bit arch the current system defaults to (if any)
    //------------------------------------------------------------------
    ArchSpec (const char *arch_name);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~ArchSpec ();

    //------------------------------------------------------------------
    /// Assignment operator.
    ///
    /// @param[in] rhs another ArchSpec object to copy.
    ///
    /// @return a const reference to this object
    //------------------------------------------------------------------
    const ArchSpec&
    operator= (const ArchSpec& rhs);

    //------------------------------------------------------------------
    /// Get a string representation of the contained architecture.
    ///
    /// Gets a C string representation of the current architecture.
    /// If the returned string is a valid architecture name, the string
    /// came from a constant string values that do not need to be freed.
    /// If the returned string uses the "N.M" format, the string comes
    /// from a static buffer that should be copied.
    ///
    /// @return a NULL terminated C string that does not need to be
    ///         freed.
    //------------------------------------------------------------------
    const char *
    AsCString () const;

    //------------------------------------------------------------------
    /// Returns a string representation of the supplied architecture.
    ///
    /// Class function to get a C string representation given a CPU type
    /// and subtype.
    ///
    /// @param[in] cpu The cpu type of the architecture.
    /// @param[in] subtype The cpu subtype of the architecture.
    ///
    /// @return a NULL terminated C string that does not need to be
    ///         freed.
    //------------------------------------------------------------------
    static const char *
    AsCString (lldb::ArchitectureType arch_type, uint32_t cpu, uint32_t subtype);

    //------------------------------------------------------------------
    /// Clears the object state.
    ///
    /// Clears the object state back to a default invalid state.
    //------------------------------------------------------------------
    void
    Clear ();

    //------------------------------------------------------------------
    /// Returns the size in bytes of an address of the current
    /// architecture.
    ///
    /// @return The byte size of an address of the current architecture.
    //------------------------------------------------------------------
    uint32_t
    GetAddressByteSize () const;


    CPU
    GetGenericCPUType () const;

    //------------------------------------------------------------------
    /// CPU subtype get accessor.
    ///
    /// @return The current value of the CPU subtype.
    //------------------------------------------------------------------
    uint32_t
    GetCPUSubtype () const;

    //------------------------------------------------------------------
    /// CPU type get accessor.
    ///
    /// @return The current value of the CPU type.
    //------------------------------------------------------------------
    uint32_t
    GetCPUType () const;

    //------------------------------------------------------------------
    /// Feature flags get accessor.
    ///
    /// @return The current value of the CPU feature flags.
    //------------------------------------------------------------------
    uint32_t
    GetFeatureFlags () const;

    //------------------------------------------------------------------
    /// Get register names of the current architecture.
    ///
    /// Get register names of the current architecture given
    /// a register number, and a flavor for that register number.
    /// There are many different register numbering schemes used
    /// on a host:
    /// @li \c eRegisterKindGCC - gcc compiler register numbering
    /// @li \c eRegisterKindDWARF - DWARF register numbering
    ///
    /// @param[in] reg_num The register number to decode.
    /// @param[in] flavor The flavor of the \a reg_num.
    ///
    /// @return the name of the register as a NULL terminated C string,
    ///         or /c NULL if the \a reg_num is invalid for \a flavor.
    ///         String values that are returned do not need to be freed.
    //------------------------------------------------------------------
    const char *
    GetRegisterName (uint32_t reg_num, uint32_t flavor) const;

    //------------------------------------------------------------------
    /// Get register names for a specified architecture.
    ///
    /// Get register names of the specified architecture given
    /// a register number, and a flavor for that register number.
    /// There are many different register numbering schemes used
    /// on a host:
    ///
    /// @li compiler register numbers (@see eRegisterKindGCC)
    /// @li DWARF register numbers (@see eRegisterKindDWARF)
    ///
    /// @param[in] cpu The cpu type of the architecture specific
    ///            register
    /// @param[in] subtype The cpu subtype of the architecture specific
    ///            register
    /// @param[in] reg_num The register number to decode.
    /// @param[in] flavor The flavor of the \a reg_num.
    ///
    /// @return the name of the register as a NULL terminated C string,
    ///         or /c NULL if the \a reg_num is invalid for \a flavor.
    ///         String values that are returned do not need to be freed.
    //------------------------------------------------------------------
    static const char *
    GetRegisterName (lldb::ArchitectureType arch_type, uint32_t cpu, uint32_t subtype, uint32_t reg_num, uint32_t flavor);

    //------------------------------------------------------------------
    /// Test if the contained architecture is valid.
    ///
    /// @return true if the current architecture is valid, false
    ///         otherwise.
    //------------------------------------------------------------------
    bool
    IsValid () const;

    //------------------------------------------------------------------
    /// Get the memory cost of this object.
    ///
    /// @return
    ///     The number of bytes that this object occupies in memory.
    //------------------------------------------------------------------
    size_t
    MemorySize() const;

    //------------------------------------------------------------------
    /// Change the CPU type and subtype given an architecture name.
    ///
    /// The architecture name supplied can also by one of the generic
    /// system default values:
    /// @li \c LLDB_ARCH_DEFAULT - The arch the current system defaults
    ///        to when a program is launched without any extra
    ///        attributes or settings.
    /// @li \c LLDB_ARCH_DEFAULT_32BIT - The default host architecture
    ///        for 32 bit (if any).
    /// @li \c LLDB_ARCH_DEFAULT_64BIT - The default host architecture
    ///        for 64 bit (if any).
    ///
    /// @param[in] arch_name The name of an architecture.
    ///
    /// @return true if \a arch_name was successfully transformed into
    ///         a valid cpu type and subtype.
    //------------------------------------------------------------------
    bool
    SetArch (const char *arch_name);

    bool
    SetArchFromTargetTriple (const char *arch_name);
    //------------------------------------------------------------------
    /// Change the CPU type and subtype given new values of the cpu
    /// type and subtype.
    ///
    /// @param[in] cpu The new CPU type
    /// @param[in] subtype The new CPU subtype
    //------------------------------------------------------------------
    void
    SetArch (uint32_t cpu, uint32_t subtype);

    //------------------------------------------------------------------
    /// Change the CPU subtype given a new value of the CPU subtype.
    ///
    /// @param[in] subtype The new CPU subtype.
    //------------------------------------------------------------------
    void
    SetCPUSubtype (uint32_t subtype);

    //------------------------------------------------------------------
    /// Change the CPU type given a new value of the CPU type.
    ///
    /// @param[in] cpu The new CPU type.
    //------------------------------------------------------------------
    void
    SetCPUType (uint32_t cpu);

    //------------------------------------------------------------------
    /// Returns the default endianness of the architecture.
    ///
    /// @return The endian enumeration for the default endianness of
    ///     the architecture.
    //------------------------------------------------------------------
    lldb::ByteOrder
    GetDefaultEndian () const;


    lldb::ArchitectureType
    GetType() const
    {
        return m_type;
    }

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    lldb::ArchitectureType m_type;
    //       m_type =>  eArchTypeMachO      eArchTypeELF
    uint32_t m_cpu; //  cpu type            ELF header e_machine
    uint32_t m_sub; //  cpu subtype         nothing
};


//------------------------------------------------------------------
/// @fn bool operator== (const ArchSpec& lhs, const ArchSpec& rhs)
/// @brief Equal to operator.
///
/// Tests two ArchSpec objects to see if they are equal.
///
/// @param[in] lhs The Left Hand Side ArchSpec object to compare.
/// @param[in] rhs The Left Hand Side ArchSpec object to compare.
///
/// @return true if \a lhs is equal to \a rhs
//------------------------------------------------------------------
bool operator==(const ArchSpec& lhs, const ArchSpec& rhs);

//------------------------------------------------------------------
/// @fn bool operator!= (const ArchSpec& lhs, const ArchSpec& rhs)
/// @brief Not equal to operator.
///
/// Tests two ArchSpec objects to see if they are not equal.
///
/// @param[in] lhs The Left Hand Side ArchSpec object to compare.
/// @param[in] rhs The Left Hand Side ArchSpec object to compare.
///
/// @return true if \a lhs is not equal to \a rhs
//------------------------------------------------------------------
bool operator!=(const ArchSpec& lhs, const ArchSpec& rhs);

//------------------------------------------------------------------
/// @fn bool operator< (const ArchSpec& lhs, const ArchSpec& rhs)
/// @brief Less than operator.
///
/// Tests two ArchSpec objects to see if \a lhs is less than \a
/// rhs.
///
/// @param[in] lhs The Left Hand Side ArchSpec object to compare.
/// @param[in] rhs The Left Hand Side ArchSpec object to compare.
///
/// @return true if \a lhs is less than \a rhs
//------------------------------------------------------------------
bool operator< (const ArchSpec& lhs, const ArchSpec& rhs);

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // #ifndef liblldb_ArchSpec_h_
