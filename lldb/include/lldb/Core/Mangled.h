//===-- Mangled.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Mangled_h_
#define liblldb_Mangled_h_
#if defined(__cplusplus)


#include "lldb/lldb-private.h"
#include "lldb/Core/ConstString.h"
#include <vector>

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Mangled Mangled.h "lldb/Core/Mangled.h"
/// @brief A class that handles mangled names.
///
/// Designed to handle mangled names. The demangled version of any names
/// will be computed when the demangled name is accessed through the
/// Demangled() acccessor. This class can also tokenize the demangled
/// version of the name for powerful searches. Functions and symbols
/// could make instances of this class for their mangled names. Uniqued
/// string pools are used for the mangled, demangled, and token string
/// values to allow for faster comparisons and for efficient memory use.
//----------------------------------------------------------------------
class Mangled
{
public:

    //------------------------------------------------------------------
    /// Token type enumerations.
    //------------------------------------------------------------------
    enum TokenType
    {
        eInvalid,       ///< Invalid token value (unitialized value)
        eNameSpace,     ///< The token is a namespace name.
        eMethodName,    ///< The token is a global or class method name
        eType,          ///< The token is a language type
        eTemplate,      ///< The token is a template class
        eTemplateBeg,   ///< The token that indicates the start of a template parameters
        eTemplateEnd,   ///< The token that indicates the end of a template parameters
        eParamsBeg,     ///< The start of a method's parameters (the open parenthesis)
        eParamsEnd,     ///< The end of a method's parameters (the open parenthesis)
        eQualifier,     ///< A language qualifier
        eError          ///< The token failed to parse
    };
    
    enum NamePreference
    {
        ePreferMangled,
        ePreferDemangled
    };

    //------------------------------------------------------------------
    /// Mangled::Token structure
    ///
    /// As demangled names get tokenized, they get broken up into chunks
    /// that have type enumerations (TokenType) and string values. Some of
    /// the tokens are scopes (eTemplateBeg, eTemplateEnd, eParamsBeg,
    /// eParamsEnd) that can indicate depth and searches can take
    /// advantage of these to match using wildcards.
    ///
    /// For example the mangled string:
    ///
    ///     "_ZNSbIhSt11char_traitsIhESaIhEE5eraseEmm"
    ///
    /// Demangles to:
    ///
    ///     "std::basic_string<unsigned char, std::char_traits<unsigned char>, std::allocator<unsigned char> >::erase(unsigned long, unsigned long)"
    ///
    /// And tokenizes to:
    ///     @li eNameSpace ("std")
    ///     @li eTemplate ("basic_string")
    ///     @li eTemplateBeg ()
    ///     @li eType ("unsigned char")
    ///     @li eNameSpace ("std")
    ///     @li eTemplate ("char_traits")
    ///     @li eTemplateBeg ()
    ///     @li eType ("unsigned char")
    ///     @li eTemplateEnd ()
    ///     @li eNameSpace ("std")
    ///     @li eTemplate ("allocator")
    ///     @li eTemplateBeg ()
    ///     @li eType ("unsigned char"
    ///     @li eTemplateEnd ()
    ///     @li eTemplateEnd ()
    ///     @li eMethodName ("erase")
    ///     @li eParamsBeg ()
    ///     @li eType ("unsigned long")
    ///     @li eType ("unsigned long")
    ///     @li eParamsEnd ()
    ///------------------------------------------------------------------
    struct Token
    {
        //--------------------------------------------------------------
        /// Default constructor.
        ///
        /// Constructs this objet with an invalid token type and an
        /// empty string.
        //--------------------------------------------------------------
        Token();

        //--------------------------------------------------------------
        /// Equal to operator.
        ///
        /// Tests if this object is equal to \a rhs.
        ///
        /// @param[in] rhs
        ///     A const Mangled::Token object reference to compare
        ///     this object to.
        ///
        /// @return
        ///     \b true if this object is equal to \a rhs, \b false
        ///     otherwise.
        //--------------------------------------------------------------
        bool
        operator== (const Token& rhs) const;

        //--------------------------------------------------------------
        /// Dump a description of this object to a Stream \a s.
        ///
        /// @param[in] s
        ///     The stream to which to dump the object descripton.
        //--------------------------------------------------------------
        void
        Dump (Stream *s) const;

        //--------------------------------------------------------------
        /// Test if this token is a wildcard token.
        ///
        /// @return
        ///     Returns \b true if this token is a wildcard, \b false
        ///     otherwise.
        //--------------------------------------------------------------
        bool
        IsWildcard() const;

        //--------------------------------------------------------------
        /// Members
        //--------------------------------------------------------------
        TokenType       type;   ///< The type of the token (Mangled::TokenType)
        ConstString value;  ///< The ConstString value associated with this token
    };

    //------------------------------------------------------------------
    /// A collection of tokens.
    ///
    /// This class can be instantiated with a demangled names that can
    /// be used as a query using the
    /// Mangled::TokenList::MatchesQuery(const TokenList&) const
    /// function.
    //------------------------------------------------------------------
    class TokenList
    {
    public:
        //--------------------------------------------------------------
        /// Construct with a demangled name.
        ///
        /// If demangled is valid the token list will parse up the
        /// demangled string it is given, else the object will
        /// initialize an empty token list.
        //--------------------------------------------------------------
        TokenList (const char *demangled = NULL);

        //--------------------------------------------------------------
        /// Destructor
        //--------------------------------------------------------------
        ~TokenList ();

        //--------------------------------------------------------------
        /// Clear the token list.
        //--------------------------------------------------------------
        void
        Clear ();

        //--------------------------------------------------------------
        /// Dump a description of this object to a Stream \a s.
        ///
        /// @param[in] s
        ///     The stream to which to dump the object descripton.
        //--------------------------------------------------------------
        void
        Dump (Stream *s) const;

        //--------------------------------------------------------------
        /// Find a token by Mangled::TokenType.
        ///
        /// Find the first token in the list that has \a token_type as
        /// its type.
        //--------------------------------------------------------------
        const Token*
        Find (TokenType token_type) const;

        //--------------------------------------------------------------
        /// Get a token by index.
        ///
        /// @return
        ///     The token at index \a idx, or NULL if the index is out
        ///     of range.
        //--------------------------------------------------------------
        const Token*
        GetTokenAtIndex (uint32_t idx) const;

        //--------------------------------------------------------------
        /// Given a token list, see if it matches this object's tokens.
        /// \a token_list can contain wild card values to enable powerful
        /// matching. Matching the std::string::erase(*) example that was
        /// tokenized above we could use a token list such as:
        ///
        ///     token           name
        ///     -----------     ----------------------------------------
        ///     eNameSpace      "std"
        ///     eTemplate       "basic_string"
        ///     eTemplateBeg
        ///     eInvalid        "*"
        ///     eTemplateEnd
        ///     eMethodName     "erase"
        ///     eParamsBeg
        ///     eInvalid        "*"
        ///     eParamsEnd
        ///
        /// @return
        ///     Returns \b true if it \a token_list matches this
        ///     object's tokens, \b false otherwise.
        //--------------------------------------------------------------
        bool
        MatchesQuery (const TokenList& token_list) const;

        //--------------------------------------------------------------
        /// Parses \a demangled into tokens.
        ///
        /// This allows complex comparisons to be done on demangled names. Comparisons can
        /// include wildcards at the namespace, method name, template,
        /// and template and parameter type levels.
        ///
        /// Example queries include:
        /// "std::basic_string<*>"  // Find all std::basic_string variants
        /// "std::basic_string<*>::erase(*)"    // Find all std::basic_string::erase variants with any number of parameters
        /// "*::clear()"            // Find all functions with a method name of
        ///                         // "clear" that are in any namespace that
        ///                         // have no parameters
        /// "::printf"      // Find the printf function in the global namespace
        /// "printf"        // Ditto
        /// "foo::*(int)"   // Find all functions in the class or namespace "foo" that take a single integer argument
        ///
        /// @return
        ///     The number of tokens that were decoded, or zero if
        ///     decoding fails.
        //--------------------------------------------------------------
        size_t
        Parse (const char *demangled);

        //--------------------------------------------------------------
        /// Get the number of tokens in the list.
        ///
        /// @return
        ///     The number of tokens in the token list.
        //--------------------------------------------------------------
        size_t
        Size () const;

    protected:
        //--------------------------------------------------------------
        // Member variables.
        //--------------------------------------------------------------
        typedef std::vector<Token> collection; ///< The collection type for a list of Token objects.
        collection m_tokens; ///< The token list.
    private:
        DISALLOW_COPY_AND_ASSIGN (TokenList);
    };

    //----------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Initialize with both mangled and demangled names empty.
    //----------------------------------------------------------------------
    Mangled ();

    //----------------------------------------------------------------------
    /// Construct with name.
    ///
    /// Constructor with an optional string and a boolean indicating if it is
    /// the mangled version.
    ///
    /// @param[in] name
    ///     The name to copy into this object.
    ///
    /// @param[in] is_mangled
    ///     If \b true then \a name is a mangled name, if \b false then
    ///     \a name is demangled.
    //----------------------------------------------------------------------
    explicit
    Mangled (const char *name, bool is_mangled);

    //----------------------------------------------------------------------
    /// Destructor
    ///
    /// Releases its ref counts on the mangled and demangled strings that
    /// live in the global string pool.
    //----------------------------------------------------------------------
    ~Mangled ();

    //----------------------------------------------------------------------
    /// Convert to pointer operator.
    ///
    /// This allows code to check a Mangled object to see if it contains
    /// a valid mangled name using code such as:
    ///
    /// @code
    /// Mangled mangled(...);
    /// if (mangled)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     A pointer to this object if either the mangled or unmangled
    ///     name is set, NULL otherwise.
    //----------------------------------------------------------------------
    operator
    void*() const;

    //----------------------------------------------------------------------
    /// Logical NOT operator.
    ///
    /// This allows code to check a Mangled object to see if it contains
    /// an empty mangled name using code such as:
    ///
    /// @code
    /// Mangled mangled(...);
    /// if (!mangled)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     Returns \b true if the object has an empty mangled and
    ///     unmangled name, \b false otherwise.
    //----------------------------------------------------------------------
    bool
    operator!() const;

    //----------------------------------------------------------------------
    /// Clear the mangled and demangled values.
    //----------------------------------------------------------------------
    void
    Clear ();

    //----------------------------------------------------------------------
    /// Compare the mangled string values
    ///
    /// Compares the Mangled::GetName() string in \a lhs and \a rhs.
    ///
    /// @param[in] lhs
    ///     A const reference to the Left Hand Side object to compare.
    ///
    /// @param[in] rhs
    ///     A const reference to the Right Hand Side object to compare.
    ///
    /// @return
    ///     @li -1 if \a lhs is less than \a rhs
    ///     @li 0 if \a lhs is equal to \a rhs
    ///     @li 1 if \a lhs is greater than \a rhs
    //----------------------------------------------------------------------
    static int
    Compare (const Mangled& lhs, const Mangled& rhs);

    //----------------------------------------------------------------------
    /// Dump a description of this object to a Stream \a s.
    ///
    /// Dump a Mangled object to stream \a s. We don't force our
    /// demangled name to be computed currently (we don't use the accessor).
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //----------------------------------------------------------------------
    void
    Dump (Stream *s) const;

    //----------------------------------------------------------------------
    /// Dump a debug description of this object to a Stream \a s.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //----------------------------------------------------------------------
    void
    DumpDebug (Stream *s) const;

    //----------------------------------------------------------------------
    /// Demangled name get accessor.
    ///
    /// @return
    ///     A const reference to the demangled name string object.
    //----------------------------------------------------------------------
    const ConstString&
    GetDemangledName () const;

    //----------------------------------------------------------------------
    /// Mangled name get accessor.
    ///
    /// @return
    ///     A reference to the mangled name string object.
    //----------------------------------------------------------------------
    ConstString&
    GetMangledName ();

    //----------------------------------------------------------------------
    /// Mangled name get accessor.
    ///
    /// @return
    ///     A const reference to the mangled name string object.
    //----------------------------------------------------------------------
    const ConstString&
    GetMangledName () const;

    //----------------------------------------------------------------------
    /// Best name get accessor.
    ///
    /// @param[in] preference
    ///     Which name would you prefer to get?
    ///
    /// @return
    ///     A const reference to the the preferred name string object if this
    ///     object has a valid name of that kind, else a const reference to the
    ///     other name is returned.
    //----------------------------------------------------------------------
    const ConstString&
    GetName (NamePreference preference = ePreferDemangled) const;

    //----------------------------------------------------------------------
    /// Check if "name" matches either the mangled or demangled name.
    ///
    /// @param[in] name
    ///     A name to match against both strings.
    ///
    /// @return
    ///     \b True if \a name matches either name, \b false otherwise.
    //----------------------------------------------------------------------
    bool
    NameMatches (const ConstString &name) const
    {
        if (m_mangled == name)
            return true;
        return GetDemangledName () == name;
    }
    
    bool
    NameMatches (const RegularExpression& regex) const;

    //----------------------------------------------------------------------
    /// Generate the tokens from the demangled name.
    ///
    /// @param[out] tokens
    ///     A token list that will get filled in with the demangled tokens.
    ///
    /// @return
    ///     The number of tokens that were parsed and stored in \a tokens.
    //----------------------------------------------------------------------
    size_t
    GetTokens (Mangled::TokenList &tokens) const;

    //----------------------------------------------------------------------
    /// Get the memory cost of this object.
    ///
    /// Return the size in bytes that this object takes in memory. This
    /// returns the size in bytes of this object, not any shared string
    /// values it may refer to.
    ///
    /// @return
    ///     The number of bytes that this object occupies in memory.
    ///
    /// @see ConstString::StaticMemorySize ()
    //----------------------------------------------------------------------
    size_t
    MemorySize () const;

    //----------------------------------------------------------------------
    /// Set the string value in this object.
    ///
    /// If \a is_mangled is \b true, then the mangled named is set to \a
    /// name, else the demangled name is set to \a name.
    ///
    /// @param[in] name
    ///     The name to copy into this object.
    ///
    /// @param[in] is_mangled
    ///     If \b true then \a name is a mangled name, if \b false then
    ///     \a name is demangled.
    //----------------------------------------------------------------------
    void
    SetValue (const char *name, bool is_mangled);

private:
    //----------------------------------------------------------------------
    /// Mangled member variables.
    //----------------------------------------------------------------------
            ConstString m_mangled;      ///< The mangled version of the name
    mutable ConstString m_demangled;    ///< Mutable so we can get it on demand with a const version of this object
};


Stream& operator << (Stream& s, const Mangled& obj);
Stream& operator << (Stream& s, const Mangled::TokenList& obj);
Stream& operator << (Stream& s, const Mangled::Token& obj);

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Mangled_h_
