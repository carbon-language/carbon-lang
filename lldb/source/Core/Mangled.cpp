//===-- Mangled.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cxxabi.h>

#include "llvm/ADT/DenseMap.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include <ctype.h>
#include <string.h>

using namespace lldb_private;

#pragma mark Mangled
//----------------------------------------------------------------------
// Default constructor
//----------------------------------------------------------------------
Mangled::Mangled () :
    m_mangled(),
    m_demangled()
{
}

//----------------------------------------------------------------------
// Constructor with an optional string and a boolean indicating if it is
// the mangled version.
//----------------------------------------------------------------------
Mangled::Mangled (const char *s, bool mangled) :
    m_mangled(),
    m_demangled()
{
    if (s && s[0])
    {
        SetValue(s, mangled);
    }
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Mangled::~Mangled ()
{
}

//----------------------------------------------------------------------
// Convert to pointer operator. This allows code to check any Mangled
// objects to see if they contain anything valid using code such as:
//
//  Mangled mangled(...);
//  if (mangled)
//  { ...
//----------------------------------------------------------------------
Mangled::operator void* () const
{
    return (m_mangled) ? const_cast<Mangled*>(this) : NULL;
}

//----------------------------------------------------------------------
// Logical NOT operator. This allows code to check any Mangled
// objects to see if they are invalid using code such as:
//
//  Mangled mangled(...);
//  if (!file_spec)
//  { ...
//----------------------------------------------------------------------
bool
Mangled::operator! () const
{
    return !m_mangled;
}

//----------------------------------------------------------------------
// Clear the mangled and demangled values.
//----------------------------------------------------------------------
void
Mangled::Clear ()
{
    m_mangled.Clear();
    m_demangled.Clear();
}


//----------------------------------------------------------------------
// Compare the the string values.
//----------------------------------------------------------------------
int
Mangled::Compare (const Mangled& a, const Mangled& b)
{
    return ConstString::Compare(a.GetName(ePreferMangled), a.GetName(ePreferMangled));
}



//----------------------------------------------------------------------
// Set the string value in this objects. If "mangled" is true, then
// the mangled named is set with the new value in "s", else the
// demangled name is set.
//----------------------------------------------------------------------
void
Mangled::SetValue (const char *s, bool mangled)
{
    m_mangled.Clear();
    m_demangled.Clear();

    if (s)
    {
        if (mangled)
            m_mangled.SetCString (s);
        else
            m_demangled.SetCString(s);
    }
}

//----------------------------------------------------------------------
// Generate the demangled name on demand using this accessor. Code in
// this class will need to use this accessor if it wishes to decode
// the demangled name. The result is cached and will be kept until a
// new string value is supplied to this object, or until the end of the
// object's lifetime.
//----------------------------------------------------------------------
const ConstString&
Mangled::GetDemangledName () const
{
    // Check to make sure we have a valid mangled name and that we
    // haven't already decoded our mangled name.
    if (m_mangled && !m_demangled)
    {
        // We need to generate and cache the demangled name.
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "Mangled::GetDemangledName (m_mangled = %s)",
                            m_mangled.GetCString());

        // We already know mangled is valid from the above check,
        // lets just make sure it isn't empty...
        const char * mangled = m_mangled.AsCString();
        if (mangled[0])
        {
            // Since demangling can be a costly, and since all names that go 
            // into a ConstString (like our m_mangled and m_demangled members)
            // end up being unique "const char *" values, we can use a DenseMap
            // to speed up our lookup. We do this because often our symbol table
            // and our debug information both have the mangled names which they
            // would each need to demangle. Also, with GCC we end up with the one
            // definition rule where a lot of STL code produces symbols that are
            // in multiple compile units and the mangled names end up being in
            // the same binary multiple times. The performance win isn't huge, 
            // but we showed a 20% improvement on darwin.
            typedef llvm::DenseMap<const char *, const char *> MangledToDemangledMap;
            static MangledToDemangledMap g_mangled_to_demangled;

            // Check our mangled string pointer to demangled string pointer map first
            MangledToDemangledMap::const_iterator pos = g_mangled_to_demangled.find (mangled);
            if (pos != g_mangled_to_demangled.end())
            {
                // We have already demangled this string, we can just use our saved result!
                m_demangled.SetCString(pos->second);
            }
            else
            {
                // We didn't already mangle this name, demangle it and if all goes well
                // add it to our map.
                char *demangled_name = abi::__cxa_demangle (mangled, NULL, NULL, NULL);

                if (demangled_name)
                {
                    m_demangled.SetCString (demangled_name);
                    // Now that the name has been uniqued, add the uniqued C string
                    // pointer from m_mangled as the key to the uniqued C string
                    // pointer in m_demangled.
                    g_mangled_to_demangled.insert (std::make_pair (mangled, m_demangled.GetCString()));
                    free (demangled_name);
                }
                else
                {
                    // Set the demangled string to the empty string to indicate we
                    // tried to parse it once and failed.
                    m_demangled.SetCString("");
                }
            }
        }
    }

    return m_demangled;
}


bool
Mangled::NameMatches (const RegularExpression& regex) const
{
    if (m_mangled && regex.Execute (m_mangled.AsCString()))
        return true;
    
    if (GetDemangledName() && regex.Execute (m_demangled.AsCString()))
        return true;
    return false;
}


//----------------------------------------------------------------------
// Mangled name get accessor
//----------------------------------------------------------------------
ConstString&
Mangled::GetMangledName ()
{
    return m_mangled;
}

//----------------------------------------------------------------------
// Mangled name const get accessor
//----------------------------------------------------------------------
const ConstString&
Mangled::GetMangledName () const
{
    return m_mangled;
}

//----------------------------------------------------------------------
// Get the demangled name if there is one, else return the mangled name.
//----------------------------------------------------------------------
const ConstString&
Mangled::GetName (Mangled::NamePreference preference) const
{
    if (preference == ePreferDemangled)
    {
        // Call the accessor to make sure we get a demangled name in case
        // it hasn't been demangled yet...
        if (GetDemangledName())
            return m_demangled;
        return m_mangled;
    }
    else
    {
        if (m_mangled)
            return m_mangled;
        return GetDemangledName();
    }
}

//----------------------------------------------------------------------
// Generate the tokens from the demangled name.
//
// Returns the number of tokens that were parsed.
//----------------------------------------------------------------------
size_t
Mangled::GetTokens (Mangled::TokenList &tokens) const
{
    tokens.Clear();
    const ConstString& demangled = GetDemangledName();
    if (demangled && !demangled.IsEmpty())
        tokens.Parse(demangled.AsCString());

    return tokens.Size();
}

//----------------------------------------------------------------------
// Dump a Mangled object to stream "s". We don't force our
// demangled name to be computed currently (we don't use the accessor).
//----------------------------------------------------------------------
void
Mangled::Dump (Stream *s) const
{
    if (m_mangled)
    {
        *s << ", mangled = " << m_mangled;
    }
    if (m_demangled)
    {
        const char * demangled = m_demangled.AsCString();
        s->Printf(", demangled = %s", demangled[0] ? demangled : "<error>");
    }
}

//----------------------------------------------------------------------
// Dumps a debug version of this string with extra object and state
// information to stream "s".
//----------------------------------------------------------------------
void
Mangled::DumpDebug (Stream *s) const
{
    s->Printf("%*p: Mangled mangled = ", (int)sizeof(void*) * 2, this);
    m_mangled.DumpDebug(s);
    s->Printf(", demangled = ");
    m_demangled.DumpDebug(s);
}

//----------------------------------------------------------------------
// Return the size in byte that this object takes in memory. The size
// includes the size of the objects it owns, and not the strings that
// it references because they are shared strings.
//----------------------------------------------------------------------
size_t
Mangled::MemorySize () const
{
    return m_mangled.MemorySize() + m_demangled.MemorySize();
}

//----------------------------------------------------------------------
// Dump OBJ to the supplied stream S.
//----------------------------------------------------------------------
Stream&
operator << (Stream& s, const Mangled& obj)
{
    if (obj.GetMangledName())
        s << "mangled = '" << obj.GetMangledName() << "'";

    const ConstString& demangled = obj.GetDemangledName();
    if (demangled)
        s << ", demangled = '" << demangled << '\'';
    else
        s << ", demangled = <error>";
    return s;
}




#pragma mark Mangled::Token

//--------------------------------------------------------------
// Default constructor
//--------------------------------------------------------------
Mangled::Token::Token () :
    type(eInvalid),
    value()
{
}

//--------------------------------------------------------------
// Equal to operator
//--------------------------------------------------------------
bool
Mangled::Token::operator== (const Token& rhs) const
{
    return type == rhs.type && value == rhs.value;
}

//--------------------------------------------------------------
// Dump the token to a stream "s"
//--------------------------------------------------------------
void
Mangled::Token::Dump (Stream *s) const
{
    switch (type)
    {
    case eInvalid:      s->PutCString("invalid    "); break;
    case eNameSpace:    s->PutCString("namespace  "); break;
    case eMethodName:   s->PutCString("method     "); break;
    case eType:         s->PutCString("type       "); break;
    case eTemplate:     s->PutCString("template   "); break;
    case eTemplateBeg:  s->PutCString("template < "); break;
    case eTemplateEnd:  s->PutCString("template > "); break;
    case eParamsBeg:    s->PutCString("params   ( "); break;
    case eParamsEnd:    s->PutCString("params   ) "); break;
    case eQualifier:    s->PutCString("qualifier  "); break;
    case eError:        s->PutCString("ERROR      "); break;
    default:
        s->Printf("type = %i", type);
        break;
    }
    value.DumpDebug(s);
}

//--------------------------------------------------------------
// Returns true if this token is a wildcard
//--------------------------------------------------------------
bool
Mangled::Token::IsWildcard () const
{
    static ConstString g_wildcard_str("*");
    return value == g_wildcard_str;
}


//----------------------------------------------------------------------
// Dump "obj" to the supplied stream "s"
//----------------------------------------------------------------------
Stream&
lldb_private::operator << (Stream& s, const Mangled::Token& obj)
{
    obj.Dump(&s);
    return s;
}


#pragma mark Mangled::TokenList
//----------------------------------------------------------------------
// Mangled::TokenList
//----------------------------------------------------------------------

//--------------------------------------------------------------
// Default constructor. If demangled is non-NULL and not-empty
// the token list will parse up the demangled string it is
// given, else the object will initialize an empty token list.
//--------------------------------------------------------------
Mangled::TokenList::TokenList (const char *demangled) :
    m_tokens()
{
    if (demangled && demangled[0])
    {
        Parse(demangled);
    }
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Mangled::TokenList::~TokenList ()
{
}

//----------------------------------------------------------------------
// Parses "demangled" into tokens. This allows complex
// comparisons to be done. Comparisons can include wildcards at
// the namespace, method name, template, and template and
// parameter type levels.
//
// Example queries include:
// "std::basic_string<*>"   // Find all std::basic_string variants
// "std::basic_string<*>::erase(*)" // Find all std::basic_string::erase variants with any number of parameters
// "*::clear()"             // Find all functions with a method name of
//                          // "clear" that are in any namespace that
//                          // have no parameters
// "::printf"               // Find the printf function in the global namespace
// "printf"                 // Ditto
// "foo::*(int)"            // Find all functions in the class or namespace "foo" that take a single integer argument
//
// Returns the number of tokens that were decoded, or zero when
// we fail.
//----------------------------------------------------------------------
size_t
Mangled::TokenList::Parse (const char *s)
{
    m_tokens.clear();

    Token token;
    token.type = eNameSpace;

    TokenType max_type = eInvalid;
    const char *p = s;
    size_t span = 0;
    size_t sep_size = 0;

    while (*p != '\0')
    {
        p = p + span + sep_size;
        while (isspace(*p))
            ++p;

        if (*p == '\0')
            break;

        span = strcspn(p, ":<>(),");
        sep_size = 1;
        token.type = eInvalid;
        switch (p[span])
        {
        case '\0':
            break;

        case ':':
            if (p[span+1] == ':')
            {
                sep_size = 2;
                if (span > 0)
                {
                    token.type = eNameSpace;
                    token.value.SetCStringWithLength (p, span);
                    m_tokens.push_back(token);
                }
                else
                    continue;
            }
            break;

        case '(':
            if (span > 0)
            {
                token.type = eMethodName;
                token.value.SetCStringWithLength (p, span);
                m_tokens.push_back(token);
            }

            token.type = eParamsBeg;
            token.value.Clear();
            m_tokens.push_back(token);
            break;

        case ',':
            if (span > 0)
            {
                token.type = eType;
                token.value.SetCStringWithLength (p, span);
                m_tokens.push_back(token);
            }
            else
            {
                continue;
            }
            break;

        case ')':
            if (span > 0)
            {
                token.type = eType;
                token.value.SetCStringWithLength (p, span);
                m_tokens.push_back(token);
            }

            token.type = eParamsEnd;
            token.value.Clear();
            m_tokens.push_back(token);
            break;

        case '<':
            if (span > 0)
            {
                token.type = eTemplate;
                token.value.SetCStringWithLength (p, span);
                m_tokens.push_back(token);
            }

            token.type = eTemplateBeg;
            token.value.Clear();
            m_tokens.push_back(token);
            break;

        case '>':
            if (span > 0)
            {
                token.type = eType;
                token.value.SetCStringWithLength (p, span);
                m_tokens.push_back(token);
            }

            token.type = eTemplateEnd;
            token.value.Clear();
            m_tokens.push_back(token);
            break;
        }

        if (max_type < token.type)
            max_type = token.type;

        if (token.type == eInvalid)
        {
            if (max_type >= eParamsEnd)
            {
                token.type = eQualifier;
                token.value.SetCString(p);
                m_tokens.push_back(token);
            }
            else if (max_type >= eParamsBeg)
            {
                token.type = eType;
                token.value.SetCString(p);
                m_tokens.push_back(token);
            }
            else
            {
                token.type = eMethodName;
                token.value.SetCString(p);
                m_tokens.push_back(token);
            }
            break;
        }
    }
    return m_tokens.size();
}


//----------------------------------------------------------------------
// Clear the token list.
//----------------------------------------------------------------------
void
Mangled::TokenList::Clear ()
{
    m_tokens.clear();
}

//----------------------------------------------------------------------
// Dump the token list to the stream "s"
//----------------------------------------------------------------------
void
Mangled::TokenList::Dump (Stream *s) const
{
    collection::const_iterator pos;
    collection::const_iterator beg = m_tokens.begin();
    collection::const_iterator end = m_tokens.end();
    for (pos = beg; pos != end; ++pos)
    {
        s->Indent("token[");
        *s << (uint32_t)std::distance(beg, pos) << "] = " << *pos << "\n";
    }
}

//----------------------------------------------------------------------
// Find the first token in the list that has "token_type" as its
// type
//----------------------------------------------------------------------
const Mangled::Token *
Mangled::TokenList::Find (TokenType token_type) const
{
    collection::const_iterator pos;
    collection::const_iterator beg = m_tokens.begin();
    collection::const_iterator end = m_tokens.end();
    for (pos = beg; pos != end; ++pos)
    {
        if (pos->type == token_type)
            return &(*pos);
    }
    return NULL;
}

//----------------------------------------------------------------------
// Return the token at index "idx", or NULL if the index is
// out of range.
//----------------------------------------------------------------------
const Mangled::Token *
Mangled::TokenList::GetTokenAtIndex (uint32_t idx) const
{
    if (idx < m_tokens.size())
        return &m_tokens[idx];
    return NULL;
}


//----------------------------------------------------------------------
// Given a token list, see if it matches this object's tokens.
// "token_list" can contain wild card values to enable powerful
// matching. Matching the std::string::erase(*) example that was
// tokenized above we could use a token list such as:
//
//      token           name
//      -----------     ----------------------------------------
//      eNameSpace      "std"
//      eTemplate       "basic_string"
//      eTemplateBeg
//      eInvalid        "*"
//      eTemplateEnd
//      eMethodName     "erase"
//      eParamsBeg
//      eInvalid        "*"
//      eParamsEnd
//
// Returns true if it "token_list" matches this object's tokens,
// false otherwise.
//----------------------------------------------------------------------
bool
Mangled::TokenList::MatchesQuery (const Mangled::TokenList &match) const
{
    size_t match_count = 0;
    collection::const_iterator pos;
    collection::const_iterator pos_end = m_tokens.end();

    collection::const_iterator match_pos;
    collection::const_iterator match_pos_end = match.m_tokens.end();
    collection::const_iterator match_wildcard_pos = match_pos_end;
    collection::const_iterator match_next_pos = match_pos_end;

    size_t template_scope_depth = 0;

    for (pos = m_tokens.begin(), match_pos = match.m_tokens.begin();
         pos != pos_end && match_pos != match_pos_end;
         ++match_pos)
    {
        match_next_pos = match_pos + 1;
        // Is this a wildcard?
        if (match_pos->IsWildcard())
        {
            if (match_wildcard_pos != match_pos_end)
                return false;   // Can't have two wildcards in effect at once.

            match_wildcard_pos = match_pos;
            // Are we at the end of the MATCH token list?
            if (match_next_pos == match_pos_end)
            {
                // There is nothing more to match, return if we have any matches so far...
                return match_count > 0;
            }
        }

        if (match_pos->type == eInvalid || match_pos->type == eError)
        {
            return false;
        }
        else
        {
            if (match_pos->type == eTemplateBeg)
            {
                ++template_scope_depth;
            }
            else if (match_pos->type == eTemplateEnd)
            {
                assert(template_scope_depth > 0);
                --template_scope_depth;
            }

            // Do we have a wildcard going right now?
            if (match_wildcard_pos == match_pos_end)
            {
                // No wildcard matching right now, just check and see if things match
                if (*pos == *match_pos)
                    ++match_count;
                else
                    return false;
            }
            else
            {
                // We have a wildcard match going

                // For template types we need to make sure to match the template depths...
                const size_t start_wildcard_template_scope_depth = template_scope_depth;
                size_t curr_wildcard_template_scope_depth = template_scope_depth;
                while (pos != pos_end)
                {
                    if (match_wildcard_pos->type == eNameSpace && pos->type == eParamsBeg)
                        return false;

                    if (start_wildcard_template_scope_depth == curr_wildcard_template_scope_depth)
                    {
                        if (*pos == *match_next_pos)
                        {
                            ++match_count;
                            match_pos = match_next_pos;
                            match_wildcard_pos = match_pos_end;
                            break;
                        }
                    }
                    if (pos->type == eTemplateBeg)
                        ++curr_wildcard_template_scope_depth;
                    else if (pos->type == eTemplateEnd)
                        --curr_wildcard_template_scope_depth;


                    ++pos;
                }
            }
        }

        if (pos != pos_end)
            ++pos;
    }
    if (match_pos != match_pos_end)
        return false;

    return match_count > 0;
}


//----------------------------------------------------------------------
// Return the number of tokens in the token collection
//----------------------------------------------------------------------
size_t
Mangled::TokenList::Size () const
{
    return m_tokens.size();
}


//----------------------------------------------------------------------
// Stream out the tokens
//----------------------------------------------------------------------
Stream&
lldb_private::operator << (Stream& s, const Mangled::TokenList& obj)
{
    obj.Dump(&s);
    return s;
}
