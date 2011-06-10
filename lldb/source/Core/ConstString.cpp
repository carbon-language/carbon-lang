//===-- ConstString.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Stream.h"
#include "lldb/Host/Mutex.h"
#include "llvm/ADT/StringMap.h"

using namespace lldb_private;


//----------------------------------------------------------------------
// The global string pool is implemented as a hash_map that maps
// std::string objects to a uint32_t reference count.
//
// In debug builds the value that is stored in the ConstString objects is
// a C string that is owned by one of the std::string objects in the
// hash map. This was done for visibility purposes when debugging as
// gcc was often generating insufficient debug info for the
// iterator objects.
//
// In release builds, the value that is stored in the ConstString objects
// is the iterator into the ConstString::HashMap. This is much faster when
// it comes to modifying the reference count, and removing strings from
// the pool.
//----------------------------------------------------------------------
class Pool
{
public:
    typedef const char * StringPoolValueType;
    typedef llvm::StringMap<StringPoolValueType, llvm::BumpPtrAllocator> StringPool;
    typedef llvm::StringMapEntry<StringPoolValueType> StringPoolEntryType;
    
    //------------------------------------------------------------------
    // Default constructor
    //
    // Initialize the member variables and create the empty string.
    //------------------------------------------------------------------
    Pool () :
        m_mutex (Mutex::eMutexTypeRecursive),
        m_string_map ()
    {
    }

    //------------------------------------------------------------------
    // Destructor
    //------------------------------------------------------------------
    ~Pool ()
    {
    }


    static StringPoolEntryType &
    GetStringMapEntryFromKeyData (const char *keyData)
    {
        char *ptr = const_cast<char*>(keyData) - sizeof (StringPoolEntryType);
        return *reinterpret_cast<StringPoolEntryType*>(ptr);
    }

    size_t
    GetConstCStringLength (const char *ccstr) const
    {
        if (ccstr)
        {
            const StringPoolEntryType&entry = GetStringMapEntryFromKeyData (ccstr);
            return entry.getKey().size();
        }
        return 0;
    }

    StringPoolValueType
    GetMangledCounterpart (const char *ccstr) const
    {
        if (ccstr)
            return GetStringMapEntryFromKeyData (ccstr).getValue();
        return 0;
    }

    bool
    SetMangledCounterparts (const char *key_ccstr, const char *value_ccstr)
    {
        if (key_ccstr && value_ccstr)
        {
            GetStringMapEntryFromKeyData (key_ccstr).setValue(value_ccstr);
            GetStringMapEntryFromKeyData (value_ccstr).setValue(key_ccstr);
            return true;
        }
        return false;
    }

    const char *
    GetConstCString (const char *cstr)
    {
        if (cstr)
            return GetConstCStringWithLength (cstr, strlen (cstr));
        return NULL;
    }

    const char *
    GetConstCStringWithLength (const char *cstr, int cstr_len)
    {
        if (cstr)
        {
            Mutex::Locker locker (m_mutex);
            llvm::StringRef string_ref (cstr, cstr_len);
            StringPoolEntryType& entry = m_string_map.GetOrCreateValue (string_ref, (StringPoolValueType)NULL);
            return entry.getKeyData();
        }
        return NULL;
    }

    const char *
    GetConstCStringAndSetMangledCounterPart (const char *demangled_cstr, const char *mangled_ccstr)
    {
        if (demangled_cstr)
        {
            Mutex::Locker locker (m_mutex);
            // Make string pool entry with the mangled counterpart already set
            StringPoolEntryType& entry = m_string_map.GetOrCreateValue (llvm::StringRef (demangled_cstr), mangled_ccstr);

            // Extract the const version of the demangled_cstr
            const char *demangled_ccstr = entry.getKeyData();
            // Now assign the demangled const string as the counterpart of the
            // mangled const string...
            GetStringMapEntryFromKeyData (mangled_ccstr).setValue(demangled_ccstr);
            // Return the constant demangled C string
            return demangled_ccstr;
        }
        return NULL;
    }

    const char *
    GetConstTrimmedCStringWithLength (const char *cstr, int cstr_len)
    {
        if (cstr)
        {
            int trimmed_len = std::min<int> (strlen (cstr), cstr_len);
            return GetConstCStringWithLength (cstr, trimmed_len);
        }
        return NULL;
    }

    //------------------------------------------------------------------
    // Return the size in bytes that this object and any items in its
    // collection of uniqued strings + reference count values takes in
    // memory.
    //------------------------------------------------------------------
    size_t
    MemorySize() const
    {
        Mutex::Locker locker (m_mutex);
        size_t mem_size = sizeof(Pool);
        const_iterator end = m_string_map.end();
        for (const_iterator pos = m_string_map.begin(); pos != end; ++pos)
        {
            mem_size += sizeof(StringPoolEntryType) + pos->getKey().size();
        }
        return mem_size;
    }

protected:
    //------------------------------------------------------------------
    // Typedefs
    //------------------------------------------------------------------
    typedef StringPool::iterator iterator;
    typedef StringPool::const_iterator const_iterator;

    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    mutable Mutex m_mutex;
    StringPool m_string_map;
};

//----------------------------------------------------------------------
// Frameworks and dylibs aren't supposed to have global C++
// initializers so we hide the string pool in a static function so
// that it will get initialized on the first call to this static
// function.
//----------------------------------------------------------------------
static Pool &
StringPool()
{
    static Pool string_pool;
    return string_pool;
}

//----------------------------------------------------------------------
// Default constructor
//
// Initializes the string to an empty string.
//----------------------------------------------------------------------
ConstString::ConstString () :
    m_string (NULL)
{
}

//----------------------------------------------------------------------
// Copy constructor
//
// Copies the string value in "rhs" and retains an extra reference
// to the string value in the string pool.
//----------------------------------------------------------------------
ConstString::ConstString (const ConstString& rhs) :
    m_string (rhs.m_string)
{
}

//----------------------------------------------------------------------
// Construct with C String value
//
// Constructs this object with a C string by looking to see if the
// C string already exists in the global string pool. If it does
// exist, it retains an extra reference to the string in the string
// pool. If it doesn't exist, it is added to the string pool with
// a reference count of 1.
//----------------------------------------------------------------------
ConstString::ConstString (const char *cstr) :
    m_string (StringPool().GetConstCString (cstr))
{
}

//----------------------------------------------------------------------
// Construct with C String value with max length
//
// Constructs this object with a C string with a length. If
// the length of the string is greather than "cstr_len", the
// string length will be truncated. This allows substrings to be
// created without the need to NULL terminate the string as it
// is passed into this function.
//
// If the C string already exists in the global string pool, it
// retains an extra reference to the string in the string
// pool. If it doesn't exist, it is added to the string pool with
// a reference count of 1.
//----------------------------------------------------------------------
ConstString::ConstString (const char *cstr, size_t cstr_len) :
    m_string (StringPool().GetConstCStringWithLength (cstr, cstr_len))
{
}

//----------------------------------------------------------------------
// Destructor
//
// Decrements the reference count on the contained string, and if
// the resulting reference count is zero, then the string is removed
// from the string pool. If the reference count is still greater
// than zero, the string will remain in the string pool
//----------------------------------------------------------------------
ConstString::~ConstString ()
{
}

bool
ConstString::operator < (const ConstString& rhs) const
{
    if (m_string == rhs.m_string)
        return false;

    llvm::StringRef lhs_string_ref (m_string, StringPool().GetConstCStringLength (m_string));
    llvm::StringRef rhs_string_ref (rhs.m_string, StringPool().GetConstCStringLength (rhs.m_string));

    // If both have valid C strings, then return the comparison
    if (lhs_string_ref.data() && rhs_string_ref.data())
        return lhs_string_ref < rhs_string_ref;

    // Else one of them was NULL, so if LHS is NULL then it is less than
    return lhs_string_ref.data() == NULL;
}

//----------------------------------------------------------------------
// Stream the string value "str" to the stream "s"
//----------------------------------------------------------------------
Stream&
lldb_private::operator << (Stream& s, const ConstString& str)
{
    const char *cstr = str.GetCString();
    if (cstr)
        s << cstr;

    return s;
}

size_t
ConstString::GetLength () const
{
    return StringPool().GetConstCStringLength (m_string);
}

//----------------------------------------------------------------------
// Compare two string objects.
//
// Returns:
//  -1 if a < b
//   0 if a == b
//   1 if a > b
//----------------------------------------------------------------------
int
ConstString::Compare (const ConstString& lhs, const ConstString& rhs)
{
    // If the iterators are the same, this is the same string
    register const char *lhs_cstr = lhs.m_string;
    register const char *rhs_cstr = rhs.m_string;
    if (lhs_cstr == rhs_cstr)
        return 0;
    if (lhs_cstr && rhs_cstr)
    {
        llvm::StringRef lhs_string_ref (lhs_cstr, StringPool().GetConstCStringLength (lhs_cstr));
        llvm::StringRef rhs_string_ref (rhs_cstr, StringPool().GetConstCStringLength (rhs_cstr));
        return lhs_string_ref.compare(rhs_string_ref);
    }

    if (lhs_cstr)
        return +1;  // LHS isn't NULL but RHS is
    else
        return -1;  // LHS is NULL but RHS isn't
}

//----------------------------------------------------------------------
// Dump the string value to the stream "s". If the contained string
// is empty, print "fail_value" to the stream instead. If
// "fail_value" is NULL, then nothing will be dumped to the
// stream.
//----------------------------------------------------------------------
void
ConstString::Dump(Stream *s, const char *fail_value) const
{
    const char *cstr = AsCString (fail_value);
    if (cstr)
        s->PutCString (cstr);
}

//----------------------------------------------------------------------
// Dump extra debug information to the stream "s".
//----------------------------------------------------------------------
void
ConstString::DumpDebug(Stream *s) const
{
    const char *cstr = GetCString ();
    size_t cstr_len = GetLength();
    // Only print the parens if we have a non-NULL string
    const char *parens = cstr ? "\"" : "";
    s->Printf("%*p: ConstString, string = %s%s%s, length = %zu", (int)sizeof(void*) * 2, this, parens, cstr, parens, cstr_len);
}

//----------------------------------------------------------------------
// Set the string value in the object by uniquing the "cstr" string
// value in our global string pool.
//
// If the C string already exists in the global string pool, it
// retains an extra reference to the string in the string
// pool. If it doesn't exist, it is added to the string pool with
// a reference count of 1.
//----------------------------------------------------------------------
void
ConstString::SetCString (const char *cstr)
{
    m_string = StringPool().GetConstCString (cstr);
}

void
ConstString::SetCStringWithMangledCounterpart (const char *demangled, const ConstString &mangled)
{
    m_string = StringPool().GetConstCStringAndSetMangledCounterPart (demangled, mangled.m_string);
}

bool
ConstString::GetMangledCounterpart (ConstString &counterpart) const
{
    counterpart.m_string = StringPool().GetMangledCounterpart(m_string);
    return counterpart;
}

//----------------------------------------------------------------------
// Set the string value in the object by uniquing "cstr_len" bytes
// starting at the "cstr" string value in our global string pool.
// If trim is true, then "cstr_len" indicates a maximum length of
// the CString and if the actual length of the string is less, then
// it will be trimmed. If trim is false, then this allows strings
// with NULL characters ('\0') to be added to the string pool.
//
// If the C string already exists in the global string pool, it
// retains an extra reference to the string in the string
// pool. If it doesn't exist, it is added to the string pool with
// a reference count of 1.
//----------------------------------------------------------------------
void
ConstString::SetCStringWithLength (const char *cstr, size_t cstr_len)
{
    m_string = StringPool().GetConstCStringWithLength(cstr, cstr_len);
}

void
ConstString::SetTrimmedCStringWithLength (const char *cstr, size_t cstr_len)
{
    m_string = StringPool().GetConstTrimmedCStringWithLength (cstr, cstr_len);
}

//----------------------------------------------------------------------
// Return the size in bytes that this object takes in memory. The
// resulting size will not include any of the C string values from
// the global string pool (see StaticMemorySize ()).
//----------------------------------------------------------------------
size_t
ConstString::MemorySize() const
{
    return sizeof(ConstString);
}

//----------------------------------------------------------------------
// Reports the the size in bytes of all shared C string values,
// containers and reference count values as a byte size for the
// entire string pool.
//----------------------------------------------------------------------
size_t
ConstString::StaticMemorySize()
{
    // Get the size of the static string pool
    return StringPool().MemorySize();
}
