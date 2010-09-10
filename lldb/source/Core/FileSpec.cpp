//===-- FileSpec.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <fcntl.h>
#include <libgen.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pwd.h>

#include <fstream>

#include "lldb/Core/FileSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Core/Stream.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

static bool
GetFileStats (const FileSpec *file_spec, struct stat *stats_ptr)
{
    char resolved_path[PATH_MAX];
    if (file_spec->GetPath(&resolved_path[0], sizeof(resolved_path)))
        return ::stat (resolved_path, stats_ptr) == 0;
    return false;
}

static const char*
GetCachedGlobTildeSlash()
{
    static std::string g_tilde;
    if (g_tilde.empty())
    {
        struct passwd *user_entry;
        user_entry = getpwuid(geteuid());
        if (user_entry != NULL)
            g_tilde = user_entry->pw_dir;

        if (g_tilde.empty())
            return NULL;
    }
    return g_tilde.c_str();
}

// Resolves the username part of a path of the form ~user/other/directories, and
// writes the result into dst_path.
// Returns 0 if there WAS a ~ in the path but the username couldn't be resolved.
// Otherwise returns the number of characters copied into dst_path.  If the return
// is >= dst_len, then the resolved path is too long...
size_t
FileSpec::ResolveUsername (const char *src_path, char *dst_path, size_t dst_len)
{
    char user_home[PATH_MAX];
    const char *user_name;
    
    if (src_path == NULL || src_path[0] == '\0')
        return 0;
    
    // If there's no ~, then just copy src_path straight to dst_path (they may be the same string...)
    if (src_path[0] != '~')
    {
        size_t len = strlen (src_path);
        if (len >= dst_len)
        {
            ::bcopy (src_path, dst_path, dst_len - 1);
            dst_path[dst_len] = '\0';
        }
        else
            ::bcopy (src_path, dst_path, len + 1);
        
        return len;
    }
    
    const char *first_slash = ::strchr (src_path, '/');
    char remainder[PATH_MAX];
    
    if (first_slash == NULL)
    {
        // The whole name is the username (minus the ~):
        user_name = src_path + 1;
        remainder[0] = '\0';
    }
    else
    {
        int user_name_len = first_slash - src_path - 1;
        ::memcpy (user_home, src_path + 1, user_name_len);
        user_home[user_name_len] = '\0';
        user_name = user_home;
        
        ::strcpy (remainder, first_slash);
    }
    
    if (user_name == NULL)
        return 0;
    // User name of "" means the current user...
    
    struct passwd *user_entry;
    const char *home_dir = NULL;
    
    if (user_name[0] == '\0')
    {
        home_dir = GetCachedGlobTildeSlash();
    }
    else
    {
        user_entry = ::getpwnam (user_name);
        if (user_entry != NULL)
            home_dir = user_entry->pw_dir;
    }
    
    if (home_dir == NULL)
        return 0;
    else 
        return ::snprintf (dst_path, dst_len, "%s%s", home_dir, remainder);
}

size_t
FileSpec::Resolve (const char *src_path, char *dst_path, size_t dst_len)
{
    if (src_path == NULL || src_path[0] == '\0')
        return 0;

    // Glob if needed for ~/, otherwise copy in case src_path is same as dst_path...
    char unglobbed_path[PATH_MAX];
    if (src_path[0] == '~')
    {
        size_t return_count = ResolveUsername(src_path, unglobbed_path, sizeof(unglobbed_path));
        
        // If we couldn't find the user referred to, or the resultant path was too long,
        // then just copy over the src_path.
        if (return_count == 0 || return_count >= sizeof(unglobbed_path)) 
            ::snprintf (unglobbed_path, sizeof(unglobbed_path), "%s", src_path);
    }
    else
        ::snprintf(unglobbed_path, sizeof(unglobbed_path), "%s", src_path);

    // Now resolve the path if needed
    char resolved_path[PATH_MAX];
    if (::realpath (unglobbed_path, resolved_path))
    {
        // Success, copy the resolved path
        return ::snprintf(dst_path, dst_len, "%s", resolved_path);
    }
    else
    {
        // Failed, just copy the unglobbed path
        return ::snprintf(dst_path, dst_len, "%s", unglobbed_path);
    }
}

FileSpec::FileSpec() :
    m_directory(),
    m_filename()
{
}

//------------------------------------------------------------------
// Default constructor that can take an optional full path to a
// file on disk.
//------------------------------------------------------------------
FileSpec::FileSpec(const char *pathname) :
    m_directory(),
    m_filename()
{
    if (pathname && pathname[0])
        SetFile(pathname);
}

//------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------
FileSpec::FileSpec(const FileSpec& rhs) :
    m_directory (rhs.m_directory),
    m_filename (rhs.m_filename)
{
}

//------------------------------------------------------------------
// Copy constructor
//------------------------------------------------------------------
FileSpec::FileSpec(const FileSpec* rhs) :
    m_directory(),
    m_filename()
{
    if (rhs)
        *this = *rhs;
}

//------------------------------------------------------------------
// Virtual destrcuctor in case anyone inherits from this class.
//------------------------------------------------------------------
FileSpec::~FileSpec()
{
}

//------------------------------------------------------------------
// Assignment operator.
//------------------------------------------------------------------
const FileSpec&
FileSpec::operator= (const FileSpec& rhs)
{
    if (this != &rhs)
    {
        m_directory = rhs.m_directory;
        m_filename = rhs.m_filename;
    }
    return *this;
}


//------------------------------------------------------------------
// Update the contents of this object with a new path. The path will
// be split up into a directory and filename and stored as uniqued
// string values for quick comparison and efficient memory usage.
//------------------------------------------------------------------
void
FileSpec::SetFile(const char *pathname)
{
    m_filename.Clear();
    m_directory.Clear();
    if (pathname == NULL || pathname[0] == '\0')
        return;

    char resolved_path[PATH_MAX];

    if (FileSpec::Resolve (pathname, resolved_path, sizeof(resolved_path)) < sizeof(resolved_path) - 1)
    {
        char *filename = ::basename (resolved_path);
        if (filename)
        {
            m_filename.SetCString (filename);
            // Truncate the basename off the end of the resolved path

            // Only attempt to get the dirname if it looks like we have a path
            if (strchr(resolved_path, '/'))
            {
                char *directory = ::dirname (resolved_path);

                // Make sure we didn't get our directory resolved to "." without having
                // specified
                if (directory)
                    m_directory.SetCString(directory);
                else
                {
                    char *last_resolved_path_slash = strrchr(resolved_path, '/');
                    if (last_resolved_path_slash)
                    {
                        *last_resolved_path_slash = '\0';
                        m_directory.SetCString(resolved_path);
                    }
                }
            }
        }
        else
            m_directory.SetCString(resolved_path);
    }
}

//----------------------------------------------------------------------
// Convert to pointer operator. This allows code to check any FileSpec
// objects to see if they contain anything valid using code such as:
//
//  if (file_spec)
//  {}
//----------------------------------------------------------------------
FileSpec::operator
void*() const
{
    return (m_directory || m_filename) ? const_cast<FileSpec*>(this) : NULL;
}

//----------------------------------------------------------------------
// Logical NOT operator. This allows code to check any FileSpec
// objects to see if they are invalid using code such as:
//
//  if (!file_spec)
//  {}
//----------------------------------------------------------------------
bool
FileSpec::operator!() const
{
    return !m_directory && !m_filename;
}

//------------------------------------------------------------------
// Equal to operator
//------------------------------------------------------------------
bool
FileSpec::operator== (const FileSpec& rhs) const
{
    return m_directory == rhs.m_directory && m_filename == rhs.m_filename;
}

//------------------------------------------------------------------
// Not equal to operator
//------------------------------------------------------------------
bool
FileSpec::operator!= (const FileSpec& rhs) const
{
    return m_filename != rhs.m_filename || m_directory != rhs.m_directory;
}

//------------------------------------------------------------------
// Less than operator
//------------------------------------------------------------------
bool
FileSpec::operator< (const FileSpec& rhs) const
{
    return FileSpec::Compare(*this, rhs, true) < 0;
}

//------------------------------------------------------------------
// Dump a FileSpec object to a stream
//------------------------------------------------------------------
Stream&
lldb_private::operator << (Stream &s, const FileSpec& f)
{
    f.Dump(&s);
    return s;
}

//------------------------------------------------------------------
// Clear this object by releasing both the directory and filename
// string values and making them both the empty string.
//------------------------------------------------------------------
void
FileSpec::Clear()
{
    m_directory.Clear();
    m_filename.Clear();
}

//------------------------------------------------------------------
// Compare two FileSpec objects. If "full" is true, then both
// the directory and the filename must match. If "full" is false,
// then the directory names for "a" and "b" are only compared if
// they are both non-empty. This allows a FileSpec object to only
// contain a filename and it can match FileSpec objects that have
// matching filenames with different paths.
//
// Return -1 if the "a" is less than "b", 0 if "a" is equal to "b"
// and "1" if "a" is greater than "b".
//------------------------------------------------------------------
int
FileSpec::Compare(const FileSpec& a, const FileSpec& b, bool full)
{
    int result = 0;

    // If full is true, then we must compare both the directory and filename.

    // If full is false, then if either directory is empty, then we match on
    // the basename only, and if both directories have valid values, we still
    // do a full compare. This allows for matching when we just have a filename
    // in one of the FileSpec objects.

    if (full || (a.m_directory && b.m_directory))
    {
        result = ConstString::Compare(a.m_directory, b.m_directory);
        if (result)
            return result;
    }
    return ConstString::Compare (a.m_filename, b.m_filename);
}

bool
FileSpec::Equal (const FileSpec& a, const FileSpec& b, bool full)
{
    if (full)
        return a == b;
    else
        return a.m_filename == b.m_filename;
}



//------------------------------------------------------------------
// Dump the object to the supplied stream. If the object contains
// a valid directory name, it will be displayed followed by a
// directory delimiter, and the filename.
//------------------------------------------------------------------
void
FileSpec::Dump(Stream *s) const
{
    if (m_filename)
        m_directory.Dump(s, "");    // Provide a default for m_directory when we dump it in case it is invalid

    if (m_directory)
    {
        // If dirname was valid, then we need to print a slash between
        // the directory and the filename
        s->PutChar('/');
    }
    m_filename.Dump(s);
}

//------------------------------------------------------------------
// Returns true if the file exists.
//------------------------------------------------------------------
bool
FileSpec::Exists () const
{
    struct stat file_stats;
    return GetFileStats (this, &file_stats);
}

bool
FileSpec::ResolveExecutableLocation ()
{
    return Host::ResolveExecutableLocation (m_directory, m_filename);
}

uint64_t
FileSpec::GetByteSize() const
{
    struct stat file_stats;
    if (GetFileStats (this, &file_stats))
        return file_stats.st_size;
    return 0;
}

FileSpec::FileType
FileSpec::GetFileType () const
{
    struct stat file_stats;
    if (GetFileStats (this, &file_stats))
    {
        mode_t file_type = file_stats.st_mode & S_IFMT;
        switch (file_type)
        {
        case S_IFDIR:   return eFileTypeDirectory;
        case S_IFIFO:   return eFileTypePipe;
        case S_IFREG:   return eFileTypeRegular;
        case S_IFSOCK:  return eFileTypeSocket;
        case S_IFLNK:   return eFileTypeSymbolicLink;
        default:
            break;
        }
        return eFileTypeUknown;
    }
    return eFileTypeInvalid;
}

TimeValue
FileSpec::GetModificationTime () const
{
    TimeValue mod_time;
    struct stat file_stats;
    if (GetFileStats (this, &file_stats))
        mod_time.OffsetWithSeconds(file_stats.st_mtime);
    return mod_time;
}

//------------------------------------------------------------------
// Directory string get accessor.
//------------------------------------------------------------------
ConstString &
FileSpec::GetDirectory()
{
    return m_directory;
}

//------------------------------------------------------------------
// Directory string const get accessor.
//------------------------------------------------------------------
const ConstString &
FileSpec::GetDirectory() const
{
    return m_directory;
}

//------------------------------------------------------------------
// Filename string get accessor.
//------------------------------------------------------------------
ConstString &
FileSpec::GetFilename()
{
    return m_filename;
}

//------------------------------------------------------------------
// Filename string const get accessor.
//------------------------------------------------------------------
const ConstString &
FileSpec::GetFilename() const
{
    return m_filename;
}

//------------------------------------------------------------------
// Extract the directory and path into a fixed buffer. This is
// needed as the directory and path are stored in separate string
// values.
//------------------------------------------------------------------
bool
FileSpec::GetPath(char *path, size_t max_path_length) const
{
    if (max_path_length == 0)
        return false;

    path[0] = '\0';
    const char *dirname = m_directory.AsCString();
    const char *filename = m_filename.AsCString();
    if (dirname)
    {
        if (filename && filename[0])
        {
            return (size_t)::snprintf (path, max_path_length, "%s/%s", dirname, filename) < max_path_length;
        }
        else
        {
            ::strncpy (path, dirname, max_path_length);
        }
    }
    else if (filename)
    {
        ::strncpy (path, filename, max_path_length);
    }
    else
    {
        return false;
    }

    // Any code paths that reach here assume that strncpy, or a similar function was called
    // where any remaining bytes will be filled with NULLs and that the string won't be
    // NULL terminated if it won't fit in the buffer.

    // If the last character is NULL, then all went well
    if (path[max_path_length-1] == '\0')
        return true;

        // Make sure the path is terminated, as it didn't fit into "path"
    path[max_path_length-1] = '\0';
    return false;
}

//------------------------------------------------------------------
// Returns a shared pointer to a data buffer that contains all or
// part of the contents of a file. The data is memory mapped and
// will lazily page in data from the file as memory is accessed.
// The data that is mappped will start "file_offset" bytes into the
// file, and "file_size" bytes will be mapped. If "file_size" is
// greater than the number of bytes available in the file starting
// at "file_offset", the number of bytes will be appropriately
// truncated. The final number of bytes that get mapped can be
// verified using the DataBuffer::GetByteSize() function.
//------------------------------------------------------------------
DataBufferSP
FileSpec::MemoryMapFileContents(off_t file_offset, size_t file_size) const
{
    DataBufferSP data_sp;
    auto_ptr<DataBufferMemoryMap> mmap_data(new DataBufferMemoryMap());
    if (mmap_data.get())
    {
        if (mmap_data->MemoryMapFromFileSpec (this, file_offset, file_size) >= file_size)
            data_sp.reset(mmap_data.release());
    }
    return data_sp;
}


//------------------------------------------------------------------
// Return the size in bytes that this object takes in memory. This
// returns the size in bytes of this object, not any shared string
// values it may refer to.
//------------------------------------------------------------------
size_t
FileSpec::MemorySize() const
{
    return m_filename.MemorySize() + m_directory.MemorySize();
}


size_t
FileSpec::ReadFileContents (off_t file_offset, void *dst, size_t dst_len) const
{
    size_t bytes_read = 0;
    char resolved_path[PATH_MAX];
    if (GetPath(resolved_path, sizeof(resolved_path)))
    {
        int fd = ::open (resolved_path, O_RDONLY, 0);
        if (fd != -1)
        {
            struct stat file_stats;
            if (::fstat (fd, &file_stats) == 0)
            {
                // Read bytes directly into our basic_string buffer
                if (file_stats.st_size > 0)
                {
                    off_t lseek_result = 0;
                    if (file_offset > 0)
                        lseek_result = ::lseek (fd, file_offset, SEEK_SET);

                    if (lseek_result == file_offset)
                    {
                        ssize_t n = ::read (fd, dst, dst_len);
                        if (n >= 0)
                            bytes_read = n;
                    }
                }
            }
        }
        close(fd);
    }
    return bytes_read;
}

//------------------------------------------------------------------
// Returns a shared pointer to a data buffer that contains all or
// part of the contents of a file. The data copies into a heap based
// buffer that lives in the DataBuffer shared pointer object returned.
// The data that is cached will start "file_offset" bytes into the
// file, and "file_size" bytes will be mapped. If "file_size" is
// greater than the number of bytes available in the file starting
// at "file_offset", the number of bytes will be appropriately
// truncated. The final number of bytes that get mapped can be
// verified using the DataBuffer::GetByteSize() function.
//------------------------------------------------------------------
DataBufferSP
FileSpec::ReadFileContents (off_t file_offset, size_t file_size) const
{
    DataBufferSP data_sp;
    char resolved_path[PATH_MAX];
    if (GetPath(resolved_path, sizeof(resolved_path)))
    {
        int fd = ::open (resolved_path, O_RDONLY, 0);
        if (fd != -1)
        {
            struct stat file_stats;
            if (::fstat (fd, &file_stats) == 0)
            {
                if (file_stats.st_size > 0)
                {
                    off_t lseek_result = 0;
                    if (file_offset > 0)
                        lseek_result = ::lseek (fd, file_offset, SEEK_SET);

                    if (lseek_result < 0)
                    {
                        // Get error from errno
                    }
                    else if (lseek_result == file_offset)
                    {
                        const size_t bytes_left = file_stats.st_size - file_offset;
                        size_t num_bytes_to_read = file_size;
                        if (num_bytes_to_read > bytes_left)
                            num_bytes_to_read = bytes_left;

                        std::auto_ptr<DataBufferHeap> data_heap_ap;
                        data_heap_ap.reset(new DataBufferHeap(num_bytes_to_read, '\0'));

                        if (data_heap_ap.get())
                        {
                            ssize_t bytesRead = ::read (fd, (void *)data_heap_ap->GetBytes(), data_heap_ap->GetByteSize());
                            if (bytesRead >= 0)
                            {
                                // Make sure we read exactly what we asked for and if we got
                                // less, adjust the array
                                if ((size_t)bytesRead < data_heap_ap->GetByteSize())
                                    data_heap_ap->SetByteSize(bytesRead);
                                data_sp.reset(data_heap_ap.release());
                            }
                        }
                    }
                }
            }
        }
        close(fd);
    }
    return data_sp;
}

bool
FileSpec::ReadFileLines (STLStringArray &lines)
{
    bool ret_val = false;
    lines.clear();

    std::string dir_str (m_directory.AsCString());
    std::string file_str (m_filename.AsCString());
    std::string full_name = dir_str + "/" + file_str;

    ifstream file_stream (full_name.c_str());

    if (file_stream)
    {
        std::string line;
        while (getline (file_stream, line))
          lines.push_back (line);
        ret_val = true;
    }

    return ret_val;
}
