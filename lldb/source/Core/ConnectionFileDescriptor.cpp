//===-- ConnectionFileDescriptor.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/ConnectionFileDescriptor.h"

// C Includes
#include <errno.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Timer.h"

using namespace lldb;
using namespace lldb_private;

ConnectionFileDescriptor::ConnectionFileDescriptor () :
    Connection(),
    m_fd (-1),
    m_fd_type (eFDTypeFile),
    m_udp_sockaddr (),
    m_udp_sockaddr_len (0),
    m_should_close_fd (false), 
    m_socket_timeout_usec(0)
{
    memset (&m_udp_sockaddr, 0, sizeof(m_udp_sockaddr));
    
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION |  LIBLLDB_LOG_OBJECT,
                                         "%p ConnectionFileDescriptor::ConnectionFileDescriptor ()",
                                         this);
}

ConnectionFileDescriptor::ConnectionFileDescriptor (int fd, bool owns_fd) :
    Connection(),
    m_fd (fd),
    m_fd_type (eFDTypeFile),
    m_udp_sockaddr (),
    m_udp_sockaddr_len (0),
    m_should_close_fd (owns_fd),
    m_socket_timeout_usec(0)
{
    memset (&m_udp_sockaddr, 0, sizeof(m_udp_sockaddr));
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION |  LIBLLDB_LOG_OBJECT,
                                         "%p ConnectionFileDescriptor::ConnectionFileDescriptor (fd = %i, owns_fd = %i)",
                                         this, 
                                         fd, 
                                         owns_fd);
}


ConnectionFileDescriptor::~ConnectionFileDescriptor ()
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION |  LIBLLDB_LOG_OBJECT,
                                         "%p ConnectionFileDescriptor::~ConnectionFileDescriptor ()",
                                         this);
    Disconnect (NULL);
}

bool
ConnectionFileDescriptor::IsConnected () const
{
    return m_fd >= 0;
}

ConnectionStatus
ConnectionFileDescriptor::Connect (const char *s, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION,
                                         "%p ConnectionFileDescriptor::Connect (url = '%s')",
                                         this, 
                                         s);

    if (s && s[0])
    {
        char *end = NULL;
        if (strstr(s, "listen://"))
        {
            // listen://HOST:PORT
            unsigned long listen_port = ::strtoul(s + strlen("listen://"), &end, 0);
            return SocketListen (listen_port, error_ptr);
        }
        else if (strstr(s, "unix-accept://"))
        {
            // unix://SOCKNAME
            return NamedSocketAccept (s + strlen("unix-accept://"), error_ptr);
        }
        else if (strstr(s, "connect://"))
        {
            return ConnectTCP (s + strlen("connect://"), error_ptr);
        }
        else if (strstr(s, "tcp://"))
        {
            return ConnectTCP (s + strlen("tcp://"), error_ptr);
        }
        else if (strstr(s, "udp://"))
        {
            return ConnectUDP (s + strlen("udp://"), error_ptr);
        }
        else if (strstr(s, "fd://"))
        {
            // Just passing a native file descriptor within this current process
            // that is already opened (possibly from a service or other source).
            s += strlen ("fd://");
            bool success = false;
            m_fd = Args::StringToSInt32 (s, -1, 0, &success);
            if (success)
            {
                // We have what looks to be a valid file descriptor, but we 
                // should make it is. We currently are doing this by trying to
                // get the flags from the file descriptor and making sure it 
                // isn't a bad fd.
                errno = 0;
                int flags = ::fcntl (m_fd, F_GETFL, 0);
                if (flags == -1 || errno == EBADF)
                {
                    if (error_ptr)
                        error_ptr->SetErrorStringWithFormat ("stale file descriptor: %s", s);
                    m_fd = -1;
                    return eConnectionStatusError;
                }
                else
                {
                    // Try and get a socket option from this file descriptor to 
                    // see if this is a socket and set m_is_socket accordingly.
                    int resuse;
                    bool is_socket = GetSocketOption (m_fd, SOL_SOCKET, SO_REUSEADDR, resuse) == 0;
                    if (is_socket)
                        m_fd_type = eFDTypeSocket;
                    m_should_close_fd = true;
                    return eConnectionStatusSuccess;
                }
            }
            
            if (error_ptr)
                error_ptr->SetErrorStringWithFormat ("invalid file descriptor: \"fd://%s\"", s);
            m_fd = -1;
            return eConnectionStatusError;
        }
        else if (strstr(s, "file://"))
        {
            // file:///PATH
            const char *path = s + strlen("file://");
            m_fd = ::open (path, O_RDWR);
            if (m_fd == -1)
            {
                if (error_ptr)
                    error_ptr->SetErrorToErrno();
                return eConnectionStatusError;
            }

            int flags = ::fcntl (m_fd, F_GETFL, 0);
            if (flags >= 0)
            {
                if ((flags & O_NONBLOCK) == 0)
                {
                    flags |= O_NONBLOCK;
                    ::fcntl (m_fd, F_SETFL, flags);
                }
            }
            m_should_close_fd = true;
            return eConnectionStatusSuccess;
        }
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat ("unsupported connection URL: '%s'", s);
        return eConnectionStatusError;
    }
    if (error_ptr)
        error_ptr->SetErrorString("invalid connect arguments");
    return eConnectionStatusError;
}

ConnectionStatus
ConnectionFileDescriptor::Disconnect (Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION,
                                 "%p ConnectionFileDescriptor::Disconnect ()",
                                 this);
    if (m_should_close_fd == false)
    {
        m_fd = -1;
        return eConnectionStatusSuccess;
    }
    return Close (m_fd, error_ptr);
}

size_t
ConnectionFileDescriptor::Read (void *dst, 
                                size_t dst_len, 
                                uint32_t timeout_usec,
                                ConnectionStatus &status, 
                                Error *error_ptr)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf ("%p ConnectionFileDescriptor::Read () ::read (fd = %i, dst = %p, dst_len = %zu)...",
                     this, m_fd, dst, dst_len);

    ssize_t bytes_read = 0;
    struct sockaddr_storage from;
    socklen_t from_len = sizeof(from);

    switch (m_fd_type)
    {
    case eFDTypeFile:       // Other FD requireing read/write
        status = BytesAvailable (timeout_usec, error_ptr);
        if (status == eConnectionStatusSuccess)
            bytes_read = ::read (m_fd, dst, dst_len);
        break;

    case eFDTypeSocket:     // Socket requiring send/recv
        if (SetSocketReceiveTimeout (timeout_usec))
        {
            status = eConnectionStatusSuccess;
            bytes_read = ::recv (m_fd, dst, dst_len, 0);
        }
        break;

    case eFDTypeSocketUDP:  // Unconnected UDP socket requiring sendto/recvfrom
        if (SetSocketReceiveTimeout (timeout_usec))
        {
            status = eConnectionStatusSuccess;
            ::memset (&from, 0, sizeof(from));
            bytes_read = ::recvfrom (m_fd, dst, dst_len, 0, (struct sockaddr *)&from, &from_len);
        }
        break;
    }

    if (status != eConnectionStatusSuccess)
        return 0;

    Error error;
    if (bytes_read == 0)
    {
        error.Clear(); // End-of-file.  Do not automatically close; pass along for the end-of-file handlers.
        status = eConnectionStatusEndOfFile;
    }
    else if (bytes_read < 0)
    {
        error.SetErrorToErrno();
    }
    else
    {
        error.Clear();
    }

    if (log)
        log->Printf ("%p ConnectionFileDescriptor::Read () ::read (fd = %i, dst = %p, dst_len = %zu) => %zi, error = %s",
                     this, 
                     m_fd, 
                     dst, 
                     dst_len, 
                     bytes_read, 
                     error.AsCString());

    if (error_ptr)
        *error_ptr = error;

    if (error.Fail())
    {
        uint32_t error_value = error.GetError();
        switch (error_value)
        {
        case EAGAIN:    // The file was marked for non-blocking I/O, and no data were ready to be read.
            status = eConnectionStatusSuccess;
            return 0;

        case EFAULT:    // Buf points outside the allocated address space.
        case EINTR:     // A read from a slow device was interrupted before any data arrived by the delivery of a signal.
        case EINVAL:    // The pointer associated with fildes was negative.
        case EIO:       // An I/O error occurred while reading from the file system.
                        // The process group is orphaned.
                        // The file is a regular file, nbyte is greater than 0,
                        // the starting position is before the end-of-file, and
                        // the starting position is greater than or equal to the
                        // offset maximum established for the open file
                        // descriptor associated with fildes.
        case EISDIR:    // An attempt is made to read a directory.
        case ENOBUFS:   // An attempt to allocate a memory buffer fails.
        case ENOMEM:    // Insufficient memory is available.
            status = eConnectionStatusError;
            break;  // Break to close....

        case ENOENT:    // no such file or directory
        case EBADF:     // fildes is not a valid file or socket descriptor open for reading.
        case ENXIO:     // An action is requested of a device that does not exist..
                        // A requested action cannot be performed by the device.
        case ECONNRESET:// The connection is closed by the peer during a read attempt on a socket.
        case ENOTCONN:  // A read is attempted on an unconnected socket.
            status = eConnectionStatusLostConnection;
            break;  // Break to close....

        case ETIMEDOUT: // A transmission timeout occurs during a read attempt on a socket.
            status = eConnectionStatusTimedOut;
            return 0;
        }

//      if (log)
//          error->Log(log, "::read ( %i, %p, %zu ) => %i", m_fd, dst, dst_len, bytesread);
        Close (m_fd, NULL);
        return 0;
    }
    return bytes_read;
}

size_t
ConnectionFileDescriptor::Write (const void *src, size_t src_len, ConnectionStatus &status, Error *error_ptr)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf ("%p ConnectionFileDescriptor::Write (src = %p, src_len = %zu)", this, src, src_len);

    if (!IsConnected ())
    {
        if (error_ptr)
            error_ptr->SetErrorString("not connected");
        status = eConnectionStatusNoConnection;
        return 0;
    }


    Error error;

    ssize_t bytes_sent = 0;

    switch (m_fd_type)
    {
        case eFDTypeFile:       // Other FD requireing read/write
            bytes_sent = ::write (m_fd, src, src_len);
            break;
            
        case eFDTypeSocket:     // Socket requiring send/recv
            bytes_sent = ::send (m_fd, src, src_len, 0);
            break;
            
        case eFDTypeSocketUDP:  // Unconnected UDP socket requiring sendto/recvfrom
            assert (m_udp_sockaddr_len != 0);
            bytes_sent = ::sendto (m_fd, 
                                   src, 
                                   src_len, 
                                   0, 
                                   (struct sockaddr *)&m_udp_sockaddr, 
                                   m_udp_sockaddr_len);
            break;
    }

    if (bytes_sent < 0)
        error.SetErrorToErrno ();
    else
        error.Clear ();

    if (log)
    {
        switch (m_fd_type)
        {
            case eFDTypeFile:       // Other FD requireing read/write
                log->Printf ("%p ConnectionFileDescriptor::Write()  ::write (fd = %i, src = %p, src_len = %zu) => %zi (error = %s)",
                             this, 
                             m_fd, 
                             src, 
                             src_len, 
                             bytes_sent, 
                             error.AsCString());
                break;
                
            case eFDTypeSocket:     // Socket requiring send/recv
                log->Printf ("%p ConnectionFileDescriptor::Write()  ::send (socket = %i, src = %p, src_len = %zu, flags = 0) => %zi (error = %s)",
                             this, 
                             m_fd, 
                             src, 
                             src_len, 
                             bytes_sent, 
                             error.AsCString());
                break;
                
            case eFDTypeSocketUDP:  // Unconnected UDP socket requiring sendto/recvfrom
                log->Printf ("%p ConnectionFileDescriptor::Write()  ::sendto (socket = %i, src = %p, src_len = %zu, flags = 0) => %zi (error = %s)",
                             this, 
                             m_fd, 
                             src, 
                             src_len, 
                             bytes_sent, 
                             error.AsCString());
                break;
        }
    }

    if (error_ptr)
        *error_ptr = error;

    if (error.Fail())
    {
        switch (error.GetError())
        {
        case EAGAIN:
        case EINTR:
            status = eConnectionStatusSuccess;
            return 0;

        case ECONNRESET:// The connection is closed by the peer during a read attempt on a socket.
        case ENOTCONN:  // A read is attempted on an unconnected socket.
            status = eConnectionStatusLostConnection;
            break;  // Break to close....

        default:
            status = eConnectionStatusError;
            break;  // Break to close....
        }

        Close (m_fd, NULL);
        return 0;
    }

    status = eConnectionStatusSuccess;
    return bytes_sent;
}

ConnectionStatus
ConnectionFileDescriptor::BytesAvailable (uint32_t timeout_usec, Error *error_ptr)
{
    LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf("%p ConnectionFileDescriptor::BytesAvailable (timeout_usec = %u)", this, timeout_usec);
    struct timeval *tv_ptr;
    struct timeval tv;
    if (timeout_usec == UINT32_MAX)
    {
        // Infinite wait...
        tv_ptr = NULL;
    }
    else
    {
        TimeValue time_value;
        time_value.OffsetWithMicroSeconds (timeout_usec);
        tv = time_value.GetAsTimeVal();
        tv_ptr = &tv;
    }

    while (IsConnected())
    {
        fd_set read_fds;
        FD_ZERO (&read_fds);
        FD_SET (m_fd, &read_fds);
        int nfds = m_fd + 1;
        
        Error error;


        log = lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION);
        if (log)
            log->Printf("%p ConnectionFileDescriptor::BytesAvailable()  ::select (nfds = %i, fd = %i, NULL, NULL, timeout = %p)...",
                        this, nfds, m_fd, tv_ptr);

        const int num_set_fds = ::select (nfds, &read_fds, NULL, NULL, tv_ptr);
        if (num_set_fds < 0)
            error.SetErrorToErrno();
        else
            error.Clear();

        log = lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION);
        if (log)
            log->Printf("%p ConnectionFileDescriptor::BytesAvailable()  ::select (nfds = %i, fd = %i, NULL, NULL, timeout = %p) => %d, error = %s",
                        this, nfds, m_fd, tv_ptr, num_set_fds, error.AsCString());

        if (error_ptr)
            *error_ptr = error;

        if (error.Fail())
        {
            switch (error.GetError())
            {
            case EBADF:     // One of the descriptor sets specified an invalid descriptor.
                return eConnectionStatusLostConnection;

            case EINVAL:    // The specified time limit is invalid. One of its components is negative or too large.
            default:        // Other unknown error
                return eConnectionStatusError;

            case EAGAIN:    // The kernel was (perhaps temporarily) unable to
                            // allocate the requested number of file descriptors,
                            // or we have non-blocking IO
            case EINTR:     // A signal was delivered before the time limit
                            // expired and before any of the selected events
                            // occurred.
                break;      // Lets keep reading to until we timeout
            }
        }
        else if (num_set_fds == 0)
        {
            return eConnectionStatusTimedOut;
        }
        else if (num_set_fds > 0)
        {
            return eConnectionStatusSuccess;
        }
    }

    if (error_ptr)
        error_ptr->SetErrorString("not connected");
    return eConnectionStatusLostConnection;
}

ConnectionStatus
ConnectionFileDescriptor::Close (int& fd, Error *error_ptr)
{
    if (error_ptr)
        error_ptr->Clear();
    bool success = true;
    if (fd >= 0)
    {
        lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION,
                                             "%p ConnectionFileDescriptor::Close (fd = %i)",
                                             this,
                                             fd);

        success = ::close (fd) == 0;
        if (!success && error_ptr)
        {
            // Only set the error if we have been asked to since something else
            // might have caused us to try and shut down the connection and may
            // have already set the error.
            error_ptr->SetErrorToErrno();
        }
        fd = -1;
    }
    m_fd_type = eFDTypeFile;
    if (success)
        return eConnectionStatusSuccess;
    else
        return eConnectionStatusError;
}

ConnectionStatus
ConnectionFileDescriptor::NamedSocketAccept (const char *socket_name, Error *error_ptr)
{
    ConnectionStatus result = eConnectionStatusError;
    struct sockaddr_un saddr_un;

    m_fd_type = eFDTypeSocket;
    
    int listen_socket = ::socket (AF_UNIX, SOCK_STREAM, 0);
    if (listen_socket == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        return eConnectionStatusError;
    }

    saddr_un.sun_family = AF_UNIX;
    ::strncpy(saddr_un.sun_path, socket_name, sizeof(saddr_un.sun_path) - 1);
    saddr_un.sun_path[sizeof(saddr_un.sun_path) - 1] = '\0';
#if defined(__APPLE__) || defined(__FreeBSD__)
    saddr_un.sun_len = SUN_LEN (&saddr_un);
#endif

    if (::bind (listen_socket, (struct sockaddr *)&saddr_un, SUN_LEN (&saddr_un)) == 0) 
    {
        if (::listen (listen_socket, 5) == 0) 
        {
            m_fd = ::accept (listen_socket, NULL, 0);
            if (m_fd > 0)
            {
                m_should_close_fd = true;

                if (error_ptr)
                    error_ptr->Clear();
                result = eConnectionStatusSuccess;
            }
        }
    }
    
    if (result != eConnectionStatusSuccess)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
    }
    // We are done with the listen port
    Close (listen_socket, NULL);
    return result;
}

ConnectionStatus
ConnectionFileDescriptor::NamedSocketConnect (const char *socket_name, Error *error_ptr)
{
    Close (m_fd, NULL);
    m_fd_type = eFDTypeSocket;

    // Open the socket that was passed in as an option
    struct sockaddr_un saddr_un;
    m_fd = ::socket (AF_UNIX, SOCK_STREAM, 0);
    if (m_fd == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        return eConnectionStatusError;
    }

    saddr_un.sun_family = AF_UNIX;
    ::strncpy(saddr_un.sun_path, socket_name, sizeof(saddr_un.sun_path) - 1);
    saddr_un.sun_path[sizeof(saddr_un.sun_path) - 1] = '\0';
#if defined(__APPLE__) || defined(__FreeBSD__)
    saddr_un.sun_len = SUN_LEN (&saddr_un);
#endif

    if (::connect (m_fd, (struct sockaddr *)&saddr_un, SUN_LEN (&saddr_un)) < 0) 
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        Close (m_fd, NULL);
        return eConnectionStatusError;
    }
    if (error_ptr)
        error_ptr->Clear();
    return eConnectionStatusSuccess;
}

ConnectionStatus
ConnectionFileDescriptor::SocketListen (uint16_t listen_port_num, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION,
                                 "%p ConnectionFileDescriptor::SocketListen (port = %i)",
                                 this, listen_port_num);

    Close (m_fd, NULL);
    m_fd_type = eFDTypeSocket;
    int listen_port = ::socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_port == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        return eConnectionStatusError;
    }

    // enable local address reuse
    SetSocketOption (listen_port, SOL_SOCKET, SO_REUSEADDR, 1);

    struct sockaddr_in sa;
    ::memset (&sa, 0, sizeof sa);
    sa.sin_family = AF_INET;
    sa.sin_port = htons (listen_port_num);
    sa.sin_addr.s_addr = htonl (INADDR_ANY);

    int err = ::bind (listen_port, (struct sockaddr *) &sa, sizeof(sa));
    if (err == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        Close (listen_port, NULL);
        return eConnectionStatusError;
    }

    err = ::listen (listen_port, 1);
    if (err == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        Close (listen_port, NULL);
        return eConnectionStatusError;
    }

    m_fd = ::accept (listen_port, NULL, 0);
    if (m_fd == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        Close (listen_port, NULL);
        return eConnectionStatusError;
    }

    // We are done with the listen port
    Close (listen_port, NULL);

    m_should_close_fd = true;

    // Keep our TCP packets coming without any delays.
    SetSocketOption (m_fd, IPPROTO_TCP, TCP_NODELAY, 1);
    if (error_ptr)
        error_ptr->Clear();
    return eConnectionStatusSuccess;
}

ConnectionStatus
ConnectionFileDescriptor::ConnectTCP (const char *host_and_port, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION,
                                 "%p ConnectionFileDescriptor::ConnectTCP (host/port = %s)",
                                 this, host_and_port);
    Close (m_fd, NULL);
    m_fd_type = eFDTypeSocket;

    RegularExpression regex ("([^:]+):([0-9]+)");
    if (regex.Execute (host_and_port, 2) == false)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("invalid host:port specification: '%s'", host_and_port);
        return eConnectionStatusError;
    }
    std::string host_str;
    std::string port_str;
    if (regex.GetMatchAtIndex (host_and_port, 1, host_str) == false ||
        regex.GetMatchAtIndex (host_and_port, 2, port_str) == false)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("invalid host:port specification '%s'", host_and_port);
        return eConnectionStatusError;
    }

    int32_t port = Args::StringToSInt32 (port_str.c_str(), INT32_MIN);
    if (port == INT32_MIN)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("invalid port '%s'", port_str.c_str());
        return eConnectionStatusError;
    }
    // Create the socket
    m_fd = ::socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_fd == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        return eConnectionStatusError;
    }

    m_should_close_fd = true;

    // Enable local address reuse
    SetSocketOption (m_fd, SOL_SOCKET, SO_REUSEADDR, 1);

    struct sockaddr_in sa;
    ::memset (&sa, 0, sizeof (sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons (port);

    int inet_pton_result = ::inet_pton (AF_INET, host_str.c_str(), &sa.sin_addr);

    if (inet_pton_result <= 0)
    {
        struct hostent *host_entry = gethostbyname (host_str.c_str());
        if (host_entry)
            host_str = ::inet_ntoa (*(struct in_addr *)*host_entry->h_addr_list);
        inet_pton_result = ::inet_pton (AF_INET, host_str.c_str(), &sa.sin_addr);
        if (inet_pton_result <= 0)
        {

            if (error_ptr)
            {
                if (inet_pton_result == -1)
                    error_ptr->SetErrorToErrno();
                else
                    error_ptr->SetErrorStringWithFormat("invalid host string: '%s'", host_str.c_str());
            }
            Close (m_fd, NULL);
            return eConnectionStatusError;
        }
    }

    if (-1 == ::connect (m_fd, (const struct sockaddr *)&sa, sizeof(sa)))
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        Close (m_fd, NULL);
        return eConnectionStatusError;
    }

    // Keep our TCP packets coming without any delays.
    SetSocketOption (m_fd, IPPROTO_TCP, TCP_NODELAY, 1);
    if (error_ptr)
        error_ptr->Clear();
    return eConnectionStatusSuccess;
}

ConnectionStatus
ConnectionFileDescriptor::ConnectUDP (const char *host_and_port, Error *error_ptr)
{
    lldb_private::LogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION,
                                         "%p ConnectionFileDescriptor::ConnectUDP (host/port = %s)",
                                         this, host_and_port);
    Close (m_fd, NULL);
    m_fd_type = eFDTypeSocketUDP;
    
    RegularExpression regex ("([^:]+):([0-9]+)");
    if (regex.Execute (host_and_port, 2) == false)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("invalid host:port specification: '%s'", host_and_port);
        return eConnectionStatusError;
    }
    std::string host_str;
    std::string port_str;
    if (regex.GetMatchAtIndex (host_and_port, 1, host_str) == false ||
        regex.GetMatchAtIndex (host_and_port, 2, port_str) == false)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("invalid host:port specification '%s'", host_and_port);
        return eConnectionStatusError;
    }
    
    int32_t port = Args::StringToSInt32 (port_str.c_str(), INT32_MIN);
    if (port == INT32_MIN)
    {
        if (error_ptr)
            error_ptr->SetErrorStringWithFormat("invalid port '%s'", port_str.c_str());
        return eConnectionStatusError;
    }
    // Create the socket
    m_fd = ::socket (AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_fd == -1)
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        return eConnectionStatusError;
    }
    
    m_should_close_fd = true;
    
    // Enable local address reuse
    SetSocketOption (m_fd, SOL_SOCKET, SO_REUSEADDR, 1);
    
    struct sockaddr_in sa;
    ::memset (&sa, 0, sizeof (sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons (port);
    
    int inet_pton_result = ::inet_pton (AF_INET, host_str.c_str(), &sa.sin_addr);
    
    if (inet_pton_result <= 0)
    {
        struct hostent *host_entry = gethostbyname (host_str.c_str());
        if (host_entry)
            host_str = ::inet_ntoa (*(struct in_addr *)*host_entry->h_addr_list);
        inet_pton_result = ::inet_pton (AF_INET, host_str.c_str(), &sa.sin_addr);
        if (inet_pton_result <= 0)
        {
            
            if (error_ptr)
            {
                if (inet_pton_result == -1)
                    error_ptr->SetErrorToErrno();
                else
                    error_ptr->SetErrorStringWithFormat("invalid host string: '%s'", host_str.c_str());
            }
            Close (m_fd, NULL);
            return eConnectionStatusError;
        }
    }
    
    if (-1 == ::connect (m_fd, (const struct sockaddr *)&sa, sizeof(sa)))
    {
        if (error_ptr)
            error_ptr->SetErrorToErrno();
        Close (m_fd, NULL);
        return eConnectionStatusError;
    }
    
    // Keep our TCP packets coming without any delays.
    SetSocketOption (m_fd, IPPROTO_TCP, TCP_NODELAY, 1);
    if (error_ptr)
        error_ptr->Clear();
    return eConnectionStatusSuccess;
}

#if defined(__MINGW32__) || defined(__MINGW64__)
typedef const char * set_socket_option_arg_type;
typedef char * get_socket_option_arg_type;
#else // #if defined(__MINGW32__) || defined(__MINGW64__)
typedef const void * set_socket_option_arg_type;
typedef void * get_socket_option_arg_type;
#endif // #if defined(__MINGW32__) || defined(__MINGW64__)

int
ConnectionFileDescriptor::GetSocketOption(int fd, int level, int option_name, int &option_value)
{
    get_socket_option_arg_type option_value_p = static_cast<get_socket_option_arg_type>(&option_value);
    socklen_t option_value_size = sizeof(int);
	return ::getsockopt(fd, level, option_name, option_value_p, &option_value_size);
}

int
ConnectionFileDescriptor::SetSocketOption(int fd, int level, int option_name, int option_value)
{
    set_socket_option_arg_type option_value_p = static_cast<get_socket_option_arg_type>(&option_value);
	return ::setsockopt(fd, level, option_name, option_value_p, sizeof(option_value));
}

bool
ConnectionFileDescriptor::SetSocketReceiveTimeout (uint32_t timeout_usec)
{
    switch (m_fd_type)
    {
        case eFDTypeFile:       // Other FD requireing read/write
            break;
            
        case eFDTypeSocket:     // Socket requiring send/recv
        case eFDTypeSocketUDP:  // Unconnected UDP socket requiring sendto/recvfrom
        {
            // Check in case timeout for m_fd has already been set to this value
            if (timeout_usec == m_socket_timeout_usec)
                return true;
            //printf ("ConnectionFileDescriptor::SetSocketReceiveTimeout (timeout_usec = %u)\n", timeout_usec);

            struct timeval timeout;
            timeout.tv_sec = timeout_usec / TimeValue::MicroSecPerSec;
            timeout.tv_usec = timeout_usec % TimeValue::MicroSecPerSec;
            if (::setsockopt (m_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0)
            {
                m_socket_timeout_usec = timeout_usec;
                return true;
            }
        }
    }
    return false;
}


