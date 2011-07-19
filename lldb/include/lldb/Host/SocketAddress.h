//===-- SocketAddress.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SocketAddress_h_
#define liblldb_SocketAddress_h_

// C Includes
#include <stdint.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>

// C++ Includes
// Other libraries and framework includes
// Project includes

namespace lldb_private {

class SocketAddress
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SocketAddress();
    SocketAddress(const SocketAddress& rhs);
    ~SocketAddress();

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const SocketAddress&
    operator=(const SocketAddress& rhs);

    const SocketAddress&
    operator=(const struct addrinfo *addr_info);

    void
    Clear ();

    socklen_t
    GetLength () const;

    static socklen_t
    GetMaxLength ();

    void
    SetLength (socklen_t len);
    
    sa_family_t
    GetFamily () const;

    void
    SetFamily (sa_family_t family);

    in_port_t
    GetPort () const;

    bool
    IsValid () const;

    struct sockaddr &
    sockaddr ()
    {
        return m_socket_addr.sa;
    }

    const struct sockaddr &
    sockaddr () const
    {
        return m_socket_addr.sa;
    }
    
    struct sockaddr_in &
    sockaddr_in ()
    {
        return m_socket_addr.sa_ipv4;
    }
    
    const struct sockaddr_in &
    sockaddr_in () const
    {
        return m_socket_addr.sa_ipv4;
    }
    
    struct sockaddr_in6 &
    sockaddr_in6 ()
    {
        return m_socket_addr.sa_ipv6;
    }
    
    const struct sockaddr_in6 &
    sockaddr_in6 () const
    {
        return m_socket_addr.sa_ipv6;
    }
    
    struct sockaddr_storage &
    sockaddr_storage ()
    {
        return m_socket_addr.sa_storage;
    }

    
    const struct sockaddr_storage &
    sockaddr_storage () const
    {
        return m_socket_addr.sa_storage;
    }
    //------------------------------------------------------------------
    // Conversion operators to allow getting the contents of this class
    // as a subclass
    //------------------------------------------------------------------
    
    operator struct sockaddr * ()
    {
        return &m_socket_addr.sa;
    }
    
    operator const struct sockaddr * () const
    {
        return &m_socket_addr.sa;
    }

    operator struct sockaddr_in * ()
    {
        return &m_socket_addr.sa_ipv4;
    }
    
    operator const struct sockaddr_in * () const
    {
        return &m_socket_addr.sa_ipv4;
    }

    operator struct sockaddr_in6 * ()
    {
        return &m_socket_addr.sa_ipv6;
    }
    
    operator const struct sockaddr_in6 * () const
    {
        return &m_socket_addr.sa_ipv6;
    }

    operator const struct sockaddr_storage * () const
    {
        return &m_socket_addr.sa_storage;
    }

    operator struct sockaddr_storage * ()
    {
        return &m_socket_addr.sa_storage;
    }

protected:
    typedef union sockaddr_tag
    {
        struct sockaddr         sa;
        struct sockaddr_in      sa_ipv4;
        struct sockaddr_in6     sa_ipv6;
        struct sockaddr_storage sa_storage;
    } sockaddr_t;

    //------------------------------------------------------------------
    // Classes that inherit from SocketAddress can see and modify these
    //------------------------------------------------------------------
    sockaddr_t m_socket_addr;
};


} // namespace lldb_private


#endif  // liblldb_SocketAddress_h_
