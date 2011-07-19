//===-- SocketAddress.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/SocketAddress.h"
#include <stddef.h>

// C Includes
#include <string.h>

// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;

//----------------------------------------------------------------------
// SocketAddress constructor
//----------------------------------------------------------------------
SocketAddress::SocketAddress()
{
    Clear ();
}

//----------------------------------------------------------------------
// SocketAddress copy constructor
//----------------------------------------------------------------------
SocketAddress::SocketAddress (const SocketAddress& rhs) :
    m_socket_addr (rhs.m_socket_addr)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SocketAddress::~SocketAddress()
{
}

void
SocketAddress::Clear ()
{
    memset (&m_socket_addr, 0, sizeof(m_socket_addr));
}

bool
SocketAddress::IsValid () const
{
    return GetLength () != 0;
}

socklen_t
SocketAddress::GetLength () const
{
    return m_socket_addr.sa.sa_len;
}

socklen_t
SocketAddress::GetMaxLength ()
{
    return sizeof (sockaddr_t);
}

void
SocketAddress::SetLength (socklen_t len)
{
    m_socket_addr.sa.sa_len = len;
}

sa_family_t
SocketAddress::GetFamily () const
{
    return m_socket_addr.sa.sa_family;
}

void
SocketAddress::SetFamily (sa_family_t family)
{
    m_socket_addr.sa.sa_family = family;
}

in_port_t
SocketAddress::GetPort () const
{
    switch (GetFamily())
    {
        case AF_INET:   return m_socket_addr.sa_ipv4.sin_port;
        case AF_INET6:  return m_socket_addr.sa_ipv6.sin6_port;
    }
    return 0;
}

//----------------------------------------------------------------------
// SocketAddress assignment operator
//----------------------------------------------------------------------
const SocketAddress&
SocketAddress::operator=(const SocketAddress& rhs)
{
    if (this != &rhs)
        m_socket_addr = rhs.m_socket_addr;
    return *this;
}

const SocketAddress&
SocketAddress::operator=(const struct addrinfo *addr_info)
{
    Clear();
    if (addr_info && 
        addr_info->ai_addr &&
        addr_info->ai_addrlen > 0&& 
        addr_info->ai_addrlen <= sizeof m_socket_addr)
    {
        ::memcpy (&m_socket_addr, 
                  addr_info->ai_addr, 
                  addr_info->ai_addrlen);
    }
    return *this;
}



